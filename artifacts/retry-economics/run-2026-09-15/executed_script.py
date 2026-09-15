"""Bounded diagnostic: deployable validation retries, never oracle selection.

Run with OPENAI_API_KEY in the environment. Writes public synthetic task data only.
Existing final cases are exposed diagnostics, not fresh qualification evidence.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop import packages, store
from workshop.completion import decode
from workshop.experiment import Local, checkpoint, read_run

RATES = {'gpt-6-astra': (10, 1, 50), 'gpt-5.6-luna': (.2, .02, 1.2)}
REPAIR = 'The output failed the task validator: {error}\nReturn a corrected JSON object only, following the original task. No explanation.'


def validated(package, value, raw):
    # Expected output is deliberately absent from this deployable feedback path.
    return packages.score(package, value, decode(raw)['content'])


def cost(model, usage):
    inp, cached, out = RATES[model]
    hit = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
    return ((usage['input_tokens'] - hit) * inp + hit * cached + usage['output_tokens'] * out) / 1e6


def policy(package, value, predict, max_attempts=3):
    messages = packages.messages(package, value)
    attempts = []
    start = time.perf_counter()
    for _ in range(max_attempts):
        prediction = predict(messages)
        verdict = validated(package, value, prediction['raw'])
        attempts.append(dict(**prediction, validation=verdict))
        if verdict['valid'] or prediction.get('error'):
            break
        messages += [dict(role='assistant', content=prediction['raw']),
                     dict(role='user', content=REPAIR.format(error=verdict['error']))]
    return dict(attempts=attempts, elapsed_seconds=time.perf_counter()-start,
                stop_reason='valid' if attempts[-1]['validation']['valid'] else 'budget_or_error')


def summarize(rows, max_attempts):
    selected = [r['attempts'][:max_attempts] for r in rows]
    successes = sum(a[-1]['score']['accepted'] for a in selected)
    costs = [a['cost_usd'] for group in selected for a in group]
    total = sum(costs) if all(c is not None for c in costs) else None
    # First-attempt timing excludes local startup; full task includes validation.
    times = [r['elapsed_seconds'] if max_attempts > 1 else a[0]['elapsed_seconds'] for r,a in zip(rows,selected)]
    return dict(tasks=len(rows), successful=successes, attempts=sum(map(len,selected)),
                total_cost_usd=total, cost_per_success_usd=total/successes if total is not None and successes else None,
                median_seconds=statistics.median(times), p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],
                valid_but_wrong=sum(a[-1]['validation']['valid'] and not a[-1]['score']['accepted'] for a in selected),
                inference_seconds=sum(a['elapsed_seconds'] for group in selected for a in group))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source='pkg-f0a24974e753'
    package=packages.load(read_run(source)['package'])
    cases=package.final[::2]
    record=dict(status='running',started_at=time.time(),source_checkpoint=source,
                package=packages.summary(package),cases=[c.model_dump() for c in cases],
                scope='Previously exposed final subset at even indices; diagnostic only',
                feedback='schema and context validation only; stop at first valid output',
                max_attempts=3,local_max_tokens=260,api_max_output_tokens=2048,api_reasoning='low',
                api_budget_usd=3,prices_checked='2026-09-15',rates_per_million=RATES,
                price_sources=['https://developers.openai.com/api/docs/models/'+m for m in RATES],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),policies={})
    store.atomic_json(args.output/'package.json',package.model_dump())
    def save(): store.atomic_json(args.output/'run.json',record)
    save()  # Freeze subset, policy and limits before the first model request.
    spent_reservations=0
    def api(model):
        def predict(messages):
            nonlocal spent_reservations
            import httpx
            # Byte length upper-bounds ordinary text tokens; allow 1024 framing tokens.
            bound=len(json.dumps(messages).encode())+1024
            reserve=(bound*RATES[model][0]*1.25 + 2048*RATES[model][2])/1e6
            if spent_reservations+reserve>3: raise RuntimeError('API reservation budget exhausted')
            spent_reservations+=reserve  # Retain reservation even on unknown billing/error.
            start=time.perf_counter()
            response=httpx.post('https://api.openai.com/v1/responses',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']},
                json=dict(model=model,input=messages,max_output_tokens=2048,reasoning={'effort':'low'},store=False),timeout=90)
            if response.status_code!=200:
                return dict(raw='',error='HTTP '+str(response.status_code),cost_usd=None,elapsed_seconds=time.perf_counter()-start)
            body=response.json()
            raw=''.join(c['text'] for o in body.get('output',[]) for c in o.get('content',[]) if c.get('type')=='output_text')
            usage=body.get('usage')
            return dict(raw=raw,usage=usage,cost_usd=cost(model,usage) if usage else None,
                        elapsed_seconds=time.perf_counter()-start,model=body['model'],response_id=body['id'],status=body['status'])
        return predict
    for name in ['gpt-6-astra','gpt-5.6-luna','local-base','local-specialist']:
        startup=time.perf_counter()
        local=None
        lock=None
        if name.startswith('local'):
            lock=(store.STATE/'accelerator.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            adapter=checkpoint(package,source) if name=='local-specialist' else None
            local=Local(adapter)
            def predict(messages):
                import mlx.core as mx
                from mlx_lm import generate
                from mlx_lm.sample_utils import make_sampler
                mx.random.seed(29)
                prompt=local.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                tokens=len(local.tokenizer.encode(prompt))
                if tokens>4000: raise RuntimeError('Local context budget exhausted')
                start=time.perf_counter()
                raw=generate(local.model,local.tokenizer,prompt=prompt,max_tokens=260,sampler=make_sampler(temp=0),verbose=False)
                return dict(raw=raw,input_tokens=tokens,output_tokens=len(local.tokenizer.encode(raw)),cost_usd=None,elapsed_seconds=time.perf_counter()-start)
        else: predict=api(name)
        result=dict(startup_seconds=time.perf_counter()-startup,rows=[])
        if name=='local-specialist':result['adapter_manifest']=read_run(source)['adapter_manifest']
        record['policies'][name]=result
        for case in cases:
            row=policy(package,case.input,predict,1 if name=='gpt-6-astra' else 3)
            # Score only after final selection. Scores never enter model messages.
            for attempt in row['attempts']:
                attempt['score']=packages.score(package,case.input,decode(attempt['raw'])['content'],case.expected)
            row['id']=case.id
            result['rows'].append(row)
            result['one_shot']=summarize(result['rows'],1)
            result['retry']=summarize(result['rows'],3)
            save()
            print(name,case.id,len(row['attempts']),row['attempts'][-1]['score']['accepted'],flush=True)
        if local:
            del predict,local
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()
            lock.close()
    record.update(status='completed',finished_at=time.time(),api_reserved_usd=spent_reservations)
    save()


if __name__=='__main__':main()
