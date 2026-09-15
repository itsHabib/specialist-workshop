"""Bounded diagnostic: deployable validation retries, never oracle selection.

Defaults to development cases and local base inference. API policies require
OPENAI_API_KEY. Output includes task data; keep private runs in .state/.
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
from workshop.experiment import Local, checkpoint, read_run, MODEL_ID, MODEL_REVISION
from workshop.domain import digest

RATES = {'gpt-6-astra': (10, 1, 50), 'gpt-5.6-luna': (.2, .02, 1.2)}
REPAIR = 'The output failed the task validator: {error}\nReturn a corrected JSON object only, following the original task. No explanation.'


def validated(package, value, raw):
    # Expected output is deliberately absent from this deployable feedback path.
    return packages.score(package, value, decode(raw)['content'])


def cost(model, usage):
    inp, cached, out = RATES[model]
    hit = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
    writes = usage.get('input_tokens_details', {}).get('cache_write_tokens', 0)
    return ((usage['input_tokens'] - hit) * inp + writes * inp * .25 + hit * cached + usage['output_tokens'] * out) / 1e6


def policy(package, value, predict, max_attempts=3):
    messages = packages.messages(package, value)
    attempts = []
    error = None
    start = time.perf_counter()
    for _ in range(max_attempts):
        try:
            prediction = predict(messages)
        except Exception as exc:
            # Keep earlier paid attempts on budget/transport failures; never log credentials.
            error = type(exc).__name__
            break
        verdict = validated(package, value, prediction['raw'])
        attempts.append(dict(**prediction, validation=verdict))
        error = prediction.get('error')
        if verdict['valid'] or error:
            break
        messages += [dict(role='assistant', content=prediction['raw']),
                     dict(role='user', content=REPAIR.format(error=verdict['error']))]
    return dict(attempts=attempts, elapsed_seconds=time.perf_counter()-start,
                error=error,stop_reason='error' if error else 'valid' if attempts[-1]['validation']['valid'] else 'attempt_limit')


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


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package",required=True,help="Imported package reference")
    parser.add_argument("--checkpoint",help="Completed training/adoption run; required for local-specialist")
    parser.add_argument("--policies",nargs="+",choices=[*RATES,"local-base","local-specialist"],default=["local-base"])
    parser.add_argument("--split",choices=["development","final"],default="development")
    parser.add_argument("--attempts",type=int,choices=[1,2,3],default=3)
    parser.add_argument("--api-budget-usd",type=float,default=3)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if not math.isfinite(args.api_budget_usd) or args.api_budget_usd<=0:
        parser.error('--api-budget-usd must be finite and positive')
    if len(set(args.policies))!=len(args.policies):parser.error('Duplicate policies')
    if 'local-specialist' in args.policies and not args.checkpoint:
        parser.error('--checkpoint is required for local-specialist')
    return args


def main(argv=None):
    args=arguments(argv)
    store.initialize()
    package=packages.load(args.package)
    source=args.checkpoint
    # Validate local weights before any paid requests or output-directory writes.
    adapter=checkpoint(package,source) if 'local-specialist' in args.policies else None
    if any(name in RATES for name in args.policies) and not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('Set OPENAI_API_KEY for API policies')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    cases=getattr(package,args.split)
    record=dict(status='running',started_at=time.time(),source_checkpoint=source,
                package=packages.summary(package),cases=[c.model_dump() for c in cases],
                scope='Explicit final evaluation' if args.split=='final' else 'Development iteration',
                evaluation_split=args.split,evaluation_hash=digest([c.model_dump() for c in cases]),
                selected_policies=args.policies,
                feedback='schema and context validation only; stop at first valid output',
                max_attempts=args.attempts,local_max_tokens=260,api_max_output_tokens=2048,api_reasoning='low',
                api_budget_usd=args.api_budget_usd,prices_checked='2026-09-15',rates_per_million=RATES,
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
            if spent_reservations+reserve>args.api_budget_usd: raise RuntimeError('API reservation budget exhausted')
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
    for name in args.policies:
        startup=time.perf_counter()
        local=None
        lock=None
        if name.startswith('local'):
            lock=(store.STATE/'accelerator.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            local=Local(adapter if name=='local-specialist' else None)
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
        if name.startswith('local'):result.update(model=MODEL_ID,revision=MODEL_REVISION)
        if name=='local-specialist':result['adapter_manifest']=read_run(source)['adapter_manifest']
        record['policies'][name]=result
        for case in cases:
            row=policy(package,case.input,predict,args.attempts)
            # Score only after final selection. Scores never enter model messages.
            for attempt in row['attempts']:
                attempt['score']=packages.score(package,case.input,decode(attempt['raw'])['content'],case.expected)
            row['id']=case.id
            result['rows'].append(row)
            if row.get('error'):
                record.update(status='failed',finished_at=time.time(),api_reserved_usd=spent_reservations,error=row['error'])
                save()
                if lock:lock.close()
                raise SystemExit('Comparison stopped; partial attempts saved in run.json')
            result['one_shot']=summarize(result['rows'],1)
            result['retry']=summarize(result['rows'],args.attempts)
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
