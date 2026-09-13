"""One bounded analytics experiment; shared numerical lock, exact artifacts."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from workshop.analytics import METRICS,MODEL_ID,MODEL_REVISION,Plan,canonical,execute,grade,messages,validate_plan
from workshop.domain import check_training_length
from workshop import store

DEST=store.ROOT/'experiments/analytics'
LOCAL=store.STATE/'analytics'


def local_load(adapter=None):
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    return load(snapshot_download(MODEL_ID,revision=MODEL_REVISION),adapter_path=adapter)


def predict(model,case,structured=False):
    import mlx.core as mx
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    network,tokenizer=model
    mx.random.seed(29)
    prompt=tokenizer.apply_chat_template(messages(case,structured),tokenize=False,add_generation_prompt=True)
    tokens=len(tokenizer.encode(prompt))
    if tokens>4000:raise ValueError('Input exceeds 4000 tokens')
    start=time.perf_counter()
    raw=generate(network,tokenizer,prompt=prompt,max_tokens=260,sampler=make_sampler(temp=0),verbose=False)
    return dict(raw=raw,elapsed_ms=round((time.perf_counter()-start)*1000),input_tokens=tokens)


def frontier(case):
    prompt=messages(case)
    command=['claude','-p','--output-format','json','--tools','','--strict-mcp-config','--mcp-config','{"mcpServers":{}}',
             '--no-session-persistence','--effort','low','--setting-sources','','--disable-slash-commands','--max-budget-usd','0.50',
             '--system-prompt',prompt[0]['content']]
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        process=subprocess.run(command,input=prompt[1]['content'],text=True,capture_output=True,cwd=tmp,timeout=180)
    if process.returncode:raise RuntimeError(process.stderr[-500:])
    envelope=json.loads(process.stdout)
    if envelope.get('is_error'):raise RuntimeError(str(envelope.get('result')))
    return dict(raw=envelope['result'],elapsed_ms=round((time.perf_counter()-start)*1000),
                usage={k:envelope.get(k) for k in ('modelUsage','usage','total_cost_usd','duration_ms')})


def check_frozen():
    for name,expected in store.read_json(DEST/'input-manifest.json').items():
        if hashlib.sha256((DEST/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen input changed: {name}')


def train(lock_fd):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    if (LOCAL/'training.json').exists():raise ValueError('A training record exists; preserve it')
    model_path=snapshot_download(MODEL_ID,revision=MODEL_REVISION)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    data=LOCAL/'data';data.mkdir(parents=True,exist_ok=True)
    lengths=[]
    for name,source in [('train','train'),('valid','development')]:
        with (data/f'{name}.jsonl').open('w') as stream:
            for row in store.read_json(DEST/f'{source}.json'):
                validate_plan(json.dumps(row['expected']),row['context'])
                msgs=messages(row,output=row['expected'])
                check_training_length(tokenizer,msgs,row['id'],limit=2048)
                lengths.append(len(tokenizer.apply_chat_template(msgs,tokenize=True,return_dict=False)))
                stream.write(json.dumps(dict(messages=msgs))+'\n')
    adapter=LOCAL/'adapter'
    command=[sys.executable,'-m','mlx_lm','lora','--model',model_path,'--train','--data',str(data),'--adapter-path',str(adapter),
             '--iters','120','--batch-size','1','--num-layers','8','--learning-rate','0.00002','--max-seq-length','2048',
             '--mask-prompt','--steps-per-report','10','--steps-per-eval','40','--val-batches','6','--save-every','120','--seed','29']
    record=dict(status='running',model=MODEL_ID,revision=MODEL_REVISION,started_at=time.time(),max_training_tokens=max(lengths),
                recipe=dict(iterations=120,batch_size=1,layers=8,learning_rate=2e-5,seed=29,max_seq_length=2048,mask_prompt=True),
                input_manifest=store.read_json(DEST/'input-manifest.json'))
    store.atomic_json(LOCAL/'training.json',record)
    try:
        with (LOCAL/'training.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400,pass_fds=(lock_fd,))
        record.update(status='completed',adapter_sha256=hashlib.sha256((adapter/'adapters.safetensors').read_bytes()).hexdigest())
    except Exception as exc:
        record.update(status='failed',error=str(exc));raise
    finally:
        record.update(finished_at=time.time())
        store.atomic_json(LOCAL/'training.json',record)
        store.atomic_json(DEST/'training.json',record)
        telemetry=[line for line in (LOCAL/'training.log').read_text().splitlines() if 'loss ' in line or 'Trainable parameters' in line]
        (DEST/'training.log').write_text('\n'.join(telemetry)+'\n')
    print(json.dumps(record),flush=True)


def adapter_path():
    record=store.read_json(LOCAL/'training.json')
    weights=LOCAL/'adapter/adapters.safetensors'
    if record['status']!='completed' or hashlib.sha256(weights.read_bytes()).hexdigest()!=record['adapter_sha256']:
        raise ValueError('Incomplete or altered checkpoint')
    return str(LOCAL/'adapter')


def fixture(case):
    """Synthetic values follow the real nullable field/pooled reference contract."""
    snapshot=dict(results={},pooled={})
    for row in case['context']['results']:
        rid=row['id'];values={m:100+(rid%13)*7+i*60 for i,m in enumerate(row['available_metrics'])}
        snapshot['results'][str(rid)]=values
        snapshot['pooled'][str(rid)]={scope:{m:180+METRICS.index(m)*50 for m in row['available_metrics']} for scope in ('demographic','division','sex','global')}
    return snapshot


def evaluate(policy):
    target=DEST/f'{policy}.json'
    if target.exists():raise ValueError('Recorded evaluation exists; preserve it')
    cases=store.read_json(DEST/'test-adjudicated.json')
    adjudication=store.read_json(DEST/'adjudication.json')
    if hashlib.sha256((DEST/'test-adjudicated.json').read_bytes()).hexdigest()!=adjudication['test_sha256']:
        raise ValueError('Adjudicated test changed')
    model=None
    if policy in ('base','structured','specialist'):
        model=local_load(adapter_path() if policy=='specialist' else None)
    rows=[];start=time.perf_counter()
    record=dict(status='running',policy=policy,model=MODEL_ID if model else policy,revision=MODEL_REVISION if model else None,
                input_manifest=store.read_json(DEST/'input-manifest.json'),test_sha256=adjudication['test_sha256'],rows=rows)
    if policy=='specialist':record['adapter_sha256']=store.read_json(LOCAL/'training.json')['adapter_sha256']
    try:
        for case in cases:
            prediction=dict(raw='{"action":"clarify","result_ids":[],"metrics":[],"scope":"none","reason":"metric"}',elapsed_ms=0)
            if model:prediction=predict(model,case,structured=policy=='structured')
            if policy=='frontier':prediction=frontier(case)
            scored=grade(prediction['raw'],case)
            outcome=None
            if scored['valid']:outcome=execute(Plan.model_validate(scored['output']),case['context'],fixture(case))
            intended_outcome=execute(Plan.model_validate(case['expected']),case['context'],fixture(case))
            rows.append(dict(id=case['id'],family=case['family'],request=case['request'],expected=case['expected'],**prediction,**scored,
                             execution=outcome,intended_execution=intended_outcome))
            store.atomic_json(target,record)
            print(policy,case['id'],'correct' if scored['intent_correct'] else 'incorrect',flush=True)
        count=len(rows)
        record.update(status='completed',wall_seconds=round(time.perf_counter()-start,3),metrics=dict(count=count,
            intent_correct=sum(r['intent_correct'] for r in rows),valid=sum(r['valid'] for r in rows),
            unsafe_execution=sum(r.get('unsafe_execution',False) for r in rows),
            unnecessary_abstention=sum(r.get('unnecessary_abstention',False) for r in rows),
            median_ms=statistics.median(r['elapsed_ms'] for r in rows)))
    except Exception as exc:
        record.update(status='failed',error=str(exc));raise
    finally:store.atomic_json(target,record)
    print(json.dumps(record['metrics']),flush=True)


def infer(path):
    request=store.read_json(Path(path));case=dict(request=request['question'],context=request['context'])
    model=local_load(adapter_path() if request['policy']=='specialist' else None)
    result=predict(model,case,structured=request['policy']=='structured')
    from workshop.completion import decode
    result.update(model=MODEL_ID,revision=MODEL_REVISION,intent_correct=None,**decode(result['raw']))
    try:
        plan=validate_plan(result['content'],case['context'])
        result.update(valid=True,plan=plan.model_dump(),execution=execute(plan,case['context'],fixture(case)))
    except ValueError as exc:
        result.update(valid=False,error=str(exc),plan=None,execution=None)
    store.atomic_json(Path(path).with_suffix('.result.json'),result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['train','base','structured','specialist','frontier','clarify-control','infer']);parser.add_argument('path',nargs='?');args=parser.parse_args()
    store.initialize();LOCAL.mkdir(parents=True,exist_ok=True);check_frozen()
    if args.operation=='frontier':evaluate('frontier');return
    with (store.STATE/'accelerator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.operation=='train':train(lock.fileno());return
        if args.operation=='infer':infer(args.path);return
        evaluate(args.operation)


if __name__=='__main__':main()
