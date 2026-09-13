"""Local task-package CLI and finite workers. No hosted job service."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from workshop import packages,store
from workshop.completion import decode,VERSION
from workshop.analytics import MODEL_ID,MODEL_REVISION
from workshop.domain import check_training_length,digest


def runs_root():return store.STATE/'package-runs'


def read_run(identity):
    import re
    if not re.fullmatch(r'pkg-[a-f0-9]{12}',identity):raise ValueError('Invalid package run ID')
    return store.read_json(runs_root()/identity/'run.json')


def checkpoint(package,identity):
    run=read_run(identity)
    if run['status']!='completed' or run['operation'] not in ('train','adopt'):raise ValueError('Checkpoint is not completed training')
    if run['contract_hash']!=packages.contract_hash(package):raise ValueError('Checkpoint contract differs')
    if (run['model'],run['revision'])!=(MODEL_ID,MODEL_REVISION):raise ValueError('Checkpoint model differs')
    path=runs_root()/identity/'adapter'
    if hashlib.sha256((path/'adapters.safetensors').read_bytes()).hexdigest()!=run['adapter_sha256']:raise ValueError('Checkpoint weights changed')
    return path


class Local:
    def __init__(self,adapter=None):
        from huggingface_hub import snapshot_download
        from mlx_lm import load
        self.model,self.tokenizer=load(snapshot_download(MODEL_ID,revision=MODEL_REVISION),adapter_path=str(adapter) if adapter else None)
    def predict(self,package,value):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        mx.random.seed(29)
        prompt=self.tokenizer.apply_chat_template(packages.messages(package,value),tokenize=False,add_generation_prompt=True)
        tokens=len(self.tokenizer.encode(prompt))
        if tokens>4000:raise ValueError('Input exceeds 4000-token limit')
        start=time.perf_counter()
        raw=generate(self.model,self.tokenizer,prompt=prompt,max_tokens=260,sampler=make_sampler(temp=0),verbose=False)
        return dict(raw=raw,**decode(raw),elapsed_ms=round((time.perf_counter()-start)*1000),input_tokens=tokens,output_tokens=len(self.tokenizer.encode(raw)))


class Reference:
    """Explicit API opt-in; credentials never enter package or run artifacts."""
    def __init__(self):
        import os
        self.url=os.environ.get('WORKSHOP_REFERENCE_URL');self.model=os.environ.get('WORKSHOP_REFERENCE_MODEL');self.key=os.environ.get('WORKSHOP_REFERENCE_KEY','')
        if not self.url or not self.model:raise ValueError('Configure an approved reference endpoint/model first')
    def predict(self,package,value):
        import httpx
        start=time.perf_counter()
        response=httpx.post(self.url.rstrip('/')+'/chat/completions',headers={'Authorization':f'Bearer {self.key}'},json=dict(model=self.model,messages=packages.messages(package,value),temperature=0,max_tokens=260),timeout=120)
        response.raise_for_status();body=response.json()
        return dict(raw=body['choices'][0]['message']['content'],elapsed_ms=round((time.perf_counter()-start)*1000),usage=body.get('usage'),model=self.model)


def preview(package):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(snapshot_download(MODEL_ID,revision=MODEL_REVISION))
    row=package.train[0]
    rendered=tokenizer.apply_chat_template(packages.messages(package,row.input,row.expected),tokenize=False)
    return dict(case_id=row.id,model=MODEL_ID,revision=MODEL_REVISION,completion_protocol=VERSION,
                template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                training_tokens=len(tokenizer.apply_chat_template(packages.messages(package,row.input,row.expected),tokenize=True,return_dict=False)),
                expected_content=row.expected,serialized_assistant=rendered.rsplit('<|im_start|>assistant',1)[-1])


def training_data(package,directory):
    data=directory/'data';data.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,rows in [('train',package.train),('valid',package.development)]:
        text=''.join(json.dumps(dict(messages=packages.messages(package,row.input,row.expected)))+'\n' for row in rows)
        (data/f'{name}.jsonl').write_text(text)
        hashes[name]=hashlib.sha256(text.encode()).hexdigest()
    return data,hashes


def train(package,run,directory):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path=snapshot_download(MODEL_ID,revision=MODEL_REVISION);tokenizer=AutoTokenizer.from_pretrained(path)
    for row in package.train+package.development:check_training_length(tokenizer,packages.messages(package,row.input,row.expected),row.id,limit=2048)
    store.atomic_json(directory/'tokenization-preview.json',preview(package))
    data,hashes=training_data(package,directory)
    command=[sys.executable,'-m','mlx_lm','lora','--model',path,'--train','--data',str(data),'--adapter-path',str(directory/'adapter'),
             '--iters',str(run['steps']),'--batch-size','1','--num-layers','8','--learning-rate','0.00002','--max-seq-length','2048',
             '--mask-prompt','--steps-per-report','10','--steps-per-eval','40','--val-batches',str(min(6,len(package.development))),
             '--save-every',str(run['steps']),'--seed','29']
    run.update(training_data_hashes=hashes,recipe=dict(steps=run['steps'],batch_size=1,layers=8,learning_rate=2e-5,seed=29,max_sequence_length=2048,mask_prompt=True))
    store.atomic_json(directory/'run.json',run)
    with (directory/'training.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=2400)
    run['adapter_sha256']=hashlib.sha256((directory/'adapter/adapters.safetensors').read_bytes()).hexdigest()


def adopt(package,run,directory):
    """Import prior weights only when exact serialized training streams match."""
    source=Path(run['source']).resolve();record=store.read_json(source/'training.json')
    if record['status']!='completed' or (record['model'],record['revision'])!=(MODEL_ID,MODEL_REVISION):raise ValueError('Source training identity mismatch')
    _,hashes=training_data(package,directory)
    for name,expected in hashes.items():
        if hashlib.sha256((source/'data'/f'{name}.jsonl').read_bytes()).hexdigest()!=expected:raise ValueError('Source training messages differ from package')
    weights=source/'adapter/adapters.safetensors'
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=record['adapter_sha256']:raise ValueError('Source weights hash mismatch')
    shutil.copytree(source/'adapter',directory/'adapter')
    run.update(training_data_hashes=hashes,adapter_sha256=record['adapter_sha256'],source_training=record)


def assess(package,value,prediction,expected=None):
    result=packages.score(package,value,prediction.get('content',prediction['raw']),expected)
    result['raw_valid']=packages.score(package,value,prediction['raw'])['valid']
    return result


def metrics(rows):
    return dict(count=len(rows),correct=sum(r.get('correct') is True for r in rows),valid=sum(r.get('valid',False) for r in rows),
                accepted=sum(r.get('accepted',False) for r in rows),abstained=sum(r.get('abstained',False) for r in rows),
                invalid=sum(not r.get('valid',False) for r in rows),raw_valid=sum(r.get('raw_valid',False) for r in rows),median_ms=statistics.median(r.get('elapsed_ms',0) for r in rows) if rows else None)


def evaluate(package,run,directory):
    policy=run['policy'];model=None
    if policy in ('base','specialist'):model=Local(checkpoint(package,run['checkpoint']) if policy=='specialist' else None)
    if policy=='reference':model=Reference();run['model']=model.model;run['revision']=None
    majority=Counter(json.dumps(r.expected,sort_keys=True) for r in package.train).most_common(1)[0][0]
    rows=[]
    for case in package.final:
        prediction=model.predict(package,case.input) if model else dict(raw=majority,elapsed_ms=0)
        result=packages.score(package,case.input,prediction.get('content',prediction.get('content',prediction['raw'])),case.expected)
        rows.append(dict(id=case.id,family=case.family,input=case.input,expected=case.expected,**prediction,**result))
        run.update(rows=rows,metrics=metrics(rows));store.atomic_json(directory/'run.json',run)


def episodes(package,run,directory):
    from workshop.environments import create
    if not package.environment or not package.episodes:raise ValueError('Package has no final environment episodes')
    model=None
    if run['policy'] in ('base','specialist'):model=Local(checkpoint(package,run['checkpoint']) if run['policy']=='specialist' else None)
    if run['policy']=='reference':model=Reference();run['model']=model.model;run['revision']=None
    rows=[]
    for case in package.episodes:
        trace=[]
        with tempfile.TemporaryDirectory(prefix='package-practice-') as tmp:
            env=create(package.environment,case,tmp)
            while not env.done and len(trace)<5:
                observation=env.observe()
                prediction=model.predict(package,observation) if model else dict(raw=json.dumps(env.baseline()),elapsed_ms=0)
                result=assess(package,observation,prediction)
                env.step(result['output'] if result['valid'] else {})
                trace.append(dict(observation=observation,**prediction,**result))
            rows.append(dict(id=case['id'],family=case['family'],trace=trace,**env.outcome()))
        run.update(rows=rows,metrics=dict(count=len(rows),success=sum(r['success'] for r in rows),invalid=sum(r['invalid'] for r in rows),harmful=sum(r['harmful'] for r in rows)))
        store.atomic_json(directory/'run.json',run)


def new_run(reference,operation,policy='base',checkpoint_id=None,input_value=None,steps=120,source=None):
    package=packages.load(reference)
    if operation not in ('train','evaluate','infer','episodes','adopt'):raise ValueError('Unknown operation')
    if policy not in ('base','specialist','reference','majority','rules'):raise ValueError('Unknown policy')
    if not 1<=steps<=300:raise ValueError('Steps must be 1..300')
    if operation=='infer' and not isinstance(input_value,(str,dict)):raise ValueError('Inference needs a JSON object or string input')
    if operation=='infer' and policy not in ('base','specialist','reference'):raise ValueError('Inference needs a model policy')
    if operation=='evaluate' and policy=='rules':raise ValueError('Rule baseline is provided by an environment; use episodes')
    if operation=='episodes' and policy=='majority':raise ValueError('Use rules or a model for episodes')
    if operation=='train' and policy!='base':raise ValueError('Training starts from the base model')
    if policy=='specialist':checkpoint(package,checkpoint_id)
    if policy=='reference':Reference()
    identity=store.new_id('pkg')
    run=dict(id=identity,package=reference,operation=operation,policy=policy,checkpoint=checkpoint_id,input=input_value,steps=steps,source=source,
             status='queued',created_at=time.time(),model=MODEL_ID,revision=MODEL_REVISION,**{k:v for k,v in packages.summary(package).items() if k in ('contract_hash','final_hash')},
             grader=package.grader,grader_hash=digest(dict(schema=package.output_schema,grader=package.grader,unordered_fields=package.unordered_fields,environment=package.environment,transport=VERSION)),completion_protocol=VERSION,local_cost_usd=None)
    if operation=='episodes':run['final_hash']=digest(package.episodes)
    store.atomic_json(runs_root()/identity/'package.json',package.model_dump());store.atomic_json(runs_root()/identity/'run.json',run)
    return run


def worker(identity):
    run=read_run(identity);directory=runs_root()/identity
    if run['status']!='queued':raise ValueError('Run already started; preserve previous artifacts')
    package=packages.load(run['package']);run.update(status='running',started_at=time.time());store.atomic_json(directory/'run.json',run)
    try:
        with (store.STATE/'accelerator.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            op=run['operation']
            if op=='train':train(package,run,directory)
            if op=='adopt':adopt(package,run,directory)
            if op=='evaluate':evaluate(package,run,directory)
            if op=='episodes':episodes(package,run,directory)
            if op=='infer':
                model=Reference() if run['policy']=='reference' else Local(checkpoint(package,run['checkpoint']) if run['policy']=='specialist' else None)
                prediction=model.predict(package,run['input']);run.update(result={**prediction,**assess(package,run['input'],prediction)})
            run['status']='completed'
    except Exception as exc:run.update(status='failed',error=str(exc))
    run['finished_at']=time.time();store.atomic_json(directory/'run.json',run)
    return run


def compare(identities):
    runs=[read_run(identity) for identity in identities]
    if len({(r['final_hash'],r['grader_hash'],r['operation']) for r in runs})!=1:raise ValueError('Different evaluation sets, graders or operation types cannot be compared')
    if any(r['status']!='completed' for r in runs):raise ValueError('Only completed runs may be compared')
    return [dict(id=r['id'],package=r['package'],policy=r['policy'],model=r['model'],checkpoint=r['checkpoint'],metrics=r.get('metrics'),wall_seconds=r['finished_at']-r['started_at'],local_cost_usd=r['local_cost_usd']) for r in runs]


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('import');p.add_argument('path')
    p=sub.add_parser('show');p.add_argument('reference')
    p=sub.add_parser('preview');p.add_argument('reference')
    p=sub.add_parser('correct');p.add_argument('reference');p.add_argument('case_path')
    p=sub.add_parser('run');p.add_argument('reference');p.add_argument('operation',choices=['train','evaluate','infer','episodes','adopt']);p.add_argument('--policy',default='base');p.add_argument('--checkpoint');p.add_argument('--input-file');p.add_argument('--steps',type=int,default=120);p.add_argument('--source')
    p=sub.add_parser('worker');p.add_argument('identity')
    p=sub.add_parser('compare');p.add_argument('identities',nargs='+')
    args=parser.parse_args();store.initialize()
    if args.command=='import':result=packages.import_package(packages.Package.model_validate(store.read_json(Path(args.path))))
    if args.command=='show':result=packages.load(args.reference).model_dump()
    if args.command=='preview':result=preview(packages.load(args.reference))
    if args.command=='correct':result=packages.correct(args.reference,packages.Case.model_validate(store.read_json(Path(args.case_path))))
    if args.command=='compare':result=compare(args.identities)
    if args.command=='worker':result=worker(args.identity)
    if args.command=='run':
        value=store.read_json(Path(args.input_file)) if args.input_file else None
        run=new_run(args.reference,args.operation,args.policy,args.checkpoint,value,args.steps,args.source)
        result=worker(run['id'])
    print(json.dumps(result,indent=2))
    if isinstance(result,dict) and result.get('status')=='failed':raise SystemExit(1)


if __name__=='__main__':main()
