"""Exploratory replay of already exposed Blox defects; never a holdout."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop.arithmetic import arithmetic_rule
from workshop.domain import Skill, grade, summarize
from workshop.runtime import LocalModel
from workshop import store

DEST = store.ROOT / 'experiments/workflows/blox-retained-diagnostics.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Path to your original failure-cases.json evidence')
    source = parser.parse_args().source
    if DEST.exists():
        raise ValueError('Retain previous evidence; do not overwrite')
    cases = json.loads(source.read_text())['cases']
    selections = [
        (0, 'baseline', 'distance-conflict', '8 down-and-back trips on a 10 metre lane; claimed distance 80 metres.'),
        (2, 'benchmark', 'time-overrun', 'Benchmark completion 598 seconds; block budget 480 seconds.'),
        (3, 'main', 'time-overrun', '4 repetitions of 371 seconds; 120 seconds rest between repetitions; budget 1200 seconds.'),
        (4, 'main', 'distance-conflict', 'Technique 4 lengths of 10 metres plus test 40 metres; claimed total 40 metres.'),
    ]
    rows=[]
    for index, block, expected, normalized in selections:
        case=cases[index]
        text=next(b['detail'] for b in case['source_session']['blocks'] if b['label']==block)
        if index == 4:
            text += '\n' + case['source_session']['why']
        rows.append(dict(id=f'retained-{index}', source_candidate=case['candidate'], input=text, expected=expected, normalized=normalized))
    skill=Skill.model_validate(store.read_json(store.ROOT/'skills/blox-arithmetic/skill.json'))
    report=dict(note='Exposed diagnostic defects selected after synthetic base results; not final holdout. Four cases within the label contract. The separate audit-overcount defect is outside this contract. Normalization is human-authored, not model extraction.',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=rows, policies={})
    with (store.STATE/'accelerator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for policy in ('rules-raw','rules-human-normalized','base','specialist'):
            model=None
            if policy=='base': model=LocalModel()
            if policy=='specialist':
                checkpoint=store.read_json(store.ROOT/'experiments/workflows/blox-arithmetic-checkpoint.json')
                model=LocalModel(store.load_job(checkpoint['id'])['adapter_path'])
            results=[]
            for row in rows:
                text=row['normalized'] if policy=='rules-human-normalized' else row['input']
                if model:
                    raw,elapsed,tokens=model.predict(skill,text)
                else:
                    start=time.perf_counter();raw=json.dumps(dict(bucket=arithmetic_rule(text),line=1));elapsed=(time.perf_counter()-start)*1000
                result=grade(skill,text,raw,row['expected']);result.update(id=row['id'],elapsed_ms=elapsed)
                results.append(result)
            report['policies'][policy]=dict(metrics=summarize(results),rows=results)
            store.atomic_json(DEST,report)
            del model
            import gc
            gc.collect()
    print({k:v['metrics'] for k,v in report['policies'].items()})


if __name__=='__main__': main()
