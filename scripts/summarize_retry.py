"""Reprice retained usage, including cache writes, without changing raw runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from retry_economics import cost, summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    path=args.directory/'run.json'
    run=json.loads(path.read_text())
    if run['status']!='completed':raise ValueError('Run incomplete')
    if hashlib.sha256((args.directory/'executed_script.py').read_bytes()).hexdigest()!=run['script_sha256']:
        raise ValueError('Executed source does not match recorded hash')
    result=dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),policies={},
                accounting='These totals reprice original usage at the documented 1.25x input write rate. The first 2026-09-15 run omitted that premium from inline estimates; raw records are unchanged.',
                local_cost_usd=None)
    for name,policy in run['policies'].items():
        rows=copy.deepcopy(policy['rows'])
        for row in rows:
            for attempt in row['attempts']:
                if attempt.get('usage'):attempt['cost_usd']=cost(name,attempt['usage'])
        result['policies'][name]=dict(startup_seconds=policy['startup_seconds'],one_shot=summarize(rows,1),retry=summarize(rows,3))
    api_costs=[p['retry']['total_cost_usd'] for name,p in result['policies'].items() if not name.startswith('local-')]
    result['api_total_usd']=sum(api_costs) if all(c is not None for c in api_costs) else None
    if {'gpt-6-astra','gpt-5.6-luna'}<=result['policies'].keys():
        frontier=result['policies']['gpt-6-astra']['retry']['total_cost_usd']
        cheap=result['policies']['gpt-5.6-luna']['retry']['total_cost_usd']
        result['api_cost_ratio']=frontier/cheap if frontier is not None and cheap else None
    (args.directory/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
