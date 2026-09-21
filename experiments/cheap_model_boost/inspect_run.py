"""Summarize validity and actual policy overhead without treating errors as reasoning."""
import ast
import json
from pathlib import Path
import sys


def inspect(root):
    rows=[]
    for item in json.loads((root/'summary.json').read_text()):
        calls={call['label']:call['receipt'] for call in item['calls']}
        for arm, labels in [('initial',['initial']),('solo',['initial','solo-2','solo-3']),('boost',['initial','critic','boost-repair'])]:
            faults=[]; costs=[]; seconds=0; usage={}
            for label in labels:
                receipt=calls[label]; response=receipt.get('response') or {}
                if receipt['error']: faults.append(label+':'+receipt['error'])
                expected='hypothesis' if label=='critic' else 'edit'
                if response.get('action')!=expected: faults.append(label+':unexpected_action')
                if label!='critic':
                    source=response.get('source','')
                    if not source.strip(): faults.append(label+':empty_source')
                    try: ast.parse(source)
                    except (ValueError,SyntaxError): faults.append(label+':invalid_python')
                costs.append(receipt.get('estimated_cost_usd'));seconds+=receipt['seconds']
                usage[label]=receipt.get('usage')
            result=item['finals'][arm]
            rows.append({'model':item['model'],'policy':arm,'passed':result['passed'],
                         'cases_passed':result['total']-result['failed'],'total':result['total'],
                         'protocol_faults':faults,'model_seconds':seconds,
                         'known_cost_usd':sum(c for c in costs if c is not None),
                         'cost_complete':all(c is not None for c in costs),'usage':usage})
    return rows


if __name__=='__main__':print(json.dumps(inspect(Path(sys.argv[1])),indent=2))
