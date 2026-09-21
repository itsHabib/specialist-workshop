"""Adaptive follow-up: simpler response schema and explicit algorithm instruction."""
import json
from pathlib import Path
import random
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import cases, geometry, grade, GEOMETRY, hashes, write, reference_source
import models

GUIDE = '''Algorithm outline (not an implementation): Use fractions.Fraction throughout. Decompose the plane into horizontal slabs at every vertex y and every proper edge-intersection y. Ignore horizontal edges. Each other directed edge has x(y)=a*y+b, active between its endpoints, with winding delta +1 for increasing y and -1 otherwise. Find pairwise intersection y exactly; include it only inside BOTH edge y ranges. Within each open slab, crossings keep their order. Sort active crossings at the slab midpoint, group coincident x values and sum their winding deltas. Scan from left to right, accumulating winding; include a gap if winding is odd or nonzero according to the rule. Integrate each included gap exactly over slab height (linear width, midpoint rule is exact). Sum all slabs. Return a canonical nonnegative rational string. Handle empty and degenerate walks. This describes a generic exact sweep; you must implement and validate it yourself.'''


def run(root):
    root.mkdir(parents=True,exist_ok=False,mode=0o700)
    development=cases(337891)['geometry'];development=development[:2]+development[-2:]
    seen={json.dumps(q,sort_keys=True) for q in development}
    final=[q for q in cases(812731)['geometry'] if json.dumps(q,sort_keys=True) not in seen]
    tags=json.load(urllib.request.urlopen('http://127.0.0.1:11434/api/tags'))
    trials=[(m,a) for m in ('llama3.2:1b','qwen2.5:7b') for a in ('simple','guided')]
    random.Random(195).shuffle(trials)
    write(root/'plan.json',{'label':'adaptive representation follow-up, not same-model critic evidence',
       'guide_origin':'Authored by the driving frontier agent using the known exact algorithm; not a cheap-model discovery',
       'guide':GUIDE,'development':development,'final':final,'trials':trials,'models':tags,
       'files':hashes(),'timeout_seconds':180,'calls':4,'source_only_schema':True})
    for label,src,want in [('reference',reference_source('geometry'),True),('invalid','def evaluate(q): return {}',False)]:
        result=grade('geometry',src,final,root/'STOP');write(root/('control-'+label+'.json'),result)
        if result['passed']!=want:raise RuntimeError('control failed')
    models.SCHEMA={'type':'object','properties':{'source':{'type':'string'}},'required':['source'],'additionalProperties':False}
    summary=[]
    for model,arm in trials:
        prompt=GEOMETRY+'\nReturn JSON with exactly one key, source, containing actual executable Python code, not a description, not markdown.\nExamples:\n'+json.dumps([{'input':q,'expected':geometry(q)} for q in development])
        if arm=='guided':prompt+='\n'+GUIDE
        label=model.replace(':','-')+'-'+arm
        receipt=models.complete(prompt,model,root/label,timeout=180,stop=root/'STOP')
        source=(receipt.get('response') or {}).get('source','') if not receipt['error'] else ''
        (root/(label+'.py')).write_text(source)
        result=grade('geometry',source,final,root/'STOP')
        summary.append({'model':model,'arm':arm,'receipt':receipt,'result':result})
        write(root/'summary.json',summary)
        print(label,receipt['error'],round(receipt['seconds'],1),result['total']-result['failed'],result['total'],flush=True)


if __name__=='__main__':run(Path(sys.argv[1]).resolve())
