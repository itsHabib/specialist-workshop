"""Export allowlisted synthetic receipts; inspect the result before publication."""
import hashlib
import json
from pathlib import Path
import re
import sys

ALLOWED={'plan.json','task.json','parent.json','evaluation-plan.json','initial-control.json',
         'reference-control.json','state.json','snapshots.json','grading.json','stop-reason.json',
         'lessons.json','provenance.json','component.json','receipt.json','schema.json','prompt.txt'}


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,str):
        value=re.sub(r'(?:/private)?/var/folders/[^\s\"\'\\]+','<client-temp>',value)
        return re.sub(r'/Users/[^\s\"\'\\]+','<local-path>',value)
    return value


def publish(source,destination):
    destination.mkdir(parents=True,exist_ok=False)
    manifest={}
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.name not in ALLOWED:continue
        relative=path.relative_to(source);target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        raw=path.read_bytes()
        data=json.dumps(clean(json.loads(raw)),indent=2)+'\n' if path.suffix=='.json' else clean(raw.decode())
        target.write_text(data)
        manifest[str(relative)]={'raw_sha256':hashlib.sha256(raw).hexdigest(),
                                 'published_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':publish(*(Path(p) for p in sys.argv[1:]))
