"""Export experiment artifacts; inspect the inventory before public release."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(root, destination):
    destination.mkdir(parents=True,exist_ok=False)
    plan = json.loads((root/'plan.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    def clean(value):
        if isinstance(value,list): return [clean(x) for x in value]
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items() if k not in ('stderr','container')}
        return value
    (destination/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    (destination/'summary.json').write_text(json.dumps(clean(summary),indent=2)+'\n')
    manifest = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file(): continue
        if path.name in ('request.json','process.json','stdout.log','stderr.log','driver.lock','started','summary.json','plan.json'): continue
        relative = path.relative_to(root)
        if path.suffix not in ('.py','.json') and path.name!='prompt.txt': continue
        target = destination/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.json': target.write_text(json.dumps(clean(json.loads(path.read_text())),indent=2)+'\n')
        else: target.write_bytes(path.read_bytes())
        manifest[str(relative)] = {'raw_sha256':digest(path),'published_sha256':digest(target)}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':publish(*(Path(x) for x in sys.argv[1:]))
