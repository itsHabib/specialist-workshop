"""Post-hoc format diagnostic, not an arm of the original experiment."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import grade


def diagnose(root):
    plan=json.loads((root/'plan.json').read_text())
    source=(root/'sonnet/solo-3.py').read_text()
    suffix='</parameter>\n</invoke>'
    stripped=source.rstrip()
    if not stripped.endswith(suffix):raise RuntimeError('expected exact observed suffix absent')
    patched=stripped[:-len(suffix)].rstrip()+'\n'
    result=grade('geometry',patched,plan['final'],root/'STOP')
    output={'label':'post-hoc formatting-only replay; final set already spent',
            'edit':'Remove exact trailing XML closing tags; no algorithm changes',
            'result':result}
    (root/'sonnet/format-only.py').write_text(patched)
    (root/'sonnet/format-diagnostic.json').write_text(json.dumps(output,indent=2)+'\n')
    print({'passed':result['passed'],'failed':result['failed'],'total':result['total']})


if __name__=='__main__':diagnose(Path(sys.argv[1]))
