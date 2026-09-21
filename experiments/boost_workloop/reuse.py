"""Export verified practice code as an inspectable dependency for a later task."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import agent
from learn import practice_packet


def export(run,grading,source,out):
    packet,digest=practice_packet(run,grading)
    if source not in packet['files']:raise ValueError('source file not in accepted snapshot')
    content=packet['files'][source]
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    agent.write(out/'component.json',{'schema':'boost-component.v1','source_name':source,
      'source':content,'source_sha256':hashlib.sha256(content.encode()).hexdigest(),
      'accepted_snapshot':digest,'accepted_goal':packet['goal'],
      'scope':'Accepted only for the recorded practice task; importing code does not prove new-task correctness.'})
    return out/'component.json'


def apply(task,component,name):
    if hashlib.sha256(component['source'].encode()).hexdigest()!=component['source_sha256']:
        raise ValueError('component digest mismatch')
    if name in task['files']:raise ValueError('dependency would overwrite an existing task file')
    files={**task['files'],name:component['source']};agent.validate(files)
    result={**task,'files':files}
    result['goal']+='\nA reusable dependency from an independently accepted earlier task is available as '+name+'. Its previously accepted goal was: '+component['accepted_goal']+' Inspect and reuse its functions where useful instead of reimplementing them. The current CLI/format/input requirements still need implementation and testing. The dependency is not automatically correct for new requirements.'
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--grading',type=Path,required=True);p.add_argument('--file',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(export(a.run,a.grading,a.file,a.out))
