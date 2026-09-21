"""Reproduce the post-hoc boundary finding without model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from screen import grade, write, ROOT
import tasks


def request():
    rng=random.Random(8675309)
    return {'rule':'nonzero','rings':[[[rng.randrange(-10**9,10**9+1),rng.randrange(-10**9,10**9+1)] for _ in range(60)]]}


def reproduce(out):
    out.mkdir(parents=True,exist_ok=False)
    q=request();write(out/'request.json',q);write(out/'expected.json',tasks.geometry(q));rows=[]
    for label,path,patched in [
        ('opus-deadline','deadline-01/geometry-opus',False),
        ('astra-deadline','deadline-01/geometry-astra',False),
        ('astra-discovery','discovery-01/geometry-astra',False),
        ('opus-serialization-control','deadline-01/geometry-opus',True),
        ('astra-serialization-control','deadline-01/geometry-astra',True),
    ]:
        source=(ROOT/'receipts'/path/'final.py').read_text()
        if patched:source='import sys\nsys.set_int_max_str_digits(0)\n'+source
        result=grade('geometry',source,[q],out/'STOP')
        rows.append({'label':label,'patched_control':patched,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),**result})
        print(label,result['passed'],result['execution']['error'],flush=True)
    write(out/'results.json',rows)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args();reproduce(args.output.resolve())
