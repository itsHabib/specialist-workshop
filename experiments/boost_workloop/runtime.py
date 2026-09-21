"""Run an editable source snapshot in a disposable, networkless Python sandbox."""
import json
import os
from pathlib import PurePosixPath
import subprocess
import tempfile
import time
import uuid

IMAGE='python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'
WRAPPER=r'''
import json,os,pathlib,resource,subprocess,sys,tempfile
resource.setrlimit(resource.RLIMIT_FSIZE,(2097152,2097152))
p=json.load(sys.stdin)
root=pathlib.Path('/tmp/repo');root.mkdir()
for name,content in p['files'].items():
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
with tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
    child=subprocess.run(p['command'],shell=True,cwd=root,input=p['stdin'].encode(),stdout=out,stderr=err,timeout=90)
    out.seek(0);err.seek(0)
    print(json.dumps({'returncode':child.returncode,'stdout':out.read(200000).decode(errors='replace'),'stderr':err.read(200000).decode(errors='replace')}))
'''


def validate(files):
    if not isinstance(files,dict) or not files or len(files)>100:
        raise ValueError('snapshot must contain 1..100 text files')
    for name,content in files.items():
        p=PurePosixPath(name)
        if not isinstance(content,str) or p.is_absolute() or '..' in p.parts or not name or str(p)!=name:
            raise ValueError('unsafe source path or non-text content')
    if len(json.dumps(files).encode())>1_000_000:
        raise ValueError('snapshot exceeds 1 MB')


def execute(files,command,stdin='',stop=None):
    validate(files)
    name='boost-workloop-'+uuid.uuid4().hex[:16];start=time.monotonic()
    args=['docker','run','--rm','--name',name,'--network','none','--read-only',
          '--log-driver=none','--cap-drop=ALL','--security-opt=no-new-privileges',
          '--pids-limit','64','--memory','256m','--memory-swap','256m','--cpus','1',
          '--user','65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=64m,mode=1777',
          '-i',IMAGE,'timeout','-s','KILL','100s','python','-I','-c',WRAPPER]
    with tempfile.TemporaryFile() as inp,tempfile.TemporaryFile() as out,tempfile.TemporaryFile() as err:
        inp.write(json.dumps({'files':files,'command':command,'stdin':stdin}).encode());inp.seek(0)
        process=subprocess.Popen(args,stdin=inp,stdout=out,stderr=err)
        failure=None;cleanup=False
        try:
            while process.poll() is None:
                if stop is not None and stop.exists():failure='stopped';break
                if time.monotonic()-start>110:failure='sandbox_timeout';break
                if os.fstat(out.fileno()).st_size+os.fstat(err.fileno()).st_size>1_000_000:failure='output_limit';break
                time.sleep(.05)
        finally:
            removed=subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=15)
            if process.poll() is None:process.kill()
            process.wait(timeout=5)
            cleanup=removed.returncode==0
            if not cleanup:
                checked=subprocess.run(['docker','inspect',name],capture_output=True,timeout=10)
                cleanup=checked.returncode!=0 and b'No such' in checked.stderr
        out.seek(0);err.seek(0)
        raw=out.read(1_000_000).decode(errors='replace');error_text=err.read(4000).decode(errors='replace')
    result={'error':failure,'returncode':process.returncode,'stdout':'','stderr':error_text,
            'seconds':time.monotonic()-start,'cleanup_confirmed':cleanup}
    if not cleanup:result['error']='cleanup_unconfirmed';return result
    if failure:return result
    if process.returncode:result['error']='sandbox_failure';return result
    try:result.update(json.loads(raw))
    except (ValueError,TypeError):result['error']='invalid_sandbox_output'
    return result
