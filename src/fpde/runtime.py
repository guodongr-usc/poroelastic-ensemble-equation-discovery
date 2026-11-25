# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from __future__ import annotations
import os,sys,json,hashlib,tempfile,time,subprocess,shlex,math
from pathlib import Path
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
for key in THREADS:os.environ[key]='1'
os.environ['PYTHONNOUSERSITE']='1'
os.environ['PYTHONUNBUFFERED']='1'

def clean_env():
    env=os.environ.copy()
    for key in ('PYTHONPATH','PYTHONHOME','JAX_PLATFORMS','JAX_PLATFORM_NAME','XLA_FLAGS'):env.pop(key,None)
    for key in THREADS:env[key]='1'
    env.update(PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1');return env

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,Path):return str(x)
    if hasattr(x,'tolist'):return plain(x.tolist())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x

def digest(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def atomic_bytes(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(5):
        tmp=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',delete=False) as f:
                tmp=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(tmp,path);return
        except OSError as e:
            print(f'IO retry {attempt+1}/5 {path}: {e}',file=sys.stderr,flush=True)
            if tmp:
                try:tmp.unlink(missing_ok=True)
                except OSError:pass
            if attempt==4:raise
            time.sleep(2**attempt)

def save_json(path,obj):atomic_bytes(path,(json.dumps(plain(obj),sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
def load_json(path):return json.loads(Path(path).read_text())

def command(argv,cwd=None,timeout=120):
    argv=list(map(str,argv));print('$ '+shlex.join(argv),flush=True)
    try:r=subprocess.run(argv,cwd=cwd,env=clean_env(),text=True,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired as e:
        print(f'return code: TIMEOUT\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}',flush=True)
        raise RuntimeError(f'Command timed out: {argv}; stderr={e.stderr!r}') from e
    print(f'return code: {r.returncode}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}',flush=True)
    if r.returncode:raise RuntimeError(f'Command failed rc={r.returncode}: {argv}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}')
    return r.stdout

def environment():
    import numpy,scipy
    record={'python':sys.version,'executable':sys.executable,'numpy':numpy.__version__,'scipy':scipy.__version__}
    for pkg in (numpy,scipy):
        record[pkg.__name__+'_path']=pkg.__file__
        if str(Path.home()/'.local') in pkg.__file__:raise RuntimeError('Wrong user-site package: '+pkg.__file__)
    try:
        from threadpoolctl import threadpool_info
        record['threadpools']=threadpool_info()
        if any(p.get('num_threads',1)>1 for p in record['threadpools']):raise RuntimeError('BLAS oversubscription: '+str(record['threadpools']))
    except ImportError:record['threadpools']='threadpoolctl optional; env variables enforce one thread'
    print(json.dumps(record,indent=2),flush=True);return record
