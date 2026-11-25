# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.
# Forward evaluation of candidate PDEs follows the system-identification method
# of J. Im, F. P. J. de Barros and S. F. Masri (2023), Nonlinear Dynamics 111,
# 1987–2007. https://doi.org/10.1007/s11071-022-08182-z
# The field weights and initial-displacement penalty belong to this study.

from __future__ import annotations
import runtime
import math,time,copy,traceback,multiprocessing as mp
import resource,signal,os,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from runtime import save_json,load_json,digest
from data_io import load_npz
from solver_cpu import CPUSolver,NumericalFailure
from candidates import build,parse,nodes,mutate,cross,BOUNDS,equations,active,MECHANICS_OPTIONAL
from pressure_objective import components as objective_components,total as objective_total
class ChunkEnd(Exception):pass
def metrics(pred,truth,masks=None):
    rec={};error=[]
    for j,f in enumerate(('P','U1','U2')):
        a=np.asarray(pred[j]);b=np.asarray(truth[j])
        if masks is not None:a=a[:,masks[j]];b=b[:,masks[j]]
        sse=float(np.sum((a-b)**2));ss=float(np.sum(b*b));sst=float(np.sum((b-b.mean())**2));rec[f+'_R2']=1-sse/sst if sst>1e-25 else float('nan');rec[f+'_relL2']=math.sqrt(sse/max(ss,1e-30));error.append(rec[f+'_relL2']**2)
    rec['joint_relL2']=math.sqrt(sum(error)/3);rec['macro_R2']=float(np.nanmean([rec[f+'_R2'] for f in ('P','U1','U2')]));rec['min_field_R2']=float(np.nanmin([rec[f+'_R2'] for f in ('P','U1','U2')]))
    return rec
_SOLVER=_TRUTH=_OBS=_PENALTY=_IC_WEIGHT=_IC_TOL=_WEIGHTS=None
def init_worker(root,grid,penalty,max_seconds,ic_weight,ic_tolerance,field_weights=(.5,.25,.25),memory_gib=10):
    global _SOLVER,_TRUTH,_OBS,_PENALTY,_IC_WEIGHT,_IC_TOL,_WEIGHTS

    if sys.platform.startswith('linux'):
        cap=int(memory_gib*1024**3)
        soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        if hard!=resource.RLIM_INFINITY:cap=min(cap,hard)
        resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    root=Path(root);known=load_npz(root/'prepared/KNOWN_INPUTS.npz');_SOLVER=CPUSolver(known,grid,max_seconds=max_seconds);truth=np.load(root/'prepared/TRAIN_ONLY.npy',mmap_mode='r');_TRUTH=np.asarray(truth[:,:,_SOLVER.ids]);_OBS=known['theta'][:int(known['n_train'])];_PENALTY=penalty;_IC_WEIGHT=ic_weight;_IC_TOL=ic_tolerance;_WEIGHTS=tuple(field_weights)
def evaluate_worker(task):
    c,substeps=task;start=time.perf_counter()
    def alarm_handler(signum,frame):raise NumericalFailure('Candidate deadline expired')
    signal.signal(signal.SIGALRM,alarm_handler)
    signal.setitimer(signal.ITIMER_REAL,float(_SOLVER.max_seconds))
    try:
        with np.errstate(over='raise',invalid='raise'):
            pred,info=_SOLVER.forward(parse(c),_OBS,substeps)
        masks=[~_SOLVER.pmask,~_SOLVER.masks['u1_mask'].ravel(),~_SOLVER.masks['u2_mask'].ravel()];m=metrics(pred[:,1:],_TRUTH[:,1:],masks)
        if not math.isfinite(m['joint_relL2']) or not math.isfinite(info['initial_mechanics_relative_mismatch']):
            raise NumericalFailure('Non-finite forward/initial fitness')
        terms=objective_components(m,info['initial_mechanics_relative_mismatch'],nodes(c),weights=_WEIGHTS,ic_weight=_IC_WEIGHT,penalty=_PENALTY)
        return dict(candidate=c,success=True,loss=objective_total(terms),metrics=m,solver=info,substeps=substeps,complexity=nodes(c),initial_error=info['initial_mechanics_relative_mismatch'],initial_compatibility_passed=max(info['initial_mechanics_field_relL2'])<=_IC_TOL,fitness_components=terms,fitness_definition='sum_j(field_weights[j]*(1-R2_j_train))+IC_weight*IC_error+penalty*nodes',field_weights=list(_WEIGHTS),equations=equations(c,_SOLVER.sc),initial_compensation=False)
    except (NumericalFailure,np.linalg.LinAlgError,FloatingPointError) as e:return dict(candidate=c,success=False,loss=1e30,metrics=None,error=f'{type(e).__name__}: {e}',substeps=substeps,elapsed_seconds=time.perf_counter()-start)
    except MemoryError:return dict(candidate=c,success=False,loss=1e30,metrics=None,error='RESOURCE_LIMIT: worker memory cap exceeded',substeps=substeps,elapsed_seconds=time.perf_counter()-start)
    except Exception:return dict(candidate=c,success=False,loss=1e30,metrics=None,unexpected_exception=traceback.format_exc(),substeps=substeps)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
class Evaluator:
    def __init__(self,cfg,deadline,grid=None):
        self.cfg=cfg;self.deadline=deadline;self.root=Path(cfg['outdir']);self.grid=int(cfg['search_grid'] if grid is None and 'search_grid' in cfg else (cfg['grid'] if grid is None else grid));self.cache={};self.count=0;self.pool=ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(str(self.root),self.grid,cfg['penalty'],cfg['candidate_seconds'],cfg['ic_weight'],cfg['ic_tolerance'],cfg['field_weights'],cfg['worker_memory_gib']))
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)
    def evaluate(self,candidates,substeps,label):
        key=digest([self.cfg['identity'],self.grid,candidates,substeps]);path=self.root/'.states'/('evaluation_'+key+'.json');records=load_json(path)['records'] if path.exists() else [None]*len(candidates);pending={}
        for i,c in enumerate(candidates):
            sig=digest([c,self.grid,substeps])
            if records[i] is not None:self.cache[sig]=records[i]
            elif sig in self.cache:records[i]=copy.deepcopy(self.cache[sig])
            else:pending.setdefault(sig,[]).append(i)
        from concurrent.futures import wait,FIRST_COMPLETED
        keys=iter(pending);running={};changed=0;last_save=time.monotonic()
        def launch():
            if time.monotonic()>self.deadline:return False
            try:sig=next(keys)
            except StopIteration:return False
            f=self.pool.submit(evaluate_worker,(candidates[pending[sig][0]],substeps))
            running[f]=sig;return True
        for _ in range(self.cfg['workers']):
            if not launch():break
        def persist():
            save_json(path,dict(label=label,records=records))
            good=[r for r in records if r is not None and r['success']]
            best=min(good,key=lambda r:r['loss']) if good else None
            save_json(self.root/'CPU_LIVE.json',dict(stage=label,completed=sum(r is not None for r in records),requested=len(candidates),successful=len(good),workers=self.cfg['workers'],grid=self.grid,substeps=substeps,best=best,metrics_are_absolute_training_only=True,initial_compensation=False))
            if best:
                m=best['metrics'];print(f"LIVE {label} {sum(r is not None for r in records)}/{len(candidates)} loss={best['loss']:.6g} ABS_TRAIN_R2=({m['P_R2']:.5f},{m['U1_R2']:.5f},{m['U2_R2']:.5f}) macro={m['macro_R2']:.5f} ICerr={best['initial_error']:.4f}",flush=True)
        while running:
            done,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
            for fut in done:
                sig=running.pop(fut)
                try:r=fut.result()
                except BaseException:
                    persist()
                    raise
                self.count+=1;changed+=1
                if 'unexpected_exception' in r:
                    save_json(self.root/'UNEXPECTED_WORKER_ERROR.json',r);raise RuntimeError(r['unexpected_exception'])
                self.cache[sig]=r
                for i in pending[sig]:records[i]=copy.deepcopy(r)
                launch()
            if changed>=max(1,self.cfg['workers']) or time.monotonic()-last_save>30 or not running:
                persist();changed=0;last_save=time.monotonic()
        if any(r is None for r in records):
            persist();raise ChunkEnd()
        return records
