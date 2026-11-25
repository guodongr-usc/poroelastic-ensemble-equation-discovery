#!/usr/bin/env python3
# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from __future__ import annotations
import runtime
import argparse,os,time,json,traceback,fcntl,copy,csv
from pathlib import Path
from contextlib import contextmanager
from runtime import load_json,save_json,digest,environment
ROOT=Path(__file__).resolve().parent
INITIALIZATION = 'classical_biot_material_means_v1'

def identity(cfg):return digest({k:v for k,v in cfg.items() if k not in ('identity','resources','python','partition','account','mem','walltime')})
def check_config(cfg):
 if cfg.get('initialization') != INITIALIZATION:
  raise ValueError('This run uses a different initialization. Start a new search directory.')
 if identity(cfg)!=cfg['identity']:raise RuntimeError('Configuration identity mismatch')
@contextmanager
def lock(path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError as e:raise RuntimeError('Another writer is using '+str(path)) from e
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def status(root,state,**kw):
 save_json(Path(root)/'STATUS.json',dict(state=state,updated_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),host=os.uname().nodename,**kw))
def require_prepared(cfg):
 root=Path(cfg['outdir'])
 record=load_json(root/'PREPARE_COMPLETE.json')
 classical=load_json(root/'Q1_CLASSICAL_LIMIT.json') if (root/'Q1_CLASSICAL_LIMIT.json').exists() else load_json(root/'Q1_LIMIT_AUDIT.json')
 if not classical['passed'] or (cfg['profile']!='smoke' and not classical['production_native_gate']):
  raise RuntimeError('The classical-limit numerical check must pass before search')
 if record['identity']!=cfg['identity']:
  raise RuntimeError('Prepared inputs belong to another search configuration')
 for name in ('KNOWN_INPUTS.npz','TRAIN_ONLY.npy','INITIAL_CANDIDATE.json'):
  if not (root/'prepared'/name).is_file():
   raise FileNotFoundError(root/'prepared'/name)

def trial_root(cfg,i):return Path(cfg['outdir'])/'trials'/f'trial_{i:02d}'
def setup_trial(cfg,i):
 root=Path(cfg['outdir']);t=trial_root(cfg,i);t.mkdir(parents=True,exist_ok=True)
 p=t/'prepared';target=os.path.relpath(root/'prepared',t)
 if p.is_symlink():
  if os.readlink(p)!=target:raise RuntimeError('Unexpected input link')
 elif p.exists():raise RuntimeError('Unexpected local inputs')
 else:p.symlink_to(target)
 return t

def prepare_stage(cfg):
 from data_io import prepare,load_npz
 from candidates import initial_candidate
 from classical_limit import check_classical_limit
 root=Path(cfg['outdir'])
 with lock(root/'.prepare.lock'):
  if (root/'PREPARE_COMPLETE.json').exists():require_prepared(cfg);print('Using prepared training inputs');return
  prepare(cfg);save_json(root/'PREPARE_ENVIRONMENT.json',environment())
  known=load_npz(root/'prepared/KNOWN_INPUTS.npz')
  save_json(root/'prepared/INITIAL_CANDIDATE.json', initial_candidate(cfg['data'], known))
  if cfg['profile']=='smoke':check_classical_limit(root,grids=(21,),frames=2,substeps=1)
  else:check_classical_limit(root,grids=(21,51,101),frames=None,substeps=2)
  for i in range(cfg['n_trials']):setup_trial(cfg,i)
  save_json(root/'PREPARE_COMPLETE.json',dict(identity=cfg['identity'],passed=True))
  print('Training inputs prepared.',flush=True)

def stage_cfg(cfg,root,mode):
 c=copy.deepcopy(cfg);c['outdir']=str(root)
 c['workers']=int(cfg['resources']['coarse_cpus' if mode=='coarse' else 'fine_cpus'])
 c['worker_memory_gib']=float(cfg['resources']['coarse_worker_memory_gib' if mode=='coarse' else 'fine_worker_memory_gib'])
 allocated=int(os.environ.get('SLURM_CPUS_PER_TASK',c['workers']))
 if c['workers']>allocated:raise RuntimeError(f"Requested {c['workers']} workers but only {allocated} allocated")
 return c

def trial_stage(cfg,i,mode):
 from candidates import equations
 from structural_ga import initialize_population,evolve
 from search import Evaluator,ChunkEnd
 from data_io import load_npz
 require_prepared(cfg)
 if not 0<=i<cfg['n_trials']:raise ValueError('Invalid trial index')
 root=setup_trial(cfg,i);c=stage_cfg(cfg,root,mode)
 with lock(root/'.writer.lock'):
  done=root/('COARSE_DONE.json' if mode=='coarse' else 'TRIAL_RESULT.json')
  if done.exists():print(f'Trial {i:02d} {mode} already complete; retained.',flush=True);return
  status(root,'running_'+mode,trial=i,workers=c['workers']);start=time.time();ev=None
  try:
   deadline=time.monotonic()+cfg['chunk_seconds'];seed=int(cfg['seed'])+i
   baseline=load_json(root/'prepared/INITIAL_CANDIDATE.json')
   if mode=='coarse':
    base=baseline;grid=cfg['search_grid'];steps=cfg['time_substeps'];gens=cfg['generations'];label='COARSE'
   else:
    prior=load_json(root/'COARSE_DONE.json');base=prior['candidate'];grid=cfg['grid'];steps=cfg['fine_time_substeps'];gens=cfg['fine_generations'];label='FINE';seed+=900000
   pop=initialize_population(base,cfg['population'],seed,wide=(mode=='coarse'),baseline=baseline)
   print(f"TRIAL={i} MODE={mode} GRID={grid} WORKERS={c['workers']} MEMORY_CAP_GiB={c['worker_memory_gib']} SEED={seed}",flush=True)
   save_json(root/f'{mode}_ENVIRONMENT.json',environment())
   ev=Evaluator(c,deadline,grid=grid)
   winner=evolve(c,ev,label,pop,gens,steps,seed,structural_rate=.55 if mode=='coarse' else .3,scale=.7 if mode=='coarse' else .3)
   if mode=='fine':
    known=load_npz(root/'prepared/KNOWN_INPUTS.npz')
    save_json(root/'TRIAL_RESULT.json',dict(identity=cfg['identity'],trial_index=i,seed=cfg['seed']+i,completed=True,training_only=True,winner=winner,equations=equations(winner['candidate'],known['scales']),elapsed_this_stage=time.time()-start,old_monitoring_data_already_seen=True))
   status(root,'coarse_complete' if mode=='coarse' else 'training_complete',trial=i,workers=c['workers'],loss=winner['loss'])
  except ChunkEnd:status(root,'checkpointed_'+mode,trial=i);raise
  except BaseException:status(root,'failed_'+mode,trial=i,traceback=traceback.format_exc());raise
  finally:
   if ev is not None:ev.close()

def aggregate(cfg,no_forecast=False):
 from search import Evaluator
 from forecast import forecast
 root=Path(cfg['outdir']);require_prepared(cfg)
 with lock(root/'.aggregate.lock'):
  records=[]
  for i in range(cfg['n_trials']):
   path=trial_root(cfg,i)/'TRIAL_RESULT.json'
   if not path.exists():raise RuntimeError(f'Trial {i} incomplete; no model selection or held-out scoring')
   r=load_json(path)
   if r['identity']!=cfg['identity']:raise RuntimeError('Trial belongs to another search configuration')
   records.append(r)
  baseline=load_json(root/'prepared/INITIAL_CANDIDATE.json')
  cands=[baseline]+[r['winner']['candidate'] for r in records]
  c=stage_cfg(cfg,root,'fine');deadline=time.monotonic()+cfg['chunk_seconds'];ev=Evaluator(c,deadline,grid=cfg['grid'])
  try:scores=ev.evaluate(cands,cfg['selection_substeps'],'COMMON_GRID_TRAIN_ONLY_RANKING')
  finally:ev.close()
  rows=[]
  for i,s in enumerate(scores):
   row=dict(origin='classical_biot_baseline' if i==0 else 'refined_candidate',trial=None if i==0 else i-1,loss=s['loss'],success=s['success'],ICerr=s.get('initial_error'),IC_OK=s.get('initial_compatibility_passed'))
   for f in ('P','U1','U2'):
    for suffix in ('R2','relL2'):row[f+'_'+suffix]=(s.get('metrics') or {}).get(f+'_'+suffix)
   rows.append(row)
  with (root/'TRAIN_RANKING.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(sorted(rows,key=lambda r:r['loss']))
  valid=[(i,s) for i,s in enumerate(scores) if s['success']]
  if not valid:raise RuntimeError('No candidate passed the common-grid numerical solve')
  compatible=[x for x in valid if x[1]['initial_compatibility_passed']]
  i,best=min(compatible or valid,key=lambda x:x[1]['loss'])
  chosen=dict(identity=cfg['identity'],selected_trial=None if i==0 else i-1,selected_origin=rows[i]['origin'],winner=best,selection_is_diagnostic_only=not bool(compatible),heldout_used_for_selection=False)
  if (root/'SELECTED_TRAINING_CANDIDATE.json').exists():
   old=load_json(root/'SELECTED_TRAINING_CANDIDATE.json')
   if old['winner']['candidate']!=chosen['winner']['candidate']:raise RuntimeError('Refuse changing an already selected model')
  else:save_json(root/'SELECTED_TRAINING_CANDIDATE.json',chosen)
  print('TRAINING-ONLY SELECTION:',chosen['selected_origin'],chosen['selected_trial'],flush=True)
  if no_forecast:return
  forecast(cfg,best,deadline)
  final=load_json(root/'FINAL_SUMMARY.json');final.update(round=4,selected_origin=chosen['selected_origin'],selected_trial=chosen['selected_trial'],prior_monitoring_already_seen=True)
  save_json(root/'FINAL_SUMMARY.json',final)

def observed_initial_stage(cfg):
 from frozen_solver import run_forward
 from evaluation_6935953 import evaluate,plot
 root=Path(cfg['outdir'])
 output=root/'observed_initial'
 output.mkdir(parents=True,exist_ok=True)
 with lock(root/'.observed_initial.lock'):
  selected=load_json(root/'SELECTED_TRAINING_CANDIDATE.json')
  if selected['identity']!=cfg['identity']:
   raise RuntimeError('The selected equation belongs to another search')
  candidate=selected['winner']['candidate']
  model_path=output/'frozen_candidate.json'
  result_file=output/'run_result.json'
  if result_file.exists():
   completed=load_json(result_file)
   if (completed['identity']!=cfg['identity'] or load_json(model_path)!=candidate):
    raise RuntimeError('The output directory contains results for another fitted equation')
   return completed
  save_json(model_path,candidate)
  prediction=output/'fpde_prediction.npz'
  info=run_forward(output,prediction,candidate_path=model_path,known_path=root/'prepared/KNOWN_INPUTS.npz')
  paths={'fpde':prediction}
  evaluate(ROOT.parents[1],output,['fpde'],prediction_paths=paths,data_path=cfg['data'])
  plot(ROOT.parents[1],output,['fpde'],prediction_paths=paths,data_path=cfg['data'])
  record=dict(identity=cfg['identity'],selected_trial=selected['selected_trial'],
              selected_origin=selected['selected_origin'],forward=info)
  save_json(result_file,record)
  return record

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--mode',choices=('prepare','coarse','fine','aggregate','observed'),required=True);ap.add_argument('--trial',type=int);ap.add_argument('--allow-local',action='store_true');ap.add_argument('--no-forecast',action='store_true');a=ap.parse_args()
 if not os.environ.get('SLURM_JOB_ID') and not a.allow_local:raise RuntimeError('Use Slurm, not a login node')
 if os.environ.get('PYTHONHASHSEED')!='0':raise RuntimeError('PYTHONHASHSEED must be 0')
 cfg=load_json(a.config);check_config(cfg)
 if a.mode=='prepare':prepare_stage(cfg)
 elif a.mode in ('coarse','fine'):trial_stage(cfg,a.trial if a.trial is not None else int(os.environ['SLURM_ARRAY_TASK_ID']),a.mode)
 elif a.mode=='aggregate':aggregate(cfg,a.no_forecast or cfg.get('observed_initial_output',False))
 else:observed_initial_stage(cfg)
if __name__=='__main__':
 try:main()
 except Exception as e:
  from search import ChunkEnd
  if isinstance(e,ChunkEnd):print('Checkpoint saved; resume required.',flush=True);raise SystemExit(75)
  traceback.print_exc();raise SystemExit(1)
