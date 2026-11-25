#!/usr/bin/env python3
# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from __future__ import annotations
import argparse,os,sys,re,math,shlex,subprocess,time,json
from pathlib import Path
from runtime import save_json,load_json,clean_env
from workflow import ROOT,INITIALIZATION,identity,check_config,lock

def command(args):
 print('$ '+shlex.join(map(str,args)),flush=True)
 r=subprocess.run(list(map(str,args)),text=True,capture_output=True,env=clean_env())
 if r.stdout:print(r.stdout.strip(),flush=True)
 if r.stderr:print(r.stderr.strip(),file=sys.stderr,flush=True)
 if r.returncode:raise RuntimeError(f'Command exited {r.returncode}: {args[0]}')
 return r.stdout.strip()
def mem_gib(text):
 m=re.fullmatch(r'(\d+(?:\.\d+)?)([MGT])',text.upper())
 if not m:raise ValueError('Use memory with units, e.g. 240G')
 return float(m[1])*{'M':1/1024,'G':1,'T':1024}[m[2]]
def assert_idle(root):
 file=Path(root)/'JOB_IDS.json'
 if not file.exists():return
 ids={str(g[k]) for g in load_json(file).get('submissions',[]) for k in ('prepare','coarse','fine','aggregate','observed') if g.get(k)}
 raw=command(['squeue','-u',os.environ['USER'],'-h','-r','-o','%i'])
 active={x.strip().split('_')[0] for x in raw.splitlines()}
 if ids&active:raise RuntimeError('Existing jobs still active: '+','.join(sorted(ids&active)))
def schedule(cfg,dry_run=False):
 root=Path(cfg['outdir']);res=cfg['resources'];excluded=[]
 if not dry_run:
  raw=command(['sinfo','-p',cfg['partition'],'-N','-h','-o','%N|%G'])
  excluded=sorted({x.split('|')[0].strip() for x in raw.splitlines() if '|' in x and 'gpu' in x.split('|',1)[1].lower()})
 group=dict(submitted_unix=time.time(),prepare=None,coarse=None,fine=None,aggregate=None,resources=res,excluded_gpu_nodes=excluded)
 ledger=load_json(root/'JOB_IDS.json') if (root/'JOB_IDS.json').exists() else dict(submissions=[])
 if not dry_run:ledger['submissions'].append(group);save_json(root/'JOB_IDS.json',ledger)
 def job(mode,cpus,dep=None,array=False):
  memory='64G' if mode=='prepare' else cfg['mem'];limit='04:00:00' if mode=='prepare' else cfg['walltime']
  args=['sbatch','--parsable','--job-name=Q4_'+mode,'--partition='+cfg['partition'],'--nodes=1','--ntasks=1',f'--cpus-per-task={cpus}','--mem='+memory,'--time='+limit,'--requeue','--kill-on-invalid-dep=yes','--chdir='+str(ROOT),'--export=ALL','--open-mode=append']
  pattern=f'{mode}_%A_%a' if array else f'{mode}_%j'
  args+=['--output='+str(root/'logs'/f'{pattern}.out'),'--error='+str(root/'logs'/f'{pattern}.err')]
  if cfg.get('account'):args+=['--account='+cfg['account']]
  if excluded:args+=['--exclude='+','.join(excluded)]
  if array:args+=[f"--array=0-{cfg['n_trials']-1}%{cfg['n_trials']}"]
  if dep:args+=['--dependency='+dep]
  args += [str(ROOT/'launch.sbatch'),str(root/'RUN_CONFIG.json'),cfg['python'],str(ROOT),mode]
  if dry_run:
   print('DRY RUN: '+shlex.join(args));j=str(900001+len([v for k,v in group.items() if k in ('prepare','coarse','fine','aggregate','observed') and v]))
  else:
   j=command(args).split(';')[0]
   if not j.isdigit():raise RuntimeError('Unexpected sbatch job id: '+j)
  group[mode]=j
  if not dry_run:save_json(root/'JOB_IDS.json',ledger)
  return j

 valid=False
 if (root/'PREPARE_COMPLETE.json').exists():
  from workflow import require_prepared
  require_prepared(cfg);valid=True
 prepare=job('prepare',4) if not valid else None
 coarse=job('coarse',res['coarse_cpus'],'afterok:'+prepare if prepare else None,True)
 fine=job('fine',res['fine_cpus'],'aftercorr:'+coarse,True)
 selected=job('aggregate',res['fine_cpus'],'afterok:'+fine)
 if cfg.get('observed_initial_output',False):
  job('observed',1,'afterok:'+selected)
 print('SUBMISSION:',group,flush=True)
 print('Coarse: 10 parallel-eligible trials; native: matching trial IDs aftercorr. No trial-to-trial serial chain.',flush=True)
 return group

def parser():
 p=argparse.ArgumentParser(description='CARC Slurm submission, separate 64-core coarse / 16-core native stages.')
 p.add_argument('--outdir',type=Path,default=ROOT/'results');p.add_argument('--data',type=Path)
 p.add_argument('--python',default=sys.executable)
 p.add_argument('--cpus',type=int,default=64);p.add_argument('--fine-cpus',type=int,default=16)
 p.add_argument('--mem',default='240G');p.add_argument('--partition',default='jha');p.add_argument('--account');p.add_argument('--walltime',default='2-00:00:00')
 p.add_argument('--trials',type=int,default=10,choices=[10]);p.add_argument('--profile',choices=['production','smoke'],default='production')
 p.add_argument('--search-grid',type=int,choices=[21,51,101],default=51);p.add_argument('--final-grid',type=int,choices=[101],default=101)
 p.add_argument('--population',type=int,default=64);p.add_argument('--generations',type=int,default=30);p.add_argument('--fine-generations',type=int,default=6)
 p.add_argument('--time-substeps',type=int,default=2);p.add_argument('--fine-time-substeps',type=int,default=4)
 p.add_argument('--field-weights',type=float,nargs=3,default=[.5,.25,.25]);p.add_argument('--ic-weight',type=float,default=.5);p.add_argument('--ic-tolerance',type=float,default=.05)
 p.add_argument('--coarse-worker-memory-gib',type=float,default=2.75);p.add_argument('--fine-worker-memory-gib',type=float,default=10.)
 p.add_argument('--seed',type=int,default=20260922);p.add_argument('--candidate-seconds',type=float,default=1800)
 p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
 p.add_argument('--observed-initial-output',action='store_true')
 return p

def create_config(a):
 data=(a.data or ROOT.parents[1]/'data/443EA_CLEAN_ICSTRAIN0.1_P_U1_U2.npz').expanduser().resolve()
 if not data.is_file():raise FileNotFoundError(data)
 cfg=dict(profile=a.profile,data=str(data),outdir=str(a.outdir),case_index=0,train_fraction=.8,initialization=INITIALIZATION,
  rho_b=2242.609582,rho_f=1000.,gravity=9.81,initial_compensation=False,n_trials=10,
  search_grid=a.search_grid,grid=a.final_grid,population=a.population,generations=a.generations,fine_generations=a.fine_generations,
  time_substeps=a.time_substeps,fine_time_substeps=a.fine_time_substeps,selection_substeps=max(8,a.fine_time_substeps),
  max_time_substeps=128,time_tolerance=.01,seed=a.seed,field_weights=a.field_weights,ic_weight=a.ic_weight,ic_tolerance=a.ic_tolerance,penalty=.0005,
  candidate_seconds=a.candidate_seconds,chunk_seconds=150000,
  python=str(Path(a.python).expanduser().absolute()),partition=a.partition,account=a.account,mem=a.mem,walltime=a.walltime,
  resources=dict(coarse_cpus=a.cpus,fine_cpus=a.fine_cpus,coarse_worker_memory_gib=a.coarse_worker_memory_gib,fine_worker_memory_gib=a.fine_worker_memory_gib))
 if a.profile=='smoke':cfg.update(search_grid=21,grid=21,population=4,generations=1,fine_generations=1,time_substeps=1,fine_time_substeps=1,selection_substeps=1)
 if a.observed_initial_output:cfg['observed_initial_output']=True
 cfg['identity']=identity(cfg);return cfg

def main():
 a=parser().parse_args();a.outdir=a.outdir.expanduser().resolve()
 if not Path(a.python).is_file():raise FileNotFoundError('Python interpreter missing: '+a.python)
 if not 1<=a.cpus<=64 or not 1<=a.fine_cpus<=16:raise ValueError('Coarse CPUs 1..64; native CPUs 1..16 to limit concurrent factorizations')
 if a.population<4 or a.generations<1 or a.fine_generations<1:raise ValueError('Invalid search budget')
 if not all(math.isfinite(v) and v>0 for v in a.field_weights) or not math.isclose(sum(a.field_weights),1.,abs_tol=1e-10):raise ValueError('Weights must be positive and sum to 1')
 mem=mem_gib(a.mem)
 for cpu,cap in ((a.cpus,a.coarse_worker_memory_gib),(a.fine_cpus,a.fine_worker_memory_gib)):
  if cap<=0 or cpu*cap+12>mem:raise ValueError('Per-worker caps plus 12GiB reserve exceed job memory')
 a.outdir.mkdir(parents=True,exist_ok=True);(a.outdir/'logs').mkdir(exist_ok=True)
 for key in list(os.environ):
  if key.startswith('SBATCH_'):os.environ.pop(key)
 with lock(a.outdir/'.submission.lock'):
  conf=a.outdir/'RUN_CONFIG.json'
  if conf.exists():
   if not a.resume:raise RuntimeError('Existing run. Use --resume; no overwrite')
   cfg=load_json(conf);check_config(cfg)
   if not a.dry_run:assert_idle(a.outdir)
   cfg.update(python=str(Path(a.python).expanduser().absolute()),partition=a.partition,account=a.account,mem=a.mem,walltime=a.walltime)
   cfg['resources'].update(coarse_cpus=a.cpus,fine_cpus=a.fine_cpus,coarse_worker_memory_gib=a.coarse_worker_memory_gib,fine_worker_memory_gib=a.fine_worker_memory_gib)
  else:
   if a.resume:raise RuntimeError('No existing configuration to resume')
   cfg=create_config(a)
  if a.dry_run:
   print('Config is NOT written in dry-run mode. Commands below use placeholder job IDs.')
  else:save_json(conf,cfg)
  print('RESULTS:',a.outdir);print('RESOURCES:',cfg['resources']);print('MECHANICS ORDERS: 0.1--3.0, Q1 exact classical anchor, optional library structurally searched.')
  schedule(cfg,a.dry_run)
if __name__=='__main__':main()
