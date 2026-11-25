# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# Method reference for the two-step field and equation fits:
# Y. Jalalian, J. F. Osorio Ramirez, A. Hsu, B. Hosseini and H. Owhadi (2025),
# "Data-Efficient Kernel Methods for Learning Differential Equations and Their
# Solution Operators: Algorithms and Error Analysis", sec. 2.1, eqs. (5)-(8).
# https://arxiv.org/abs/2503.01036v2
# This study fits a discrete solution kernel followed by a ten-term closure
# of the fixed Biot equations for the 443-member ensemble mean.

from pathlib import Path
import json
import time
import numpy as np
import scipy.sparse as sp
from .data import metadata,read_prefix,time_slice,free_ids
from .solution import fit_internal_solution
from .operators import build_system,context,solve_total
from .library import HEADS,build_library,embed_biot,assemble
from .regression import fit_centered

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'outputs/kernel/fit'
GRADIENT_AMPLITUDE=.1
RIDGE=.1

def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def log(value):print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


def discrepancy(a,b):
    norm=sp.linalg.norm if sp.issparse(a) else np.linalg.norm
    return float(norm(a-b)/max(float(norm(b)),1e-30))


def prior_rowweights(A,M,times,fixed,n):
    a=np.asarray(abs(A).sum(axis=1)).ravel();m=np.asarray(abs(M).sum(axis=1)).ravel()
    weights=np.broadcast_to(a,(len(times),len(a))).copy()
    weights[1:]+=m[None,:]/np.diff(times)[:,None]
    weights=1/np.maximum(weights,max(float(a.max())*1e-12,1e-16))
    weights[:,fixed]=0.;weights[0,2*n:]=0.
    return weights


def prepare():
    if OUT.exists():raise FileExistsError('Fresh output required: '+str(OUT))
    OUT.mkdir()
    write(OUT/'protocol.json',dict(dataset='443EA_CLEAN_ICSTRAIN0.1_P_U1_U2',case=1,
        train_output_frames=list(range(39)),heldout_output_frames=list(range(39,49)),
        gradient_amplitude=GRADIENT_AMPLITUDE,ridge=RIDGE,fixed_material=True))
    fullmeta=metadata();y=read_prefix(38);ids=free_ids(fullmeta)
    scales=np.asarray([np.std(y[h][0,1:,ids[h]]) for h in HEADS])
    original=build_system(time_slice(fullmeta,38),scales)
    m={k:v for k,v in fullmeta.items() if not k.startswith('mean_')}
    times=m['times_s'][:39];ctx=context(m,times,scales,original.row_scales)
    libs=build_library(ctx['engine'],scales,ctx['rows'])
    tau=original.engine.assemble(original.coefficients).tau
    prior=embed_biot(libs,original.coefficients,tau)
    assert all(np.count_nonzero(c[:,:10])==0 for c in prior)
    A,M,offset=assemble(libs,prior)
    check=dict(A=discrepancy(A,original.A_static),M=discrepancy(M,original.M),
        loads=discrepancy(ctx['rhs']-offset,np.stack([original.rhs(i) for i in range(39)])))
    assert max(check.values())<2e-12,check
    rowweights=prior_rowweights(A,M,times,ctx['fixed'],ctx['n'])
    obs=np.stack([y[h][0] for h in HEADS],axis=1)
    fixed=np.ones((3,ctx['n']),dtype=bool)
    for j,h in enumerate(HEADS):fixed[j,ctx['freeids'][h]]=False
    known=(ctx['known']*np.repeat(scales,ctx['n'])).reshape(3,ctx['n'])
    fields,solaudit=fit_internal_solution(obs,times,np.arange(39),m['x_m'],m['y_m'],fixed,known,scales)
    state=np.concatenate([fields[h]/scales[j] for j,h in enumerate(HEADS)],axis=1)
    for j,h in enumerate(HEADS):np.testing.assert_array_equal(fields[h][0],y[h][0,0])
    np.savez_compressed(OUT/'known_inputs.npz',**m)
    np.savez_compressed(OUT/'true_initial.npz',**{h:y[h][0,0] for h in HEADS})
    np.savez_compressed(OUT/'solution.npz',state=state,times_s=times,state_scales=scales,
        row_scales=ctx['rows'],rowweights=rowweights)
    np.savez_compressed(OUT/'mean_biot_prior.npz',native_mean_coefficients=original.coefficients,
        tau=tau,offset=offset,**{h:v for h,v in zip(HEADS,prior)})
    log(dict(stage='fresh_Solution_complete',embedding=check,
        R2={h:solaudit['observed_value_fit_unconstrained'][h]['R2'] for h in HEADS}))
    return solaudit


def inputs():
    with np.load(OUT/'known_inputs.npz') as z:m={k:z[k] for k in z.files}
    with np.load(OUT/'solution.npz') as z:
        state=z['state'];times=z['times_s'];scales=z['state_scales'];rows=z['row_scales'];rw=z['rowweights']
    with np.load(OUT/'mean_biot_prior.npz') as z:prior=[z[h] for h in HEADS]
    ctx=context(m,times,scales,rows)
    return m,state,times,scales,rows,rw,prior,ctx,build_library(ctx['engine'],scales,rows)


def train():
    m,state,times,scales,rows,rw,prior,ctx,libs=inputs()
    cfg=dict(name='fixed_biot_control',ridge=RIDGE,gradient_amplitude=GRADIENT_AMPLITUDE,fixed_biot=True)
    name=cfg['name']
    folder=OUT/name
    folder.mkdir()
    start=time.perf_counter()
    fit=fit_centered(libs,state,times,ctx['rhs'],ctx['freeids'],rw,RIDGE,prior,
                     gradient_amplitude=GRADIENT_AMPLITUDE)
    A,M,offset=assemble(libs,fit['assembly_coefficients'])
    fit['audit'].update(configuration=cfg,elapsed_seconds=time.perf_counter()-start)
    np.savez_compressed(folder/'kernel.npz',state_scales=scales,row_scales=rows,offset=offset,
        **{f'weights_{h}':w for h,w in zip(HEADS,fit['weights'])},
        **{f'feature_scales_{h}':w for h,w in zip(HEADS,fit['feature_scales'])},
        **{f'coefficients_{h}':w for h,w in zip(HEADS,fit['assembly_coefficients'])},
        **{f'prior_coefficients_{h}':w for h,w in zip(HEADS,prior)},
        **{f'features_{h}':np.array([f.name for f in l]) for h,l in zip(HEADS,libs)})
    sp.save_npz(folder/'A_total.npz',A);sp.save_npz(folder/'M_total.npz',M)
    write(folder/'training.json',fit['audit'])
    log(dict(stage='equation_frozen',name=name,objective=fit['audit']['objective']))



def forward(name='fixed_biot_control'):
    with np.load(OUT/'known_inputs.npz') as z:m={k:z[k] for k in z.files}
    with np.load(OUT/'true_initial.npz') as z:initial={h:z[h] for h in HEADS}
    folder=OUT/name
    prediction_file=folder/'full_forward.npz'
    if prediction_file.exists():raise FileExistsError(f'Forward prediction already exists: {prediction_file}')
    with np.load(folder/'kernel.npz') as z:scales=z['state_scales'];rows=z['row_scales'];offset=z['offset']
    start=time.perf_counter();log(dict(stage='forward_start',name=name))
    pred,summary=solve_total(sp.load_npz(folder/'A_total.npz'),sp.load_npz(folder/'M_total.npz'),
        offset,m,initial,scales,rows)
    sampled=pred[m['output_solver_step_indices']]
    for j,h in enumerate(HEADS):np.testing.assert_array_equal(sampled[0,j],initial[h])
    np.savez_compressed(prediction_file,fields=sampled,times_s=m['times_s'])
    summary['elapsed_seconds']=time.perf_counter()-start
    log(dict(stage='forward_finished',name=name,**summary))
    return summary
