# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from pathlib import Path
import csv, io, time
import numpy as np
from runtime import load_json,save_json
from data_io import load_npz,save_npz,scoring_truth
from candidates import parse,equations
from solver_cpu import CPUSolver
from search import metrics,ChunkEnd
from model_config import MODEL

def _convergence(fine,coarse):
    delta=fine-fine[:,0:1]
    absolute=[];dynamic=[]
    for j in range(3):
        num=np.linalg.norm(fine[j]-coarse[j]);den=np.linalg.norm(fine[j])
        absolute.append(float(num/max(den,1e-30)))
        dynamic.append(float(num/max(np.linalg.norm(delta[j]),1e-8*den,1e-30)))
    return dict(absolute_relative_difference=absolute,dynamic_relative_difference=dynamic)

def forecast(cfg,winner,deadline=None):
    root=Path(cfg['outdir']);known=load_npz(root/'prepared/KNOWN_INPUTS.npz');k=int(known['n_train'])
    c=winner['candidate'];p=parse(c)
    solver=CPUSolver(known,cfg['grid'],rtol=1e-9,max_seconds=21600)
    frozen=root/'FROZEN_MODEL.json'
    frozen_record=dict(identity=cfg['identity'],candidate=c,
        equations=equations(c,known['scales']),normalization=load_json(root/'NORMALIZATION.json'),
        initial_compensation=False,background_function_from_IC=False,
        physical_loads={'rho_b':cfg['rho_b'],'rho_f':cfg['rho_f'],'gravity_m_s2':[0.,-cfg['gravity']]},
        training_calls_after_freeze=0,future_truth_used=False,substeps=int(winner['substeps']),
        pressure_operator='conservative two-sided RL-integral Darcy flux; not old shifted-GL pressure stencil',
        mechanics_initialization='unchanged physical scaling: K4 U0=(fg+alpha_B*P_scale*B4 P0)/C; no IC residual compensation')
    if frozen.exists():
        existing=load_json(frozen)
        if existing['identity']!=cfg['identity'] or existing['candidate']!=c:raise RuntimeError('Frozen model mismatch')
    else:save_json(frozen,frozen_record)
    def solve(s):
        stem=root/'.states'/f'full_noB0_s{s:04d}'
        ap=stem.with_suffix('.npz');jp=stem.with_suffix('.json')
        if ap.exists() and jp.exists():
            r=load_json(jp)
            if r['identity']!=cfg['identity']:raise RuntimeError('Cached prediction belongs to another run')
            if r.get('candidate')==c:
                return load_npz(ap)['prediction'],r['solver']
        if deadline is not None and time.monotonic()>deadline:raise ChunkEnd()
        print(f'FROZEN FULL-HISTORY IC/BC solve: substeps={s}; no future truth read; no B0',flush=True)
        pred,info=solver.forward(p,known['theta'],s)
        if not np.array_equal(pred[0,0],known['initial_normalized'][0,solver.ids]):raise RuntimeError('Initial pressure mismatch: never overwrite or conceal')
        save_npz(ap,prediction=pred)
        save_json(jp,dict(identity=cfg['identity'],candidate=c,solver=info))
        return pred,info
    s=int(winner['substeps']);pred,info=solve(s);history=[];passed=False
    while 2*s<=cfg['max_time_substeps']:
        fine,info=solve(2*s);err=_convergence(fine,pred)
        history.append(dict(coarse_substeps=s,fine_substeps=2*s,**err));s*=2;pred=fine
        passed=max(err['dynamic_relative_difference'])<=cfg['time_tolerance']
        save_json(root/'FORECAST_TIME_CONVERGENCE.json',dict(passed=passed,history=history,tolerance=cfg['time_tolerance'],comparison='prediction vs prediction ONLY',future_truth_read=False))
        print('TIME REFINEMENT',history[-1],flush=True)
        if passed:break

    physical=pred*known['scales'][:,None,None]
    save_npz(root/'FROZEN_PREDICTION.npz',prediction_normalized=pred,absolute_prediction=physical,
        P_pred_Pa=physical[0],Ux_pred_m=physical[1],Uy_pred_m=physical[2],
        representation=np.array('absolute_prediction: physical units; prediction_normalized: scale-only'),
        scales=known['scales'],times_s=known['times_s'],node_ids=solver.ids,
        substeps=np.int64(s))
    save_json(root/'FORECAST_SOLVER_AUDIT.json',dict(full_horizon_time_convergence_passed=passed,
        initial_compensation=False,continuous_from_t0=True,training_boundary_reset=False,solver=info,initial_displacement_condition_satisfied=max(info['initial_mechanics_field_relL2'])<=cfg['ic_tolerance'],
        physical_IC_error_max=np.max(np.abs(physical[:,0]-known['initial_absolute'][:,solver.ids]),axis=1).tolist(),
        future_truth_used=False,substeps=s))
    if not passed:raise RuntimeError('Prediction-only time refinement did not pass. Predictions retained; no future scoring. Do not select tolerance by test performance.')
    truth_full=scoring_truth(cfg);truth=truth_full[:,:,solver.ids]
    truth_physical=truth*known['scales'][:,None,None]

    initial=known['initial_normalized'][:,solver.ids][:,None,:]
    pred_resp=pred-initial;true_resp=truth-initial
    free=[~solver.pmask,~solver.masks['u1_mask'].ravel(),~solver.masks['u2_mask'].ravel()]
    scores={};rscores={}
    for name,sl in [('train',slice(1,k)),('test',slice(k,None)),('last_frame',slice(-1,None))]:
        scores[name]=metrics(pred[:,sl],truth[:,sl]);rscores[name]=metrics(pred_resp[:,sl],true_resp[:,sl])
        if name!='last_frame':
            scores[name+'_free_dofs']=metrics(pred[:,sl],truth[:,sl],free)
            rscores[name+'_free_dofs']=metrics(pred_resp[:,sl],true_resp[:,sl],free)
    save_npz(root/'FORWARD_VS_TRUTH.npz',absolute_prediction=physical,absolute_truth=truth_physical,
         prediction_normalized=pred,truth_normalized=truth,scales=known['scales'],
         times_s=known['times_s'],node_ids=solver.ids,node_xy_m=np.asarray([[known['x_m'][i%101],known['y_m'][i//101]] for i in solver.ids]),
         n_train=np.int64(k),representation=np.array('absolute_prediction and absolute_truth are Pa,m,m; no additional decoding'))
    rows=[]
    for i,t in enumerate(known['times_s']):
        am=metrics(pred[:,i:i+1],truth[:,i:i+1]);rm=metrics(pred_resp[:,i:i+1],true_resp[:,i:i+1]) if i else {}
        row=dict(frame=i,time_s=float(t),time_days=float(t/86400),split='initial' if i==0 else ('train' if i<k else 'test'))
        for j,f in enumerate(('P','U1','U2')):
            row[f+'_absolute_R2']=am[f+'_R2'];row[f+'_absolute_relL2']=am[f+'_relL2'];row[f+'_response_R2']=rm.get(f+'_R2',float('nan'));row[f+'_response_relL2']=rm.get(f+'_relL2',float('nan'))
            row[f+'_RMSE_physical']=float(np.sqrt(np.mean((physical[j,i]-truth_physical[j,i])**2)))
        rows.append(row)
    with (root/'PER_TIME_METRICS.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    static=np.repeat(initial,pred.shape[1],axis=1)
    ic_pass=max(info['initial_mechanics_field_relL2'])<=cfg['ic_tolerance']
    report=dict(identity=cfg['identity'],model=MODEL,scores=scores,response_scores=rscores,initial_displacement_condition_satisfied=ic_pass,ic_tolerance=cfg['ic_tolerance'],validation_status='IC_COMPATIBLE_DIAGNOSTIC_COMPLETE' if ic_pass else 'INITIAL_DISPLACEMENT_NOT_FITTED',
        initial_state_persistence_test=metrics(static[:,k:],truth[:,k:]),
        equations=equations(c,known['scales']),training_calls_after_freeze=0,
        test_used_for_selection=False,initial_compensation=False,solver=info,substeps=s,
        initial_mechanics_relative_mismatch=info['initial_mechanics_relative_mismatch'])
    save_json(root/'FINAL_SUMMARY.json',report)
    lines=[f'# Model {MODEL}: NO-B0 frozen IC/BC temporal test','',
        'Absolute state, scale-only normalization. No initial-residual source; physical gravity retained.',
        'No retraining after freeze; repeated temporal monitoring, NOT new blind validation. Held-out truth used only for scoring.',
        f"Initial quasistatic mechanics mismatch (diagnostic; not cancelled): {info['initial_mechanics_relative_mismatch']:.8g}",
        f"Initial displacement compatibility: {ic_pass} at relative L2 threshold {cfg['ic_tolerance']}. If False, these are diagnostic forecasts, NOT a PDE validated for the full observed IC.",
        'Pressure starts at the observed P0. Displacement starts at candidate mechanical equilibrium and is not replaced by observed U0. Initial data mismatch was included in fitness.','']
    for label,ss in [('Absolute-field R2',scores),('Response-field R2 (post-hoc only)',rscores)]:
        lines += ['## '+label,'','|Split|P|Ux|Uy|Macro|','|---|---:|---:|---:|---:|']
        for name in ('train','test','last_frame'):
            rec=ss[name];lines.append('|'+name+'|'+'|'.join(f"{rec[key]:.7f}" for key in ('P_R2','U1_R2','U2_R2','macro_R2'))+'|')
        lines.append('')
    lines+=['## Absolute-field relative L2','', '|Split|P|U1|U2|Joint|','|---|---:|---:|---:|---:|']
    for name in ('train','test','last_frame'):
        rec=scores[name];lines.append('|'+name+'|'+'|'.join(f"{rec[key]:.7f}" for key in ('P_relL2','U1_relL2','U2_relL2','joint_relL2'))+'|')
    lines+=['','## Equations','']+[k+': '+str(v) for k,v in report['equations'].items()]
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
