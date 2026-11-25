# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from __future__ import annotations
import io, zipfile
from pathlib import Path
import numpy as np
from runtime import atomic_bytes,save_json,load_json
from uniform48_schema import inspect_npz,EXPECTED_SHAPE,chronological_split
FIELDS=('P','U1','U2')

def _training_prefix_only(path,key,count):


    from uniform48_schema import read_header
    with zipfile.ZipFile(path) as archive, archive.open(key+'.npy') as stream:
        h=read_header(stream);dtype=np.dtype(h['descr'])
        if tuple(h['shape'])!=(1,49,10201) or h['fortran_order']:raise ValueError('Unexpected layout')
        needed=count*10201*dtype.itemsize;raw=stream.read(needed)
        if len(raw)!=needed:raise IOError('Truncated training prefix')
        return np.frombuffer(raw,dtype=dtype).reshape(count,10201).astype(float)


def save_npz(path,**arr):
    f=io.BytesIO();np.savez_compressed(f,**arr);atomic_bytes(path,f.getvalue())
def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k]) for k in z.files}

def _masks(x,y):
    X,Y=np.meshgrid(x,y); xy=np.c_[X.ravel(),Y.ravel()]
    pm=((np.isclose(xy[:,0],0)|np.isclose(xy[:,0],500)) & np.isin(np.round(xy[:,1],8),np.array([165.,200.,235.,265.,300.,335.])))
    u1=np.isclose(xy[:,1],0)|np.isclose(xy[:,0],0)
    u2=np.isclose(xy[:,1],0)
    if pm.sum()!=12 or u1.sum()!=201 or u2.sum()!=101: raise RuntimeError('Known BC topology audit failed')
    return pm,u1,u2

def prepare(cfg):
    root=Path(cfg['outdir']);folder=root/'prepared';folder.mkdir(parents=True,exist_ok=True);marker=root/'TIME_SPLIT.json'
    if marker.exists():
        d=load_json(marker)
        if d['identity']!=cfg['identity']: raise RuntimeError('Split/config mismatch; use a new output dir')
        return d
    audit=inspect_npz(cfg['data'],cfg['train_fraction']); path=Path(cfg['data'])
    with np.load(path,allow_pickle=False) as z:
        times=np.asarray(z['times_s'],float); k=chronological_split(times,cfg['train_fraction'])['train_count']

        if int(cfg['case_index'])!=0: raise ValueError('This 500EA archive contains one forcing EA; case-index must be 0')
        raw=np.stack([_training_prefix_only(path,f+'_EA',k) for f in FIELDS])
        if raw.shape!=(3,k,10201) or not np.isfinite(raw).all(): raise ValueError('Invalid absolute training fields')
        initial=raw[:,0].copy()

        scales=np.sqrt(np.mean(raw*raw,axis=(1,2)))
        if np.any(~np.isfinite(scales)) or np.any(scales<=0): raise ValueError('Invalid training-only absolute scales')
        truth=raw/scales[:,None,None]
        x=np.asarray(z['x_m'],float);y=np.asarray(z['y_m'],float);pm,u1m,u2m=_masks(x,y)

        if not np.allclose(raw[0,1:][:,pm],1e7,rtol=0,atol=1e-6): raise ValueError('Known 10 MPa drainage BC disagrees with training prefix')
        if np.max(np.abs(raw[1,:,u1m]))>1e-10 or np.max(np.abs(raw[2,:,u2m]))>1e-10: raise ValueError('Known displacement essential BC mismatch')
        q=np.asarray(z['forcing_q_matrix_m2_s'][:,0],float);q_total=float(z['q_total_m2_s'])
        if q.shape!=(10201,) or np.any(q<0) or not np.isclose(q.sum(),q_total,rtol=1e-12,atol=1e-15): raise ValueError('Source metadata mismatch')
        q_scale=max(float(np.max(q)/25.0),1e-30); q_density=q/(25.0*q_scale)
        known=dict(rho_b=np.float64(cfg['rho_b']),rho_f=np.float64(cfg['rho_f']),gravity=np.array([0.,-cfg['gravity']]),internal_theta=np.asarray(z['solver_times_s'],float)/times[k-1],times_s=times,theta=times/times[k-1],x_m=x,y_m=y,p_mask=pm,u1_mask=u1m,u2_mask=u2m,
                   p_boundary_positive=np.float64(1e7/scales[0]),q_density=q_density,q_nodal_m2_s=q,scales=scales,q_scale=np.float64(q_scale),
                   time_scale_s=np.float64(times[k-1]),initial_absolute=initial,initial_normalized=initial/scales[:,None],n_train=np.int64(k),
                   case_index=np.int64(0),case_id=np.int64(np.asarray(z['case_id'])[0]),source_xy_m=np.asarray(z['case_source_center_xy_m'][0],float),
                   source_total_m2_s=np.float64(q_total),absolute_training=np.bool_(True))
    save_npz(folder/'KNOWN_INPUTS.npz',**known)
    b=io.BytesIO();np.save(b,truth,allow_pickle=False);atomic_bytes(folder/'TRAIN_ONLY.npy',b.getvalue())
    split=chronological_split(times,cfg['train_fraction'])
    rec=dict(identity=cfg['identity'],case_index=0,**split,reference_loaded=False,other_EAs_used=False,future_truth_read=False,
             target='ABSOLUTE EA fields; scale-only normalization; NO EA(t)-EA(t0) subtraction',
             ensemble_size=audit['nreal'],filtered_ensemble=audit['filtered'],
             initial_condition='exact pressure IC; displacement IC computed by candidate Q1 equilibrium and fitted against observed U0',coordinate='theta=t/train_end; xi=x/Lx; eta=y/Ly')
    save_json(marker,rec)
    save_json(root/'PHYSICS_ASSUMPTIONS.json',dict(rho_b_kg_m3=cfg['rho_b'],rho_f_kg_m3=cfg['rho_f'],gravity_m_s2=[0.,-cfg['gravity']],
        source='Default physical constants from uploaded CARC_H050_STD2_500EA_KREF1MD_ONEWELL_UNIFORM48_DIRECT/base/fbm500_biot_core.py; latest NPZ does not store rho_b; overridable at submit',
        no_initial_stress_field_invented=True,body_force_computed_from_initial_state=False,
        prescribed_pressure_Pa=1e7,pressure_nodes=12,mechanics='bottom Ux=Uy=0, left Ux=0, top/right zero total traction, left free component zero shear',
        pressure_natural_boundary='zero fractional Darcy flux; not zero pressure gradient in the gravity direction',
        quasistatic_displacement_is_algebraic=True,initial_displacement_fit_is_explicit=True))
    save_json(root/'NORMALIZATION.json',dict(P=float(scales[0]),U1=float(scales[1]),U2=float(scales[2]),Q=float(q_scale),T_s=float(times[k-1]),
              Lx_m=500.,Ly_m=500.,operation='divide only; NO offset subtraction',source='training prefix only'))
    return rec

def scoring_truth(cfg):
    root=Path(cfg['outdir'])
    if not (root/'FROZEN_MODEL.json').is_file() or not (root/'FROZEN_PREDICTION.npz').is_file(): raise RuntimeError('Future truth locked until model/prediction freeze')
    audit=load_json(root/'FORECAST_SOLVER_AUDIT.json')
    if not audit.get('full_horizon_time_convergence_passed'): raise RuntimeError('Future truth locked until prediction-only refinement passes')
    k=load_npz(root/'prepared/KNOWN_INPUTS.npz')
    with np.load(cfg['data'],allow_pickle=False) as z:
        a=np.stack([np.asarray(z[f+'_EA'][0],float) for f in FIELDS])
    if a.shape!=(3,49,10201) or not np.isfinite(a).all(): raise ValueError('Invalid full truth')
    return a/k['scales'][:,None,None]
