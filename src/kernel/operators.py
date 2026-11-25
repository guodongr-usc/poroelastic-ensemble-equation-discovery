# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import numpy as np
import scipy.sparse as sp
from .fem import FEM, LinearFactor

HEADS=('U1','U2','P')

UNITS=np.array([1e9,1e9,1.,1e-9,1e-11,1e-11])


KEYS=('mean_lambda_elem_pa','mean_G_elem_pa','mean_alpha_elem','mean_S_elem_pa_inv',
      'mean_mobility_elem_m2_pa_s')


def make_engine(meta,times,stride=1):
    if any(k.startswith('mean_') for k in meta):
        raise ValueError('No mean material maps may enter this interface')
    constructor={k:np.asarray(v).copy() for k,v in meta.items()}
    constructor['times_s']=np.asarray(times).copy()
    constructor['forcing_temporal_profile']=np.r_[0.,np.ones(len(times)-1)]
    shape=(len(meta['y_m'])-1,len(meta['x_m'])-1)
    for k,unit in zip(KEYS,UNITS[:5]):constructor[k]=np.full(shape,unit)
    engine=FEM(constructor,stride=stride)


    engine.base[:]=np.nan
    for k in KEYS:engine.metadata.pop(k)
    return engine


def _scale_vector(value, nn, name):
    value = np.asarray(value, dtype=float)
    if value.shape == (3,):
        value = np.repeat(value, nn)
    if value.shape != (3 * nn,) or np.any(value <= 0) or not np.isfinite(value).all():
        raise ValueError(f"{name} requires three or 3*nn positive finite scales")
    return value.copy()


def derivative_operators(x_m, y_m):
    
    x, y = np.asarray(x_m, float), np.asarray(y_m, float)

    def one_dimensional(axis):
        if axis.ndim != 1 or len(axis) < 4 or not np.isfinite(axis).all():
            raise ValueError("Each axis requires at least four finite coordinates")
        steps = np.diff(axis)
        if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-12, atol=1e-12):
            raise ValueError("Derivative operators require a uniform increasing grid")
        n, h = len(axis), steps[0]
        d1 = sp.lil_matrix((n, n), dtype=float)
        d2 = sp.lil_matrix((n, n), dtype=float)
        for i in range(1, n - 1):
            d1[i, i - 1:i + 2] = np.asarray([-0.5, 0.0, 0.5]) / h
            d2[i, i - 1:i + 2] = np.asarray([1.0, -2.0, 1.0]) / h**2
        d1[0, :3] = np.asarray([-1.5, 2.0, -0.5]) / h
        d1[-1, -3:] = np.asarray([0.5, -2.0, 1.5]) / h
        d2[0, :4] = np.asarray([2.0, -5.0, 4.0, -1.0]) / h**2
        d2[-1, -4:] = np.asarray([-1.0, 4.0, -5.0, 2.0]) / h**2
        return d1.tocsr(), d2.tocsr()

    dx, dxx = one_dimensional(x)
    dy, dyy = one_dimensional(y)
    ix, iy = sp.eye(len(x), format="csr"), sp.eye(len(y), format="csr")
    result = {
        "I": sp.eye(len(x) * len(y), format="csr"),
        "Dx": sp.kron(iy, dx, format="csr"),
        "Dy": sp.kron(dy, ix, format="csr"),
        "Dxx": sp.kron(iy, dxx, format="csr"),
        "Dyy": sp.kron(dyy, ix, format="csr"),
        "Dxy": sp.kron(dy, dx, format="csr"),
    }
    result["Lap"] = result["Dxx"] + result["Dyy"]
    return result


@dataclass
class FrozenSystem:
    metadata: dict
    engine: FEM
    coefficients: np.ndarray
    nn: int
    state_scales: np.ndarray
    state_scale_vector: np.ndarray
    row_scales: np.ndarray
    A_physical: sp.csr_matrix
    M_physical: sp.csr_matrix
    A_static: sp.csr_matrix
    M: sp.csr_matrix
    f_static: np.ndarray
    f_source: np.ndarray
    times_s: np.ndarray
    profile: np.ndarray
    fixed: np.ndarray
    free: np.ndarray
    fixed_u: np.ndarray
    free_u: np.ndarray
    known_positive: np.ndarray
    p0_normalized: np.ndarray
    case_id: int


    def rhs(self, time_index):
        
        return self.f_static + self.f_source * self.profile[int(time_index)]



def build_system(metadata: Mapping, state_scales, row_scales=None,
                 coefficients=None, case_id=1):
    
    state_scales = np.asarray(state_scales, dtype=float)
    if state_scales.shape != (3,) or np.any(state_scales <= 0) or not np.isfinite(state_scales).all():
        raise ValueError("state_scales must contain positive U1,U2,P scales")
    meta = {k: np.asarray(v).copy() for k, v in metadata.items()}
    engine = FEM(meta, stride=1)
    nn, nu = engine.nn, engine.nu
    c = engine.base.copy() if coefficients is None else np.asarray(coefficients, float).copy()
    op = engine.assemble(c)
    zero_up = sp.csr_matrix((nu, nn))
    zero_uu = sp.csr_matrix((nu, nu))
    zero_pu = sp.csr_matrix((nn, nu))
    physical_A_interleaved = sp.bmat([[op.K, -op.Q], [zero_pu, op.H]], format="csr")
    physical_M_interleaved = sp.bmat([[zero_uu, zero_up], [op.Q.T, op.C]], format="csr")
    order = np.r_[np.arange(0, nu, 2), np.arange(1, nu, 2), np.arange(nu, nu + nn)]
    A_physical = physical_A_interleaved[order][:, order].tocsr()
    M_physical = physical_M_interleaved[order][:, order].tocsr()
    scales = np.repeat(state_scales, nn)
    A_scaled = A_physical.multiply(scales[None, :]).tocsr()
    M_scaled = M_physical.multiply(scales[None, :]).tocsr()
    if row_scales is None:
        tref = float(np.median(np.diff(engine.times)))
        magnitudes = np.asarray((abs(A_scaled) + abs(M_scaled) / tref).sum(axis=1)).ravel()
        head_scales = np.sqrt(np.mean(magnitudes.reshape(3, nn)**2, axis=1))
        row_scales = _scale_vector(head_scales, nn, "row_scales")
    else:
        row_scales = _scale_vector(row_scales, nn, "row_scales")
    A = A_scaled.multiply((1.0 / row_scales)[:, None]).tocsr()
    M = M_scaled.multiply((1.0 / row_scales)[:, None]).tocsr()
    physical_static = np.r_[engine.body, op.gravity][order]
    physical_source = np.r_[np.zeros(nu), engine.sources((int(case_id),))[:, 0]][order]
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order))
    fixed_u = np.sort(inverse_order[engine.fixed_u])
    fixed = np.sort(inverse_order[engine.fixed])
    free_u = np.setdiff1d(np.arange(nu), fixed_u)
    free = np.setdiff1d(np.arange(3 * nn), fixed)
    known = np.zeros(3 * nn)
    known[2 * nn + engine.fixed_p] = engine.p_boundary / state_scales[2]
    return FrozenSystem(
        meta, engine, c, nn, state_scales.copy(), scales, row_scales,
        A_physical, M_physical, A, M,
        physical_static / row_scales, physical_source / row_scales,
        engine.times.copy(), engine.profile.copy(), fixed, free, fixed_u,
        free_u, known, engine.p0 / state_scales[2], int(case_id),
    )


def unit_matrices(engine,scales):
    c=np.broadcast_to(UNITS,(len(engine.conn),6)).copy();c[:,2]=.5
    op=engine.assemble(c);n=engine.nn;nu=engine.nu
    A=sp.bmat([[op.K,-op.Q],[sp.csr_matrix((n,nu)),op.H]],format='csr')
    M=sp.bmat([[sp.csr_matrix((nu,nu)),sp.csr_matrix((nu,n))],[op.Q.T,op.C]],format='csr')
    order=np.r_[np.arange(0,nu,2),np.arange(1,nu,2),nu+np.arange(n)]
    sc=np.repeat(scales,n)
    return [a[order][:,order].multiply(sc[None,:]).tocsr() for a in (A,M)]


def context(meta,times,scales,rows=None):
    if any(k.startswith('mean_') for k in meta):raise ValueError('No mean material maps')
    engine=make_engine(meta,times);n=engine.nn
    scales=np.asarray(scales,float);sc=np.repeat(scales,n)
    if rows is None:
        Au,Mu=unit_matrices(engine,scales)
        raw=np.asarray((abs(Au)+abs(Mu)/np.median(np.diff(times))).sum(axis=1)).ravel()
        rows=np.repeat(np.sqrt(np.mean(raw.reshape(3,n)**2,axis=1)),n)
    rows=np.asarray(rows,float)
    assert rows.shape==(3*n,) and np.all(rows>0)
    freeids=engine.free_ids
    free=np.concatenate([freeids[h]+j*n for j,h in enumerate(HEADS)])
    fixed=np.setdiff1d(np.arange(3*n),free)
    known=np.zeros(3*n);known[2*n+engine.fixed_p]=engine.p_boundary/scales[2]
    body=np.r_[engine.body[0::2],engine.body[1::2],np.zeros(n)]/rows
    source=np.r_[np.zeros(2*n),engine.sources((1,))[:,0]]/rows
    profile=np.r_[0.,np.ones(len(times)-1)]
    return dict(engine=engine,n=n,scales=scales,rows=rows,freeids=freeids,
        free=free,fixed=fixed,known=known,body=body,source=source,
        rhs=body[None,:]+profile[:,None]*source[None,:])


def solve_total(A,M,offset,meta,initial,scales,rows):
    
    times=np.asarray(meta['solver_times_s']);ctx=context(meta,times,scales,rows)
    n=ctx['n'];nt=len(times);sc=np.repeat(scales,n)
    A=sp.csr_matrix(A);M=sp.csr_matrix(M);offset=np.asarray(offset,float)
    assert A.shape==M.shape==(3*n,3*n) and offset.shape==(3*n,)
    assert not M[:2*n].nnz,'Quasistatic mechanics cannot contain time terms'
    assert np.isfinite(A.data).all() and np.isfinite(M.data).all() and np.isfinite(offset).all()
    state=np.full((nt,3*n),np.nan)
    state[0]=np.concatenate([np.asarray(initial[h])/scales[j] for j,h in enumerate(HEADS)])
    residuals=[];error=None;last=0
    for i in range(1,nt):
        try:
            dt=times[i]-times[i-1];matrix=(A+M/dt).tocsr()
            rhs=ctx['rhs'][i]-offset+M@state[i-1]/dt
            current=ctx['known'].copy()
            reduced=rhs[ctx['free']]-matrix[ctx['free']][:,ctx['fixed']]@current[ctx['fixed']]
            factor=LinearFactor(matrix[ctx['free']][:,ctx['free']],np.ones(len(ctx['free'])))
            solved,residual=factor.solve(reduced[:,None]);current[ctx['free']]=solved[:,0]
            if not np.isfinite(current).all() or np.max(abs(current))>1e12:
                raise FloatingPointError('Nonfinite or normalized state exceeds1e12; no clipping')
            state[i]=current;residuals.append(float(residual));last=i
        except Exception as exc:
            error=f'{type(exc).__name__}: {exc}';break
    fields=(state*sc[None,:]).reshape(nt,3,n)
    for j,h in enumerate(HEADS):fields[0,j]=initial[h]
    return fields,dict(complete=error is None,error=error,last_completed_internal_index=last,
        internal_state_count=nt,maximum_scaled_linear_residual=max(residuals,default=0.),
        initial_three_fields_exact=True,initial_mechanical_projection=False,
        fixed_Biot_matrix_added=False,all_matrices_from_frozen_closure=True,
        true_state_resets=False,future_truth_used=False,Solution_trajectory_used=False,
        known_body_and_well_retained=True,fluid_gravity_inside_learned_features=True,
        original_internal_schedule=True,no_clipping=True)
