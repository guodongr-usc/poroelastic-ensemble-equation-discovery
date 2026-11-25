# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# Method reference for a known operator plus an RKHS correction:
# Jalalian et al. (2025), sec. 2.1, eq. (8). https://arxiv.org/abs/2503.01036v2
# This study fixes the Biot operator and fits ten closure coefficients at each
# free node using weighted ridge regression.

from __future__ import annotations
import numpy as np
from .library import HEADS, rates

def raw_batch(library,state,derivative,ids):
    out=np.empty((len(state),len(ids),len(library)))
    for j,f in enumerate(library):
        value=np.broadcast_to(f.offset[ids],(len(state),len(ids))).copy()
        if f.R is not None:value+=(f.R[ids]@(derivative if f.temporal else state).T).T
        out[:,:,j]=value
    return out


def solve_centered_delta(X, residual_target, penalties):
    
    X = np.asarray(X, dtype=float)
    r = np.asarray(residual_target, dtype=float)
    if X.ndim != 3 or r.shape != X.shape[:2] or len(X) < 1:
        raise ValueError('Expected X=(time,node,feature), residual_target=(time,node)')
    if not np.isfinite(X).all() or not np.isfinite(r).all():
        raise ValueError('Regression arrays must be finite')
    try:
        penalty = np.broadcast_to(np.asarray(penalties, float), X.shape[1:])
    except ValueError as exc:
        raise ValueError('Penalties must broadcast to (node,feature)') from exc
    if not np.isfinite(penalty).all() or np.any(penalty <= 0):
        raise ValueError('All penalties must be positive and finite')
    nt, nodes, p = X.shape
    if p == 0:
        return np.zeros((nodes, 0)), dict(KKT_normalized_max_error=0.,
                                         KKT_absolute_max_error=0.)
    root = np.sqrt(penalty)
    Z = X / root[None, :, :]
    H = np.einsum('tnj,tnk->njk', Z, Z) / nt + np.eye(p)[None, :, :]
    g = np.einsum('tnj,tn->nj', Z, r) / nt
    whitened = np.linalg.solve(H, g[..., None])[..., 0]
    delta = whitened / root

    residual = np.einsum('tnj,nj->tn', X, delta) - r
    data_gradient = np.einsum('tnj,tn->nj', X, residual) / nt
    penalty_gradient = penalty * delta
    kkt = data_gradient + penalty_gradient



    rhs = np.einsum('tnj,tn->nj', X, r) / nt
    scale = np.maximum(np.linalg.norm(rhs, axis=1)
                       + np.linalg.norm(penalty_gradient, axis=1), 1e-20)
    certificate = dict(
        KKT_normalized_max_error=float(np.max(np.linalg.norm(kkt, axis=1) / scale)),
        KKT_absolute_max_error=float(np.max(abs(kkt))))
    if not np.isfinite(delta).all() or certificate['KKT_normalized_max_error'] > 1e-6:
        raise ArithmeticError('Centered kernel ridge failed its KKT check: ' + str(certificate))
    return delta, certificate


def fit_centered(libraries, state, times, knownrhs, freeids, rowweights, ridge,
                 prior_coefficients, *, gradient_amplitude=.1,
                 batch_size=512):
    
    state = np.asarray(state, float)
    knownrhs = np.asarray(knownrhs, float)
    rowweights = np.asarray(rowweights, float)
    if state.ndim != 2 or state.shape[1] % 3:
        raise ValueError('state must have shape (time,3*node)')
    nt, d = state.shape
    n = d // 3
    if knownrhs.shape != state.shape or rowweights.shape != state.shape:
        raise ValueError('knownrhs and rowweights must have the state shape')
    if not all(np.isfinite(v).all() for v in (state, knownrhs, rowweights)):
        raise ValueError('Input arrays must be finite')
    if np.any(rowweights < 0):
        raise ValueError('Row weights must be nonnegative')
    for label, value in [('ridge', ridge), ('gradient_amplitude', gradient_amplitude)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(label + ' must be positive and finite')
    if int(batch_size) != batch_size or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if len(libraries) != 3 or len(prior_coefficients) != 3:
        raise ValueError('Exactly three heads and prior arrays are required')
    derivative = rates(state, times)
    allweights, allscales, allcoeff = [], [], []
    allprior, alldelta, allpenalties, audit = [], [], [], []
    for h, library in enumerate(libraries):
        p = len(library)
        prior = np.asarray(prior_coefficients[h], float)
        if prior.shape != (n, p) or not np.isfinite(prior).all():
            raise ValueError('Prior for ' + HEADS[h] + ' must be finite, shape (node,feature)')
        idsall = np.asarray(freeids[HEADS[h]], dtype=int)
        if idsall.ndim != 1 or len(np.unique(idsall)) != len(idsall) or np.any(idsall < 0) or np.any(idsall >= n):
            raise ValueError('Invalid or duplicate free node indices')
        rw = rowweights[:, h*n:(h+1)*n]
        scales = np.ones((n, p))
        global_square = np.zeros(p)
        for start in range(0, n, batch_size):
            ids = np.arange(start, min(n, start+batch_size))
            raw = raw_batch(library, state, derivative, ids)
            square = np.mean((raw * rw[:, ids, None])**2, axis=0)
            scales[ids] = np.sqrt(square)
            global_square += square.sum(axis=0) / n
        scales = np.maximum(scales, np.maximum(np.sqrt(global_square)*1e-4, 1e-14)[None, :])
        amplitude, penalties = np.ones(p), np.full(p, float(ridge))
        active = np.array([f.family == 'old_closure' for f in library])
        assert np.array_equal(np.flatnonzero(active), np.arange(10))
        amplitude[4:10] = gradient_amplitude
        scales /= amplitude[None, :]
        delta = np.zeros((n, p))
        coefficient = prior.copy()
        loss = regularizer = prior_loss = 0.
        kktmax = kktabsolute = 0.
        for start in range(0, len(idsall), batch_size):
            ids = idsall[start:start+batch_size]
            raw = raw_batch(library, state, derivative, ids)


            prior_value = np.einsum('tnj,nj->tn', raw, prior[ids])
            target = (knownrhs[:, h*n+ids] - prior_value) * rw[:, ids]
            X = raw * rw[:, ids, None] / scales[None, ids, :]
            di, certificate = solve_centered_delta(X[:, :, active], target, penalties[active])
            delta[np.ix_(ids, active)] = di
            coefficient[ids] += delta[ids] / scales[ids]
            residual = np.einsum('tnj,nj->tn', X[:, :, active], di) - target
            loss += .5 * float(np.sum(residual**2)) / nt
            prior_loss += .5 * float(np.sum(target**2)) / nt
            regularizer += .5 * float(np.sum(di**2 * penalties[active][None, :]))
            kktmax = max(kktmax, certificate['KKT_normalized_max_error'])
            kktabsolute = max(kktabsolute, certificate['KKT_absolute_max_error'])
        prior_weights = prior * scales


        allweights.append(coefficient * scales)
        allscales.append(scales)
        allcoeff.append(coefficient)
        allprior.append(prior_weights)
        alldelta.append(delta)
        allpenalties.append(penalties)
        audit.append(dict(
            head=HEADS[h], feature_count=p, trainable_feature_count=int(active.sum()),
            learned_node_count=len(idsall), residual_loss=loss/n,
            prior_residual_loss=prior_loss/n, RKHS_penalty=regularizer/n,
            KKT_normalized_max_error=kktmax, KKT_absolute_max_error=kktabsolute,
            feature_names=[f.name for f in library], penalties=penalties.tolist(),
            shift_diagnostics=dict(
                units='Per-feature coefficient norms, not field-prediction L2 errors',
                delta_scaled_l2_by_feature=np.linalg.norm(delta[idsall], axis=0).tolist(),
                prior_scaled_l2_by_feature=np.linalg.norm(prior_weights[idsall], axis=0).tolist(),
                delta_assembly_l2_by_feature=np.linalg.norm((coefficient-prior)[idsall], axis=0).tolist(),
                prior_assembly_l2_by_feature=np.linalg.norm(prior[idsall], axis=0).tolist())))
    return dict(
        weights=allweights, feature_scales=allscales, assembly_coefficients=allcoeff,
        prior_weights=allprior, delta_weights=alldelta, feature_penalties=allpenalties,
        audit=dict(
            heads=audit, objective=sum(v['residual_loss']+v['RKHS_penalty'] for v in audit),
            ridge=float(ridge), biot_multiplier=1.,
            temporal_multiplier=1., gradient_amplitude=float(gradient_amplitude),
            biot_amplitude=1.,
            optimizer='exact centered anisotropic kernel ridge; closed-form whitened delta solve',
            kernel='cardinal spatial sections times a degree-one normalized jet kernel with diagonal covariance',
            prior='supplied mean-Biot assembly coefficient center; caller audits its provenance',
            gauge=None, physical_Biot_coefficients_fixed=True,
            fixed_Biot_control=True, forward_calls_in_fitting=0,
            observation_times_only=True, spatial_coefficient_ties=False,
            stability_or_positivity_constraint_in_fit=False))
