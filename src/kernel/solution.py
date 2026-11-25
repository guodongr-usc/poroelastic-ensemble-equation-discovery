# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# Method reference for field reconstruction in a solution RKHS:
# Jalalian et al. (2025), sec. 2.1. https://arxiv.org/abs/2503.01036v2
# This study uses a finite cardinal basis on the spatial and log-time grids.

from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
from .operators import HEADS
from .metrics import absolute_metrics

def _axis_laplacian(count: int) -> sparse.csr_matrix:
    
    if count < 2:
        raise ValueError("Each spatial axis requires at least two nodes")
    incidence = sparse.diags((-np.ones(count - 1), np.ones(count - 1)),
                             (0, 1), shape=(count - 1, count), format="csr")
    return (incidence.T @ incidence).tocsr()


def time_second_difference(times_s: np.ndarray) -> tuple[np.ndarray, sparse.csr_matrix]:
    
    times = np.asarray(times_s, dtype=float)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("Supply at least two finite time coordinates")
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Times must be nonnegative and strictly increasing")
    log_times = np.log1p(times)
    tau = (log_times - log_times[0]) / (log_times[-1] - log_times[0])
    if len(times) == 2:
        return tau, sparse.csr_matrix((0, 2), dtype=float)
    intervals = np.diff(tau)
    left, right = intervals[:-1], intervals[1:]
    prefactor = 2.0 * float(intervals.mean())**2 / (left + right)
    vals = np.column_stack((prefactor / left,
                            -prefactor * (1.0 / left + 1.0 / right),
                            prefactor / right))
    rows = np.repeat(np.arange(len(times) - 2), 3)
    cols = (np.arange(len(times) - 2)[:, None] + np.arange(3)).ravel()
    difference = sparse.coo_matrix((vals.ravel(), (rows, cols)),
                                   shape=(len(times) - 2, len(times))).tocsr()
    return tau, difference


class CardinalPrecision:
    

    def __init__(self, times_s, x_m, y_m, *, head_count=3,
                 space_reg=1.0, time_reg=1.0):
        self.times_s = np.asarray(times_s, dtype=float).copy()
        self.x_m = np.asarray(x_m, dtype=float).copy()
        self.y_m = np.asarray(y_m, dtype=float).copy()
        for axis in (self.x_m, self.y_m):
            if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
                raise ValueError("Spatial axes must be finite and increasing")
        if not isinstance(head_count, (int, np.integer)) or head_count < 1:
            raise ValueError("head_count must be a positive integer")
        if not np.isfinite([space_reg, time_reg]).all() or min(space_reg, time_reg) < 0:
            raise ValueError("Precision regularization weights must be nonnegative")
        self.space_reg, self.time_reg = float(space_reg), float(time_reg)
        self.tau, self.time_difference = time_second_difference(self.times_s)
        self.time_laplacian = (self.time_difference.T @ self.time_difference).tocsr()
        nx, ny = len(self.x_m), len(self.y_m)
        self.space_laplacian = (
            sparse.kron(sparse.eye(ny, format="csr"), _axis_laplacian(nx), format="csr")
            + sparse.kron(_axis_laplacian(ny), sparse.eye(nx, format="csr"), format="csr")
        ).tocsr()
        self.shape = (len(self.times_s), int(head_count), nx * ny)
        self.size = int(np.prod(self.shape))
        self.diagonal_full = np.broadcast_to(
            1.0 + self.space_reg * self.space_laplacian.diagonal()[None, None, :]
            + self.time_reg * self.time_laplacian.diagonal()[:, None, None],
            self.shape).copy().ravel()
        self.operator = LinearOperator((self.size, self.size), matvec=self.matvec,
                                       rmatvec=self.matvec, dtype=np.float64)

    def matvec(self, vector):
        values = np.asarray(vector, dtype=float).reshape(self.shape)
        result = values.copy()
        if self.space_reg:
            spatial = (self.space_laplacian @ values.reshape(-1, self.shape[-1]).T).T
            result += self.space_reg * spatial.reshape(self.shape)
        if self.time_reg:
            temporal = self.time_laplacian @ values.reshape(self.shape[0], -1)
            result += self.time_reg * temporal.reshape(self.shape)
        return result.ravel()

    def quadratic(self, vector):
        
        values = np.asarray(vector, dtype=float).reshape(self.shape)
        yx = values.reshape(self.shape[0], self.shape[1], len(self.y_m), len(self.x_m))
        result = float(np.sum(values * values))
        if self.space_reg:
            result += self.space_reg * (float(np.sum(np.diff(yx, axis=2)**2))
                                        + float(np.sum(np.diff(yx, axis=3)**2)))
        if self.time_reg:
            d = self.time_difference @ values.reshape(self.shape[0], -1)
            result += self.time_reg * float(np.sum(d*d))
        return result


def fit_internal_solution(observed_physical, times_internal, output_indices,
                          x_m, y_m, fixed_positive_mask,
                          known_positive_physical, scales, eta=1e-8, *,
                          space_reg=1.0, time_reg=1.0):
    
    times = np.asarray(times_internal, dtype=float)
    ids = np.asarray(output_indices)
    observed = np.asarray(observed_physical, dtype=float)
    fixed_positive = np.asarray(fixed_positive_mask)
    known_positive = np.asarray(known_positive_physical, dtype=float)
    scale = np.asarray(scales, dtype=float)
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive for latent internal states")
    q = CardinalPrecision(times, x_m, y_m, head_count=3,
                         space_reg=space_reg, time_reg=time_reg)
    nt, nh, nn = q.shape
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("output_indices must be a one-dimensional integer array")
    ids = ids.astype(int, copy=True)
    if len(ids) < 2 or ids[0] != 0 or ids[-1] != nt-1 or np.any(np.diff(ids) <= 0):
        raise ValueError("Observed indices must be increasing and include both time endpoints")
    if observed.shape != (len(ids), nh, nn) or not np.isfinite(observed).all():
        raise ValueError("Observed fields must be finite and have shape (nobs,3,nnode)")
    if fixed_positive.dtype != bool or fixed_positive.shape != (nh, nn):
        raise ValueError("Positive-time fixed mask must be boolean, shape (3,nnode)")
    if known_positive.shape != (nh, nn) or not np.isfinite(known_positive[fixed_positive]).all():
        raise ValueError("Known positive-time constrained values must be finite")
    if scale.shape != (nh,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("scales must contain three finite positive physical unit scales")

    observed_normalized = observed / scale[None, :, None]
    fixed_mask = np.broadcast_to(fixed_positive[None, :, :], q.shape).copy()
    fixed_mask[0] = True
    fixed = np.zeros(q.shape, dtype=float)
    fixed[1:, fixed_positive] = (known_positive / scale[:, None])[fixed_positive]
    fixed[0] = observed_normalized[0]
    fixed_flat = fixed.ravel()
    free = np.flatnonzero(~fixed_mask.ravel())
    if not len(free):
        raise ValueError("At least one unconstrained state entry is required")

    observed_time_mask = np.zeros(nt, dtype=bool)
    observed_time_mask[ids] = True
    observation_mask = np.broadcast_to(observed_time_mask[:, None, None], q.shape)
    observation_diagonal = observation_mask.ravel()[free].astype(float)


    observation_rhs = np.zeros(q.shape, dtype=float)
    observation_rhs[ids] = observed_normalized
    rhs = observation_rhs.ravel()[free] - eta*q.matvec(fixed_flat)[free]

    def action(v):
        full = np.zeros(q.size, dtype=float)
        full[free] = v
        return observation_diagonal*np.asarray(v) + eta*q.matvec(full)[free]




    spatial_degree = q.space_laplacian.diagonal()
    temporal = eta*(np.eye(nt) + q.time_reg*q.time_laplacian.toarray())
    temporal += np.diag(observed_time_mask.astype(float))
    groups = []
    for head in range(nh):
        patterns, labels = np.unique((~fixed_mask[:, head, :]).T,
                                     axis=0, return_inverse=True)
        for label, pattern in enumerate(patterns):
            tids = np.flatnonzero(pattern)
            if not len(tids):
                continue
            for degree in np.unique(spatial_degree[labels == label]):
                nids = np.flatnonzero((labels == label) & (spatial_degree == degree))
                block = temporal[np.ix_(tids, tids)] + eta*q.space_reg*degree*np.eye(len(tids))
                groups.append((head, tids, nids, cho_factor(block, check_finite=True)))

    def inverse_temporal(v):
        full = np.zeros(q.size, dtype=float)
        full[free] = v
        full = full.reshape(q.shape)
        result = np.zeros(q.shape, dtype=float)
        for head, tids, nids, factor in groups:
            result[:, head, :][np.ix_(tids, nids)] = cho_solve(
                factor, full[:, head, :][np.ix_(tids, nids)], check_finite=False)
        return result.ravel()[free]




    diagonal_scale = np.sqrt(observation_diagonal + eta*q.diagonal_full[free])
    shape = (len(free), len(free))
    normal = LinearOperator(shape, dtype=float,
        matvec=lambda z: action(np.asarray(z)/diagonal_scale)/diagonal_scale,
        rmatvec=lambda z: action(np.asarray(z)/diagonal_scale)/diagonal_scale)
    preconditioner = LinearOperator(shape, dtype=float,
        matvec=lambda z: diagonal_scale*inverse_temporal(diagonal_scale*np.asarray(z)),
        rmatvec=lambda z: diagonal_scale*inverse_temporal(diagonal_scale*np.asarray(z)))
    scaled_rhs = rhs/diagonal_scale
    iterations = [0]
    def callback(_):
        iterations[0] += 1
    if np.any(scaled_rhs):
        fitted_scaled, info = cg(normal, scaled_rhs, M=preconditioner,
            rtol=1e-12, atol=0.0, maxiter=1000, callback=callback)
    else:
        fitted_scaled, info = np.zeros_like(scaled_rhs), 0
    fitted = fitted_scaled/diagonal_scale
    scaled_residual = normal@fitted_scaled-scaled_rhs
    residual = action(fitted)-rhs
    relative_scaled_residual = float(np.linalg.norm(scaled_residual)/max(np.linalg.norm(scaled_rhs), 1e-300))
    relative_residual = float(np.linalg.norm(residual)/max(np.linalg.norm(rhs), 1e-300))
    if info or not np.isfinite(fitted).all() or relative_scaled_residual > 5e-12:
        raise ArithmeticError(f"Masked Solution PCG failed: info={info}, scaled residual={relative_scaled_residual}")
    normalized = fixed_flat.copy()
    normalized[free] = fitted
    normalized = normalized.reshape(q.shape)
    physical = normalized*scale[None, :, None]


    physical[0] = observed[0]
    physical[1:, fixed_positive] = known_positive[fixed_positive]
    fields = {head: physical[:, h].copy() for h, head in enumerate(HEADS)}

    fitted_observed = physical[ids]
    metrics_all, metrics_free = {}, {}
    for h, head in enumerate(HEADS):
        metrics_all[head] = absolute_metrics(observed[:, h], fitted_observed[:, h], head)
        take = ~fixed_mask[ids, h]
        metrics_free[head] = absolute_metrics(observed[:, h][take], fitted_observed[:, h][take], head)
    error = normalized[ids]-observed_normalized
    data_sse = float(np.sum(error*error))
    norm2 = q.quadratic(normalized)
    hidden_ids = np.flatnonzero(~observed_time_mask)
    audit = {
        "kernel": "k_U(z,z') = phi(z)^T Q^{-1} phi(z')",
        "basis": "Original spatial nodal Q1 times cardinal piecewise-linear hats in normalized log1p(internal time)",
        "precision": "Q = I + space_reg*(I_time tensor I_head tensor Lxy) + time_reg*(Dtau.T Dtau tensor I_head tensor I_node)",
        "space_reg": q.space_reg, "time_reg": q.time_reg,
        "objective": "0.5*||H v-y||^2 + 0.5*eta*v.T Q v, with exact affine IC/BC constraints",
        "eta": float(eta), "shape": list(q.shape), "heads": list(HEADS),
        "observation_time_indices": ids.tolist(), "observed_time_count": len(ids),
        "hidden_time_indices": hidden_ids.tolist(), "hidden_time_count": len(hidden_ids),
        "hidden_state_entry_count": int(len(hidden_ids)*nh*nn),
        "hidden_free_state_entry_count": int(np.sum(~fixed_mask[hidden_ids])),
        "observed_scalar_value_count": int(observed.size),
        "observed_free_value_count": int(np.sum(observation_diagonal)),
        "free_state_entry_count": len(free), "fixed_state_entry_count": int(fixed_mask.sum()),
        "hidden_state_truth_labels_supplied": False,
        "hidden_state_observation_loss_weight": 0.0,
        "observed_value_fit_all": metrics_all, "observed_value_fit_unconstrained": metrics_free,
        "data_normalized_SSE": data_sse, "full_RKHS_norm2": norm2,
        "objective_value": 0.5*(data_sse+eta*norm2),
        "linear_solve": {"method": "masked normal equations, diagonal congruence, temporal-block PCG",
            "info": int(info), "iterations": iterations[0], "rtol": 1e-12,
            "relative_residual": relative_residual,
            "relative_congruence_scaled_residual": relative_scaled_residual,
            "preconditioner_block_count": len(groups),
            "spatial_off_diagonals_retained_in_actual_operator": True},
        "initial_three_fields_exact": bool(np.array_equal(physical[0], observed[0])),
        "positive_Dirichlet_values_exact": bool(np.array_equal(
            physical[1:, fixed_positive], np.broadcast_to(known_positive[fixed_positive],
                                                         physical[1:, fixed_positive].shape))),
        "state_order": "time,head,node", "absolute_total_fields_only": True,
        "reference_or_response_subtraction": False,
        "PDE_solver_called": False, "file_or_dataset_access": False,
        "internal_states_are_inferred_not_measured": True,
        "derivatives": "Original Q1 weak functionals and internal-grid backward-Euler differences are downstream observations; no analytic strong-form jets claimed",
    }
    return fields, audit
