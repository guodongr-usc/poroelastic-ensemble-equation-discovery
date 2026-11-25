# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter



for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

FIELD_ORDER = ("Ux", "Uy", "P")


def _solve_equilibrated(matrix, rhs):
    
    matrix = matrix.tocsr()
    norms = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Invalid matrix row norm")
    row_multiplier = 1.0 / norms
    scaled = matrix.multiply(row_multiplier[:, None]).tocsc()
    factor = splu(scaled)
    b = row_multiplier[:, None] * rhs[:, None]
    value = factor.solve(np.asfortranarray(b))
    proposal = value + factor.solve(np.asfortranarray(b - scaled @ value))
    improved = np.linalg.norm(b - scaled @ proposal, axis=0) < np.linalg.norm(b - scaled @ value, axis=0)
    value[:, improved] = proposal[:, improved]
    residual = np.linalg.norm(scaled @ value - b, axis=0) / np.maximum(np.linalg.norm(b, axis=0), 1e-300)
    if not np.isfinite(value).all() or residual.max() > 1e-7:
        raise np.linalg.LinAlgError(f"Forward residual gate: {residual.max()}")
    return value[:, 0], float(residual.max())


def run_forward(package_root: Path, output_path: Path) -> dict:
    
    started = perf_counter()
    root = Path(package_root).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Prediction already exists: {output_path}; choose a new output path")
    model = root / "models" / "kernel_g0p1"
    manifest = json.loads((model / "manifest.json").read_text())
    if manifest["gradient_amplitude"] != 0.1:
        raise ValueError("This release requires the frozen g=0.1 model")

    with np.load(root / "inputs" / "kernel_inputs.npz", allow_pickle=False) as archive:
        inputs = {key: archive[key].copy() for key in archive.files}
    with np.load(model / "kernel.npz", allow_pickle=False) as archive:
        scales = np.asarray(archive["state_scales"], float)
        rows = np.asarray(archive["row_scales"], float)
        offset = np.asarray(archive["offset"], float)
    A, M = (sp.load_npz(model / name).tocsr() for name in ("A_total.npz", "M_total.npz"))

    xy = np.asarray(inputs["node_xy_m"], float)
    times = np.asarray(inputs["solver_times_s"], float)
    output_ids = np.asarray(inputs["output_solver_step_indices"], int)
    initial = np.asarray(inputs["initial_si"], float)
    free = np.asarray(inputs["free_dof_ids"], int)
    fixed = np.asarray(inputs["fixed_dof_ids"], int)
    known = np.asarray(inputs["fixed_state_normalized"], float)
    body = np.asarray(inputs["body_rhs_normalized"], float)
    source = np.asarray(inputs["source_rhs_normalized"], float)
    profile = np.asarray(inputs["source_temporal_profile"], float)
    n, nt = len(xy), len(times)
    if A.shape != M.shape or A.shape != (3 * n, 3 * n):
        raise ValueError("Frozen matrix shape does not match native geometry")
    if scales.shape != (3,) or np.any(scales <= 0) or rows.shape != (3 * n,) or np.any(rows <= 0):
        raise ValueError("Invalid normalization")
    if initial.shape != (3, n) or not np.isfinite(initial).all():
        raise ValueError("Invalid initial fields")
    if not np.array_equal(inputs["field_order"], FIELD_ORDER):
        raise ValueError("Unexpected input field order")
    if not np.array_equal(np.sort(np.r_[free, fixed]), np.arange(3 * n)):
        raise ValueError("Fixed/free DOFs must form a disjoint full partition")
    if times[0] != 0 or not np.all(np.diff(times) > 0) or profile.shape != (nt,):
        raise ValueError("Invalid internal schedule")
    if output_ids[0] != 0 or output_ids[-1] != nt - 1 or not np.all(np.diff(output_ids) > 0):
        raise ValueError("Invalid output selection")
    if not np.array_equal(times[output_ids], inputs["times_s"]):
        raise ValueError("Output times do not match selected solver times")
    for vector in (offset, known, body, source):
        if vector.shape != (3 * n,) or not np.isfinite(vector).all():
            raise ValueError("Invalid frozen vector")
    if M[:2 * n].nnz:
        raise ValueError("The frozen model must have quasistatic mechanics")
    if not np.isfinite(A.data).all() or not np.isfinite(M.data).all():
        raise ValueError("Nonfinite frozen matrix")

    state = np.empty((nt, 3 * n), dtype=np.float64)
    state[0] = (initial / scales[:, None]).reshape(-1)
    residuals = []
    for i in range(1, nt):
        dt = times[i] - times[i - 1]
        matrix = (A + M / dt).tocsr()
        rhs = body + profile[i] * source - offset + M @ state[i - 1] / dt
        current = known.copy()
        reduced = rhs[free] - matrix[free][:, fixed] @ current[fixed]
        current[free], residual = _solve_equilibrated(matrix[free][:, free], reduced)
        if not np.isfinite(current).all() or np.max(abs(current)) > 1e12:
            raise FloatingPointError(f"Unbounded normalized state at step {i}; no clipping applied")
        state[i] = current
        residuals.append(residual)
    internal = (state * np.repeat(scales, n)[None, :]).reshape(nt, 3, n)
    internal[0] = initial
    prediction = internal[output_ids]
    if not np.array_equal(prediction[0], initial):
        raise AssertionError("Initial fields were changed")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, prediction_si=prediction, times_s=times[output_ids],
                        field_order=np.asarray(FIELD_ORDER), node_xy_m=xy,
                        internal_prediction_si=internal, internal_times_s=times)
    return dict(method="frozen_kernel_closure", gradient_amplitude=0.1,
                field_order=list(FIELD_ORDER), units=["m", "m", "Pa"],
                internal_state_count=nt, output_state_count=len(output_ids),
                maximum_scaled_linear_residual=max(residuals, default=0.0),
                elapsed_seconds=perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description='Independent forward solve of the frozen CLEAN443 kernel model (g=0.1).\n\nOnly the saved total operators, known forcing/boundaries, and initial fields\nare read. No Solution fit, equation discovery, optimization, or later truth is\nperformed. The Biot skeleton is already included in A_total and M_total.\n')
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2]/"assets/kernel")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output if args.output is not None else Path.cwd() / "outputs" / "kernel_forward.npz"
    print(json.dumps(run_forward(args.root, output), indent=2))


if __name__ == "__main__":
    main()
