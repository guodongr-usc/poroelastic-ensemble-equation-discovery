# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


N_SIDE = 10


NCASE = N_SIDE * N_SIDE


MAX_WELL_NODES = 6


@dataclass(frozen=True)
class ForcingDesign:
    q_matrix: np.ndarray
    unit_directions: np.ndarray
    source_node_ids: np.ndarray
    source_node_xy_m: np.ndarray
    source_center_xy_m: np.ndarray
    source_x_indices: np.ndarray
    source_y_band_ids: np.ndarray
    q_total_m2_s: np.ndarray
    q_per_node_m2_s: np.ndarray
    gram: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float
    max_abs_offdiag_gram: float
    max_support_overlap_nodes: int


def _select_distinct_interior_x(nx: int, count: int = N_SIDE) -> np.ndarray:
    interior = np.arange(1, nx, dtype=np.int64)
    if interior.size < count:
        raise ValueError(f"Need at least {count} interior x nodes; nx={nx} gives {interior.size}")


    bands = [np.asarray(b, dtype=np.int64) for b in np.array_split(interior, count)]
    out = np.asarray([band[(len(band) - 1) // 2] for band in bands], dtype=np.int64)
    if np.unique(out).size != count:
        raise RuntimeError(f"Could not select {count} distinct interior x indices from nx={nx}: {out}")
    return out


def _select_disjoint_y_segments(ny: int, count: int = N_SIDE, max_nodes: int = MAX_WELL_NODES) -> list[np.ndarray]:
    interior = np.arange(1, ny, dtype=np.int64)
    if interior.size < count:
        raise ValueError(f"Need at least {count} interior y nodes; ny={ny} gives {interior.size}")
    bands = [np.asarray(b, dtype=np.int64) for b in np.array_split(interior, count)]
    nodes_per_well = min(max_nodes, min(len(b) for b in bands))
    if nodes_per_well < 1:
        raise RuntimeError("No interior y nodes available for a forcing support")
    selected: list[np.ndarray] = []
    for band in bands:
        start = (len(band) - nodes_per_well) // 2
        selected.append(band[start : start + nodes_per_well].copy())
    flat = np.concatenate(selected)
    if np.unique(flat).size != flat.size:
        raise RuntimeError("The selected y supports overlap")
    return selected


def build_forcing_design(geom: Any, q_total: float) -> ForcingDesign:
    
    if not np.isfinite(q_total) or q_total == 0.0:
        raise ValueError(f"q_total must be finite and nonzero, got {q_total}")

    x_indices = _select_distinct_interior_x(int(geom.nx), N_SIDE)
    y_segments = _select_disjoint_y_segments(int(geom.ny), N_SIDE, MAX_WELL_NODES)
    nodes_per_well = len(y_segments[0])
    if any(len(s) != nodes_per_well for s in y_segments):
        raise RuntimeError("All source supports must have the same node count")

    entries: list[tuple[float, int, int, np.ndarray]] = []
    cx = float(geom.lx) / 2.0
    cy = float(geom.ly) / 2.0
    for ix in x_indices:
        for iy_band, ys in enumerate(y_segments):
            nodes = ys * int(geom.nxnode) + int(ix)
            center = np.mean(np.asarray(geom.node_xy)[nodes], axis=0)
            distance2 = float((center[0] - cx) ** 2 + (center[1] - cy) ** 2)
            entries.append((distance2, int(ix), int(iy_band), nodes.astype(np.int64)))


    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(entries) != NCASE:
        raise RuntimeError(f"Expected {NCASE} forcing entries, got {len(entries)}")

    source_node_ids = np.vstack([item[3] for item in entries]).astype(np.int64)
    if np.unique(source_node_ids).size != source_node_ids.size:
        raise RuntimeError("Forcing supports are not globally disjoint")

    q_matrix = np.zeros((int(geom.np), NCASE), dtype=np.float64)
    q_per_node = float(q_total) / float(nodes_per_well)
    for j, nodes in enumerate(source_node_ids):
        q_matrix[nodes, j] = q_per_node

    norms = np.linalg.norm(q_matrix, axis=0)
    if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
        raise RuntimeError(f"Invalid forcing norms: {norms}")
    unit = q_matrix / norms[None, :]
    gram = unit.T @ unit
    eigvals = np.linalg.eigvalsh(gram)
    singular_values = np.sqrt(np.clip(eigvals, 0.0, None))[::-1]
    tol = max(gram.shape) * np.finfo(np.float64).eps * max(float(singular_values[0]), 1.0)
    rank = int(np.count_nonzero(singular_values > tol))
    condition_number = float(singular_values[0] / singular_values[-1])
    offdiag = gram - np.diag(np.diag(gram))
    max_abs_offdiag = float(np.max(np.abs(offdiag)))

    support_mask = (q_matrix != 0.0).astype(np.int16)
    overlaps = support_mask.T @ support_mask
    max_support_overlap = int(np.max(overlaps - np.diag(np.diag(overlaps))))

    if rank != NCASE:
        raise RuntimeError(f"Forcing design is rank {rank}, not {NCASE}")
    if not np.allclose(gram, np.eye(NCASE), rtol=0.0, atol=5.0e-14):
        raise RuntimeError(
            f"Normalized forcing Gram is not I_100: max off-diagonal={max_abs_offdiag:.3e}, "
            f"diag range=({np.diag(gram).min():.16g},{np.diag(gram).max():.16g})"
        )
    if max_support_overlap != 0:
        raise RuntimeError(f"Forcing supports overlap by as many as {max_support_overlap} nodes")

    source_xy = np.asarray(geom.node_xy)[source_node_ids]
    centers = np.mean(source_xy, axis=1)
    source_x_indices = np.asarray([item[1] for item in entries], dtype=np.int64)
    source_y_band_ids = np.asarray([item[2] for item in entries], dtype=np.int64)
    q_totals = np.full(NCASE, float(q_total), dtype=np.float64)
    q_per_nodes = np.full(NCASE, q_per_node, dtype=np.float64)

    return ForcingDesign(
        q_matrix=q_matrix,
        unit_directions=unit,
        source_node_ids=source_node_ids,
        source_node_xy_m=source_xy,
        source_center_xy_m=centers,
        source_x_indices=source_x_indices,
        source_y_band_ids=source_y_band_ids,
        q_total_m2_s=q_totals,
        q_per_node_m2_s=q_per_nodes,
        gram=gram,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition_number,
        max_abs_offdiag_gram=max_abs_offdiag,
        max_support_overlap_nodes=max_support_overlap,
    )


DAY_S = 86400.0


END_S = 730.0 * DAY_S


OUTPUT_INTERVALS = 48


def build_time_schedule():
    output = np.linspace(0.0, END_S, OUTPUT_INTERVALS + 1)
    original = np.expm1(np.linspace(0.0, np.log1p(6.3e7), 23))
    solver = np.unique(np.r_[original, output])
    indices = np.searchsorted(solver, output)
    np.testing.assert_array_equal(solver[indices], output)
    assert output[0] == 0 and output[-1] == END_S
    assert np.all(np.diff(solver) > 0)
    assert np.max(np.diff(solver)) <= END_S / OUTPUT_INTERVALS + 1e-7
    return output, solver, indices.astype(np.int64)
