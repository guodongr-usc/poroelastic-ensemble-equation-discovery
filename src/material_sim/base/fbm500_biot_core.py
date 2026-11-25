# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.
# Code attribution: GaussianRandomFieldCircEmbed, PrefbmCov, FBmGenerator and
# compute_c are adapted from Binhao Li's FlowSimulation, simulation/grf_generation.py.
# https://github.com/lastoor/FlowSimulation/blob/4afc232100a16e8c6e636a534da732c5d126f947/simulation/grf_generation.py
# The associated study is Li, de Barros & Sahimi (2026), https://doi.org/10.1029/2026GL124085.
# The fBm construction follows M. L. Stein (2002), "Fast and Exact Simulation
# of Fractional Brownian Surfaces," https://doi.org/10.1198/106186002466.
#
# Constitutive models: invert_kc uses the normalized Kozeny-Carman relation
# (P. C. Carman, 1937, "Fluid flow through granular beds," 1997 reprint,
# https://doi.org/10.1016/S0263-8762(97)80003-2). The Young's modulus in route1
# adapts Eq. 3 of Osorio, Sahimi, Barati & Ghanbarian (2025) to two dimensions
# with study-specific parameters and a prescribed modulus below the percolation
# threshold, https://doi.org/10.1029/2024JB030836.
# The coupled finite-element equations use the poroelastic theory of M. A. Biot
# (1941), "General Theory of Three-Dimensional Consolidation," https://doi.org/10.1063/1.1712886.

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.fft import dctn, fftn, next_fast_len


class GaussianRandomFieldCircEmbed:
    def __init__(self, mean: np.ndarray, cov, pts: Tuple[np.ndarray, ...], minpadding=0):
        self.mean = mean
        self.cov = cov
        self.pts = pts
        normedpts = [p - p[0] for p in pts]
        pads = [minpadding] * len(pts) if np.isscalar(minpadding) else minpadding
        dims = [self.circulant_minsize(cov, normedpts[i], pads[i]) for i in range(len(pts))]
        lam = self.circulant_eigvals(cov, normedpts, dims)
        n_total = np.prod(dims)
        neg = lam < 0
        lam[neg] = 0.0
        lam[~neg] = np.sqrt(lam[~neg] / n_total)
        self.data = (lam, tuple(dims))

    def sample(self) -> np.ndarray:
        lam, shape_emb = self.data
        xi = np.random.randn(*shape_emb)
        w = fftn(lam * xi)
        slices = tuple(slice(0, len(p)) for p in self.pts)
        wc = w[slices]
        return self.mean + wc.real + wc.imag

    @staticmethod
    def circulant_minsize(cov, pts: np.ndarray, minpadding: int) -> int:
        n = len(pts)
        base = n + minpadding - (1 if cov.is_even else 0)
        return 2 * next_fast_len(base)

    @staticmethod
    def circulant_eigvals(cov, pts: Iterable[np.ndarray], dims: Iterable[int]) -> np.ndarray:
        pts = [np.asarray(p) for p in pts]
        dims = tuple(dims)
        starts = [p[0] for p in pts]
        steps = [p[1] - p[0] for p in pts]
        cov_vec = np.vectorize(cov, signature="(n)->()")
        half_shape = tuple(d // 2 + 1 for d in dims)
        coords = [starts[d] + np.arange(half_shape[d]) * steps[d] for d in range(len(dims))]
        mesh = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1)
        cvals = dctn(cov_vec(mesh), type=1, norm=None)
        idx = [np.minimum(np.arange(d), d - np.arange(d)) for d in dims]
        return cvals[np.ix_(*idx)]


class PrefbmCov:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.is_even = True
        self.c0 = None
        self.c2 = None

    def __call__(self, x: np.ndarray) -> float:
        r = np.linalg.norm(x)
        a = self.alpha
        if 0 < a <= 1.5:
            self.c0 = 1 - a / 2
            self.c2 = a / 2
            if r == 0:
                return self.c0
            if r <= 1:
                return self.c0 - r**a + self.c2 * r**2
            return 0.0
        if 1.5 < a < 2:
            self.c0 = 1 - a / 6 - a**2 / 6
            self.c2 = a * (5 + 2 * a) / 18
            if r == 0:
                return self.c0
            if r <= 1:
                return self.c0 - r**a + self.c2 * r**2
            if r <= 2:
                return a * (2 - a) / 18 * (2 - r) ** 3 / r
            return 0.0
        raise ValueError("alpha must be in (0, 2)")


class FBmGenerator:
    def __init__(self, *, c: float, alpha: float, num_pt: int, dim: int = 2, minpadding: int = 0):
        self.c = c
        self.alpha = alpha
        self.num_pt = num_pt
        self.dim = dim
        self.pts = (np.arange(num_pt) / num_pt,) * dim
        self.minpadding = minpadding if alpha <= 1.5 else minpadding + num_pt
        mean = np.zeros(tuple(len(p) for p in self.pts))
        cov = PrefbmCov(alpha)
        self._prefbm_gen = GaussianRandomFieldCircEmbed(mean, cov, self.pts, minpadding=self.minpadding)
        self._sigma_trend = np.sqrt(2 * self._prefbm_gen.cov.c2)

    def sample(self) -> np.ndarray:
        prefbm = self._prefbm_gen.sample()
        coords = np.meshgrid(*self.pts, indexing="ij")
        slopes = np.random.normal(scale=self._sigma_trend, size=len(coords))
        trend = sum(coord * slope for coord, slope in zip(coords, slopes))
        z = prefbm + trend
        origin = tuple([0] * z.ndim)
        return np.sqrt(self.c) * self.num_pt ** (self.alpha / 2) * (z - z[origin])


def compute_c(length: int, hurst: float, std_target: float) -> float:
    coords = np.arange(length) - (length - 1) / 2
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    avg = np.mean((xx**2 + yy**2) ** hurst)
    return std_target**2 / (2 * avg)


K_REF_MD = 10.0


PHI_REF = 0.20


NU = 0.25


KS = 36.0e9


KF = 2.2e9


PC = 0.12


PX = 0.78


Z_COORD = 8.0


F_EXP = 3.97


E_SOLID = 40.0e9


MD_TO_M2 = 9.869233e-16


MU_F = 1.0e-3


RHO_F = 1000.0


RHO_B = 2242.609582


GRAVITY = np.array([0.0, -9.81])


SPECIFIC_WEIGHT = RHO_F * 9.81


OUTER_PRESSURE = 10.0e6


TOP_INITIAL_PRESSURE = 9.81e6


P_SCALE = 1.0e7


DEFAULT_Q_TOTAL = 6.0e-5


def invert_kc(k_md: np.ndarray) -> np.ndarray:
    log_target = (
        np.log(np.maximum(k_md, 1.0e-30) / K_REF_MD)
        + 3 * np.log(PHI_REF)
        - 2 * np.log1p(-PHI_REF)
    )
    phi = np.full_like(k_md, PHI_REF, dtype=float)
    for _ in range(30):
        residual = 3 * np.log(phi) - 2 * np.log1p(-phi) - log_target
        derivative = 3 / phi + 2 / (1 - phi)
        phi = np.clip(phi - residual / derivative, 1.0e-5, 0.95)
    return phi


def route1(phi: np.ndarray) -> Tuple[np.ndarray, ...]:
    psolid = 1 - phi
    critical = 4 / Z_COORD
    young = np.empty_like(psolid, dtype=np.float64)
    low = psolid <= PC
    middle = (psolid > PC) & (psolid <= PX)
    high = psolid > PX
    young[low] = 1.0e6
    young[middle] = (
        E_SOLID
        * (PX - critical)
        / (1 - critical)
        * ((psolid[middle] - PC) / (PX - PC)) ** F_EXP
    )
    young[high] = E_SOLID * (psolid[high] - critical) / (1 - critical)
    shear = young / (2 * (1 + NU))
    lame = young * NU / ((1 + NU) * (1 - 2 * NU))
    kdr = young / (3 * (1 - 2 * NU))
    alpha = 1 - kdr / KS
    storage = (alpha - phi) / KS + phi / KF
    return young, lame, shear, alpha, storage


def material_from_logk(logk500: np.ndarray, nx: int, ny: int) -> Dict[str, np.ndarray]:
    if 500 % nx or 500 % ny:
        raise ValueError("nx and ny must divide 500 exactly for block averaging")
    by = 500 // ny
    bx = 500 // nx
    logk_elem = logk500.reshape(ny, by, nx, bx).mean(axis=(1, 3))
    k_md = K_REF_MD * np.exp(logk_elem)
    phi = invert_kc(k_md)
    young, lame, shear, alpha, storage = route1(phi)
    return {
        "logk": logk_elem,
        "kmd": k_md,
        "phi": phi,
        "E": young,
        "lam": lame,
        "G": shear,
        "alpha_mech": alpha,
        "alpha_mass": alpha,
        "S": storage,
        "mobility": k_md * MD_TO_M2 / MU_F,
    }


def material_from_mean_k(k_md: np.ndarray) -> Dict[str, np.ndarray]:
    phi = invert_kc(k_md)
    young, lame, shear, alpha, storage = route1(phi)
    return {
        "logk": np.log(np.maximum(k_md, 1e-300) / K_REF_MD),
        "kmd": k_md,
        "phi": phi,
        "E": young,
        "lam": lame,
        "G": shear,
        "alpha_mech": alpha,
        "alpha_mass": alpha,
        "S": storage,
        "mobility": k_md * MD_TO_M2 / MU_F,
    }


def q1_reference(hx: float, hy: float) -> Dict[str, np.ndarray]:
    gp = 1 / np.sqrt(3)
    points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    det_j = hx * hy / 4
    k_lam = np.zeros((8, 8))
    k_g = np.zeros((8, 8))
    b_unit = np.zeros((8, 4))
    m_unit = np.zeros((4, 4))
    h_unit = np.zeros((4, 4))
    fbx = np.zeros(8)
    fby = np.zeros(8)
    gflow_unit = np.zeros(4)
    c_lam = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], float)
    c_g = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]], float)
    for xi, eta in points:
        n = 0.25 * np.array(
            [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]
        )
        dxi = 0.25 * np.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)])
        deta = 0.25 * np.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi])
        dx = dxi * 2 / hx
        dy = deta * 2 / hy
        grad = np.vstack([dx, dy])
        bmat = np.zeros((3, 8))
        div = np.zeros(8)
        for a in range(4):
            bmat[0, 2 * a] = dx[a]
            bmat[1, 2 * a + 1] = dy[a]
            bmat[2, 2 * a] = dy[a]
            bmat[2, 2 * a + 1] = dx[a]
            div[2 * a] = dx[a]
            div[2 * a + 1] = dy[a]
        k_lam += bmat.T @ c_lam @ bmat * det_j
        k_g += bmat.T @ c_g @ bmat * det_j
        b_unit += np.outer(div, n) * det_j
        m_unit += np.outer(n, n) * det_j
        h_unit += grad.T @ grad * det_j
        fbx[0::2] += n * det_j
        fby[1::2] += n * det_j
        gflow_unit += grad.T @ (RHO_F * GRAVITY) * det_j
    return {"Kl": k_lam, "Kg": k_g, "B": b_unit, "M": m_unit, "H": h_unit, "Fbx": fbx, "Fby": fby, "Gflow": gflow_unit}


class Geometry:
    def __init__(self, nx: int = 100, ny: int = 100, lx: float = 500.0, ly: float = 500.0, q_total: float = DEFAULT_Q_TOTAL):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.hx = lx / nx
        self.hy = ly / ny
        self.nxnode = nx + 1
        self.nynode = ny + 1
        self.nnode = self.nxnode * self.nynode
        self.nu = 2 * self.nnode
        self.np = self.nnode
        self.ndof = self.nu + self.np
        self.x = np.linspace(0, lx, self.nxnode)
        self.y = np.linspace(0, ly, self.nynode)
        xx, yy = np.meshgrid(self.x, self.y)
        self.node_xy = np.column_stack([xx.ravel(), yy.ravel()])

        conn = np.empty((nx * ny, 4), int)
        idx = 0
        for ey in range(ny):
            base = ey * self.nxnode
            for ex in range(nx):
                n0 = base + ex
                conn[idx] = [n0, n0 + 1, n0 + self.nxnode + 1, n0 + self.nxnode]
                idx += 1
        self.conn = conn
        self.ne = len(conn)
        udofs = np.empty((self.ne, 8), int)
        udofs[:, 0::2] = 2 * conn
        udofs[:, 1::2] = 2 * conn + 1
        self.udofs = udofs
        self.ref = q1_reference(self.hx, self.hy)
        self.rows_k = np.repeat(udofs, 8, axis=1).ravel()
        self.cols_k = np.tile(udofs, (1, 8)).ravel()
        self.rows_b = np.repeat(udofs, 4, axis=1).ravel()
        self.cols_b = np.tile(conn, (1, 8)).ravel()
        self.rows_p = np.repeat(conn, 4, axis=1).ravel()
        self.cols_p = np.tile(conn, (1, 4)).ravel()

        bottom = np.arange(self.nxnode)
        left = np.arange(0, self.nnode, self.nxnode)
        target_y = np.linspace(ly / 3, 2 * ly / 3, 6)
        y_indices = np.unique(np.rint(target_y / self.hy).astype(int))
        if len(y_indices) != 6:
            raise RuntimeError(f"Expected six distinct well/outer y indices, got {y_indices}")
        well_ix = int(round((lx / 2) / self.hx))
        self.well_nodes = y_indices * self.nxnode + well_ix
        self.outer_nodes = np.concatenate([y_indices * self.nxnode, y_indices * self.nxnode + nx]).astype(int)

        self.u_fixed = np.unique(np.concatenate([2 * bottom, 2 * bottom + 1, 2 * left])).astype(int)

        self.p_fixed = np.unique(self.outer_nodes).astype(int)
        self.all_u = np.arange(self.nu)
        self.u_free = np.setdiff1d(self.all_u, self.u_fixed)
        all_dofs = np.arange(self.ndof)
        self.fixed_global = np.concatenate([self.u_fixed, self.nu + self.p_fixed])
        self.free_global = np.setdiff1d(all_dofs, self.fixed_global)

        self.p0 = TOP_INITIAL_PRESSURE + SPECIFIC_WEIGHT * (ly - self.node_xy[:, 1])
        self.fe_unit = RHO_B * (GRAVITY[0] * self.ref["Fbx"] + GRAVITY[1] * self.ref["Fby"])
        self.fu = np.zeros(self.nu)
        np.add.at(self.fu, self.udofs.ravel(), np.tile(self.fe_unit, self.ne))

        self.q_total = float(q_total)
        self.q_well = np.zeros(self.np)
        self.q_well[self.well_nodes] = self.q_total / len(self.well_nodes)


def assemble(geom: Geometry, coeff: Dict[str, np.ndarray]):
    r = geom.ref
    lame = np.asarray(coeff["lam"]).ravel()
    shear = np.asarray(coeff["G"]).ravel()
    alpha_mech = np.asarray(coeff["alpha_mech"]).ravel()
    alpha_mass = np.asarray(coeff.get("alpha_mass", coeff["alpha_mech"])).ravel()
    storage = np.asarray(coeff["S"]).ravel()
    mobility = np.asarray(coeff["mobility"]).ravel()

    k = sp.coo_matrix(
        ((lame[:, None, None] * r["Kl"] + shear[:, None, None] * r["Kg"]).reshape(-1), (geom.rows_k, geom.cols_k)),
        shape=(geom.nu, geom.nu),
    ).tocsr()
    k.sum_duplicates()
    bm = sp.coo_matrix(
        ((alpha_mech[:, None, None] * r["B"]).reshape(-1), (geom.rows_b, geom.cols_b)),
        shape=(geom.nu, geom.np),
    ).tocsr()
    bm.sum_duplicates()
    ba = sp.coo_matrix(
        ((alpha_mass[:, None, None] * r["B"]).reshape(-1), (geom.rows_b, geom.cols_b)),
        shape=(geom.nu, geom.np),
    ).tocsr()
    ba.sum_duplicates()
    m = sp.coo_matrix(
        ((storage[:, None, None] * r["M"]).reshape(-1), (geom.rows_p, geom.cols_p)),
        shape=(geom.np, geom.np),
    ).tocsr()
    m.sum_duplicates()
    h = sp.coo_matrix(
        ((mobility[:, None, None] * r["H"]).reshape(-1), (geom.rows_p, geom.cols_p)),
        shape=(geom.np, geom.np),
    ).tocsr()
    h.sum_duplicates()
    tau = (math.sqrt(geom.hx * geom.hy) ** 2) / (12 * (lame + 2 * shear))
    t_stab = sp.coo_matrix(
        ((tau[:, None, None] * r["H"]).reshape(-1), (geom.rows_p, geom.cols_p)),
        shape=(geom.np, geom.np),
    ).tocsr()
    t_stab.sum_duplicates()
    gflow = np.zeros(geom.np)
    np.add.at(gflow, geom.conn.ravel(), (mobility[:, None] * r["Gflow"][None, :]).ravel())
    return k, bm, ba, m, h, t_stab, gflow


def target_standardize(field: np.ndarray, std_target: float) -> np.ndarray:
    
    out = np.asarray(field, dtype=np.float64)
    out = out - np.mean(out)
    realized = float(np.std(out, ddof=0))
    if not np.isfinite(realized) or realized <= 0.0:
        raise FloatingPointError(f"Invalid raw fBm standard deviation: {realized}")
    out = out / realized
    out = out - np.mean(out)
    out = out / np.std(out, ddof=0)
    out = out * float(std_target)

    out = out - np.mean(out)
    out = out * (float(std_target) / np.std(out, ddof=0))
    return out
