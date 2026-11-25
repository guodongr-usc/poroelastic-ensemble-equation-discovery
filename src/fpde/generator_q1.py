# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.
# Element and assembly routines are drawn from this study's realization generator,
# src/material_sim/base/fbm500_biot_core.py. For the coupled pressure–displacement
# formulation see M. A. Biot (1941), https://doi.org/10.1063/1.1712886.

from __future__ import annotations
import math
from typing import Dict
import numpy as np
import scipy.sparse as sp
RHO_F=1000.0
RHO_B=2242.609582
GRAVITY=np.array([0.,-9.81])
SPECIFIC_WEIGHT=RHO_F*9.81
TOP_INITIAL_PRESSURE=9.81e6
OUTER_PRESSURE=1.0e7
P_SCALE=1.0e7
DEFAULT_Q_TOTAL=6.0e-5
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
