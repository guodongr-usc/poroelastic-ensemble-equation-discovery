# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.
# Code source: B-RefStruct, spatial_kernels.py, distributed with the archived
# poroelastic solver. Its license text is in the accompanying LICENSE file.
# SPDX-License-Identifier: GPL-3.0-or-later
# For the shifted Grünwald spatial stencil, see M. M. Meerschaert and C. Tadjeran
# (2004), J. Comput. Appl. Math. 172, 65–77. https://doi.org/10.1016/j.cam.2004.01.033

import numpy as np
GRUNWALD_TAIL_EXTRA=1280

def grunwald_weights(order: float, kmax: int) -> np.ndarray:
    w = np.empty(kmax + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, kmax + 1):
        w[k] = w[k-1] * ((k - 1.0 - float(order)) / k)
    return w

def beta_left_matrix(n: int, h: float, beta: float, left_dirichlet: bool) -> np.ndarray:
    beta = float(np.clip(beta, 1.0e-12, 1.0))
    K = n + GRUNWALD_TAIL_EXTRA
    w = grunwald_weights(beta, K)
    tail = np.cumsum(w[::-1])[::-1]
    A = np.zeros((n, n), dtype=np.float64)
    s = h ** (-beta)
    for i in range(n):

        ks = np.arange(i, -1, -1)
        A[i, :i+1] += w[ks] * s

        if not left_dirichlet:
            A[i, 0] += tail[i+1] * s
    return A

def gamma_shifted_matrix(
    n: int, h: float, gamma_order: float, q: float,
    left_dirichlet: bool, right_dirichlet: bool,
) -> np.ndarray:
    g = float(np.clip(gamma_order, 1.0 + 1.0e-12, 2.0))
    q = float(np.clip(q, 0.0, 1.0))
    K = n + GRUNWALD_TAIL_EXTRA
    w = grunwald_weights(g, K)
    tail = np.cumsum(w[::-1])[::-1]
    L = np.zeros((n, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    s = h ** (-g)

    for i in range(n):

        jmax = min(n-1, i+1)
        for j in range(0, jmax+1):
            k = i + 1 - j
            L[i, j] += w[k] * s

        if i == n-1 and not right_dirichlet:
            L[i, n-1] += w[0] * s

        if not left_dirichlet:
            L[i, 0] += tail[i+2] * s


        jmin = max(0, i-1)
        for j in range(jmin, n):
            k = j - i + 1
            R[i, j] += w[k] * s

        if i == 0 and not left_dirichlet:
            R[i, 0] += w[0] * s

        if not right_dirichlet:
            start = n - i + 1
            if start <= K:
                R[i, n-1] += tail[start] * s

    return q * L + (1.0-q) * R

def _classic_line_matrix(n: int, h: float, derivative: int, left_dirichlet: bool, right_dirichlet: bool) -> np.ndarray:
    
    if n < 3:
        raise ValueError("classic line operator requires n>=3")
    M = np.zeros((n, n), dtype=np.float64)
    if derivative == 1:
        for i in range(1, n-1):
            M[i, i-1] = -0.5/h
            M[i, i+1] = 0.5/h











        M[0, 0], M[0, 1], M[0, 2] = -1.5/h, 2.0/h, -0.5/h
        M[-1, -1], M[-1, -2], M[-1, -3] = 1.5/h, -2.0/h, 0.5/h
    elif derivative == 2:
        for i in range(1, n-1):
            M[i, i-1], M[i, i], M[i, i+1] = 1.0/h**2, -2.0/h**2, 1.0/h**2
        if left_dirichlet:
            M[0, 0], M[0, 1], M[0, 2] = 1.0/h**2, -2.0/h**2, 1.0/h**2
        else:
            M[0, 0], M[0, 1] = -1.0/h**2, 1.0/h**2
        if right_dirichlet:
            M[-1, -3], M[-1, -2], M[-1, -1] = 1.0/h**2, -2.0/h**2, 1.0/h**2
        else:
            M[-1, -2], M[-1, -1] = 1.0/h**2, -1.0/h**2
    else:
        raise ValueError("derivative must be 1 or 2")
    return M
