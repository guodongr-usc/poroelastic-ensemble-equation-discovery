# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from .operators import HEADS, derivative_operators

BIOT_UNITS = dict(lambda_=1e9, G=1e9, alpha=1., S=1e-9,
                  mobility_x=1e-11, mobility_y=1e-11)


OLD_NAMES = ('1', 'U1_I', 'U2_I', 'P_I', 'U1_Dx', 'U1_Dy',
             'U2_Dx', 'U2_Dy', 'P_Dx', 'P_Dy')


@dataclass(frozen=True)
class Feature:
    name: str
    R: object
    temporal: bool
    offset: np.ndarray
    family: str = 'old_closure'
    corner: int | None = None
    unit: float = 1.
    element_for_node: np.ndarray | None = None


def _scales(value, n, name):
    value = np.asarray(value, dtype=float)
    if value.shape == (3,):
        value = np.repeat(value, n)
    if value.shape != (3*n,) or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError(name + ' must be positive finite, shape (3,) or (3*n,)')
    return value


def build_library(engine, state_scales, row_scales, ops=None):
    
    n = int(engine.nn)
    state = _scales(state_scales, n, 'state_scales')
    rows = _scales(row_scales, n, 'row_scales').reshape(3, n)
    conn = np.asarray(engine.conn, dtype=int)
    if conn.ndim != 2 or conn.shape[1] != 4:
        raise ValueError('Native Q1 four-node connectivity required')
    if ops is None:
        ops = derivative_operators(engine.x, engine.y)
    zero = sp.csr_matrix((n, n))
    old = [Feature('1', None, False, np.ones(n))]
    for derivative, factor in [('I', 1.), ('Dx', 500.), ('Dy', 500.)]:

        if derivative != 'I':
            continue
        for h, name in enumerate(HEADS):
            blocks = [zero, zero, zero]
            blocks[h] = ops[derivative] * factor
            old.append(Feature(name+'_'+derivative, sp.hstack(blocks, format='csr'),
                               False, np.zeros(n)))
    for h, name in enumerate(HEADS):
        for derivative in ('Dx', 'Dy'):
            blocks = [zero, zero, zero]
            blocks[h] = ops[derivative] * 500.
            old.append(Feature(name+'_'+derivative, sp.hstack(blocks, format='csr'),
                               False, np.zeros(n)))
    assert tuple(f.name for f in old) == OLD_NAMES
    result = [old.copy() for _ in HEADS]
    local = engine.local


    ucolumns = np.empty((len(conn), 8), dtype=int)
    ucolumns[:, 0::2] = conn
    ucolumns[:, 1::2] = n + conn
    pcolumns = 2*n + conn
    adjacency = []
    for corner in range(4):
        ids = np.full(n, -1, dtype=int)
        ids[conn[:, corner]] = np.arange(len(conn))
        adjacency.append(ids)

    def add(head, family, corner, values, columns, unit, temporal=False,
            local_offset=None):
        node = conn[:, corner]
        sparse_rows = np.broadcast_to(node[:, None], columns.shape).ravel()
        R = sp.coo_matrix((np.asarray(values).ravel(),
                           (sparse_rows, columns.ravel())), shape=(n, 3*n)).tocsr()
        R = R.multiply(state[None, :]).multiply((unit / rows[head])[:, None]).tocsr()
        R.eliminate_zeros()
        offset = np.zeros(n)
        if local_offset is not None:
            offset[node] = np.asarray(local_offset) * unit / rows[head, node]
        result[head].append(Feature('biot_'+family+'_corner'+str(corner), R,
                                    temporal, offset, family, corner, unit,
                                    adjacency[corner]))

    tau_unit = float(engine.hx * engine.hy / (12.*(1e9+2e9)))
    for h in (0, 1):
        for family, local_matrix, unit in (
            ('lambda', local.K_lambda, BIOT_UNITS['lambda_']),
            ('G', local.K_G, BIOT_UNITS['G']),
            ('alpha', -local.Q, BIOT_UNITS['alpha'])):
            columns = pcolumns if family == 'alpha' else ucolumns
            for corner in range(4):
                add(h, family, corner, local_matrix[:, 2*corner+h, :], columns, unit)
    for family, matrix, columns, unit, temporal in (
        ('alpha_U1t', local.Q[:, 0::2, :].transpose(0, 2, 1), conn, 1., True),
        ('alpha_U2t', local.Q[:, 1::2, :].transpose(0, 2, 1), n+conn, 1., True),
        ('S', local.M, pcolumns, BIOT_UNITS['S'], True),
        ('mobility_x', local.Hx, pcolumns, BIOT_UNITS['mobility_x'], False),
        ('mobility_y', local.Hy, pcolumns, BIOT_UNITS['mobility_y'], False),
        ('tau', engine.gradient_local, pcolumns, tau_unit, True)):
        for corner in range(4):
            local_offset = -engine.gravity_local[:, corner] if family == 'mobility_y' else None
            add(2, family, corner, matrix[:, corner, :], columns, unit, temporal, local_offset)
    assert [len(l) for l in result] == [22, 22, 34]
    return result


def rates(states, times):
    states, times = np.asarray(states, float), np.asarray(times, float)
    if states.ndim != 2 or times.shape != (len(states),) or np.any(np.diff(times) <= 0):
        raise ValueError('states=(nt,3*n), strictly increasing times=(nt,) required')
    rate = np.zeros_like(states)
    rate[1:] = np.diff(states, axis=0) / np.diff(times)[:, None]
    return rate


def assemble(libraries, weights):
    
    n = len(libraries[0][0].offset)
    matrices, masses, offsets = [], [], []
    for library, coefficient in zip(libraries, weights):
        coefficient = np.asarray(coefficient, dtype=float)
        if coefficient.shape != (n, len(library)) or not np.isfinite(coefficient).all():
            raise ValueError('Each head weight array must be finite, shape (n,features)')
        a = sp.csr_matrix((n, 3*n)); m = a.copy(); offset = np.zeros(n)
        for j, feature in enumerate(library):
            w = coefficient[:, j]
            offset += w * feature.offset
            if feature.R is not None:
                term = feature.R.multiply(w[:, None])
                if feature.temporal:
                    m += term
                else:
                    a += term
        matrices.append(a); masses.append(m); offsets.append(offset)
    return sp.vstack(matrices, format='csr'), sp.vstack(masses, format='csr'), np.concatenate(offsets)


def embed_biot(libraries, coefficients, tau):
    
    coefficient = np.asarray(coefficients, dtype=float)
    tau = np.asarray(tau, dtype=float)
    if coefficient.ndim != 2 or coefficient.shape[1] != 6 or tau.shape != coefficient.shape[:1]:
        raise ValueError('Expected supplied (nelem,6) coefficients and (nelem,) tau')
    if not np.isfinite(coefficient).all() or not np.isfinite(tau).all():
        raise ValueError('Diagnostic supplied coefficients must be finite')
    indices = dict(lambda_=0, G=1, alpha=2, alpha_U1t=2, alpha_U2t=2,
                   S=3, mobility_x=4, mobility_y=5)
    result = []
    for library in libraries:
        n = len(library[0].offset)
        weights = np.zeros((n, len(library)))
        for j, feature in enumerate(library):
            if feature.family == 'old_closure':
                continue
            ids = feature.element_for_node
            keep = ids >= 0
            if np.any(ids[keep] >= len(coefficient)):
                raise ValueError('Supplied coefficient element count differs from library')
            values = tau if feature.family == 'tau' else coefficient[:, indices['lambda_' if feature.family == 'lambda' else feature.family]]
            weights[keep, j] = values[ids[keep]] / feature.unit
        result.append(weights)
    return result
