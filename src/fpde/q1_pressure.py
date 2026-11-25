# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from functools import lru_cache
import numpy as np
import scipy.sparse as sp
from scipy.linalg import toeplitz
from scipy.special import gamma as Gamma

@lru_cache(maxsize=12)
def line_parts(n, order, q):
    if n < 3 or not 1.0 < order <= 2.0 or not 0 <= q <= 1:
        raise ValueError('Invalid Q1 fractional flux arguments')
    h=1.0/(n-1);m=n-1;delta=2.0-float(order)
    G=sp.diags([-np.ones(m),np.ones(m)],[0,1],shape=(m,n),format='csr')/h
    if abs(delta)<1e-13:
        T=sp.eye(m,format='csr')
    else:
        k=np.arange(m,dtype=float)
        w=h**delta*((k+1.0)**delta-k**delta)/Gamma(1.0+delta)
        lo=toeplitz(w,np.r_[w[0],np.zeros(m-1)])
        T=sp.csr_matrix(q*lo+(1.0-q)*lo.T)
    K=(G.T@(h*T)@G).tocsr()
    fg=np.asarray(G.T@(h*T@np.ones(m))).ravel()
    diag=np.full(n,4.0);diag[[0,-1]]=2.0
    M=sp.diags([np.ones(n-1),diag,np.ones(n-1)],[-1,0,1],format='csr')*(h/6.0)
    return K,fg,M


def flux_parts(n, axis, order, q, gravity_gradient):
    K,fg,M=line_parts(int(n),float(order),float(q));ones=np.ones(n)
    lump=np.asarray(M@ones).ravel()
    if axis==0:
        return sp.kron(M,K,format='csr'),float(gravity_gradient)*np.kron(lump,fg)
    if axis==1:
        return sp.kron(K,M,format='csr'),float(gravity_gradient)*np.kron(fg,lump)
    raise ValueError('axis must be 0 or 1')
