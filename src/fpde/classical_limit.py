#!/usr/bin/env python3
# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

import runtime
from pathlib import Path
import time,argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from data_io import load_npz
from poroelastic_solver import PoroelasticSolver
from solver_cpu import CPUSolver
from candidates import parse
from runtime import load_json, save_json


class ClassicalQ1Solver(PoroelasticSolver):
    def mechanics(self, parameters):
        r = float(parameters['r'])
        if r in self.cache:
            return self.cache[r]
        geom = self.geom
        x = np.array([0, 1, 1, 0])
        y = np.array([0, 0, 1, 1])
        slope = np.array([-1., 1.])
        mass = np.array([[2., 1.], [1., 2.]]) / 6.
        mass_x, mass_y = mass[np.ix_(x, x)], mass[np.ix_(y, y)]
        xx = (geom.hy / geom.hx) * np.outer(slope[x], slope[x]) * mass_y
        yy = (geom.hx / geom.hy) * np.outer(slope[y], slope[y]) * mass_x
        xy = np.outer(slope[x], slope[y]) / 4.
        stiffness = np.empty((8, 8))
        stiffness[0::2, 0::2] = xx + r * yy
        stiffness[1::2, 1::2] = r * xx + yy
        stiffness[0::2, 1::2] = (1. - 2. * r) * xy + r * xy.T
        stiffness[1::2, 0::2] = stiffness[0::2, 1::2].T
        coupling = np.empty((8, 4))
        coupling[0::2] = (geom.hy / 2.) * slope[x, None] * mass_y
        coupling[1::2] = (geom.hx / 2.) * slope[y, None] * mass_x
        K = sp.coo_matrix((np.tile(stiffness.ravel(), geom.ne), (geom.rows_k, geom.cols_k)),
                          shape=(geom.nu, geom.nu)).tocsr()[self.ufree][:, self.ufree].tocsc()
        B = sp.coo_matrix((np.tile(coupling.ravel(), geom.ne), (geom.rows_b, geom.cols_b)),
                          shape=(geom.nu, geom.np)).tocsr()[self.ufree]
        body = np.zeros(geom.nu)
        element_load = np.tile(self.rho_b * self.gravity * geom.hx * geom.hy / 4., 4)
        np.add.at(body, geom.udofs.ravel(), np.tile(element_load, geom.ne))
        self.cache[r] = sla.splu(K), B, K, body[self.ufree]
        return self.cache[r]


def rel(a,b):
 if hasattr(a,'tocsr'):return float(sla.norm(a-b)/max(float(sla.norm(b)),1e-30))
 return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(b)),1e-30))

def check_classical_limit(root,grids=(21,51,101),frames=None,substeps=2):
 root=Path(root);known=load_npz(root/'prepared/KNOWN_INPUTS.npz');p=parse(load_json(root/'prepared/INITIAL_CANDIDATE.json'))
 k=int(known['n_train']);nf=k if frames is None else min(k,int(frames));t0=time.time()
 report=dict(passed=False,training_only=True,heldout_truth_read=False,parameter_updates=0,initial_compensation=False,matrices={},trajectories={},substeps=substeps)
 for n in grids:
  a=ClassicalQ1Solver(known,n,rtol=1e-9,max_seconds=10800);b=CPUSolver(known,n,rtol=1e-9,max_seconds=10800)
  _,B0,K0,f0=a.mechanics(p);K1,B1,f1=b.matrices(p);u0,i0=a.initial_equilibrium(p);u1,i1=b.initial_equilibrium(p)
  checks=dict(K_relative=rel(K1,K0),B_relative=rel(B1,B0),fg_relative=rel(f1,f0),
   C_relative=abs(b.modulus(p)-a.modulus(p))/a.modulus(p),
   rhs_relative=rel((f1+p['alpha_B']*b.sc[0]*(B1@b.p0))/b.modulus(p),(f0+p['alpha_B']*a.sc[0]*(B0@a.p0))/a.modulus(p)),
   U0_relative=rel(u1,u0),free_dofs_equal=bool(np.array_equal(a.ufree,b.ufree)),
   reference_IC_mismatch=i0['initial_mechanics_relative_mismatch'],new_IC_mismatch=i1['initial_mechanics_relative_mismatch'])
  report['matrices'][str(n)]=checks;print('MATRIX AUDIT',n,checks,flush=True)
  if max(checks[z] for z in ('K_relative','B_relative','fg_relative','C_relative','rhs_relative','U0_relative'))>1e-10 or not checks['free_dofs_equal']:
   save_json(root/'Q1_CLASSICAL_LIMIT.json',report);raise RuntimeError('Classical matrix regression failed')
  count=nf if n==max(grids) else min(nf,4)
  a0,s0=a.forward(p,known['theta'][:count],substeps);a1,s1=b.forward(p,known['theta'][:count],substeps)
  delta={f:rel(a1[j],a0[j]) for j,f in enumerate(('P','U1','U2'))}
  report['trajectories'][str(n)]=dict(frames=count,relative_difference=delta,reference_pressure_residual=s0['max_true_pressure_residual'],new_pressure_residual=s1['max_true_pressure_residual'])
  print('TRAJECTORY AUDIT',n,count,delta,flush=True)
  if max(delta.values())>1e-7:save_json(root/'Q1_CLASSICAL_LIMIT.json',report);raise RuntimeError('Matched trajectory regression failed')
  del a,b,a0,a1
 q=p.copy();q.update(r=.3,g=1.7,alpha_B=.7)
 a=ClassicalQ1Solver(known,21,rtol=1e-9,max_seconds=1000);b=CPUSolver(known,21,rtol=1e-9,max_seconds=1000)
 _,Br,Kr,fr=a.mechanics(q);Kn,Bn,fn=b.matrices(q)
 report['second_parameter_matrices']=dict(K=rel(Kn,Kr),B=rel(Bn,Br),fg=rel(fn,fr))
 if max(report['second_parameter_matrices'].values())>1e-10:raise RuntimeError('Second-parameter regression failed')
 report.update(passed=True,production_native_gate=(101 in grids and nf==k),elapsed_seconds=time.time()-t0)
 save_json(root/'Q1_CLASSICAL_LIMIT.json',report);print('Q1_CLASSICAL_LIMIT PASSED',report['production_native_gate'],flush=True);return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--frames',type=int);ap.add_argument('--grids',type=int,nargs='+',default=[21,51,101]);ap.add_argument('--substeps',type=int,default=2);a=ap.parse_args();check_classical_limit(a.outdir,a.grids,a.frames,a.substeps)
