# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

import gc
import numpy as np
import scipy.sparse.linalg as sla
from poroelastic_solver import PoroelasticSolver, NumericalFailure, time_grid, l1_weights, manufactured_errors
from mechanics_operators import corrections,classical,operator_audit,MANDATORY,OPTIONAL
class CPUSolver(PoroelasticSolver):
 def matrices(self,p):
  r=float(p['r'])
  if not 0<r<.75:raise NumericalFailure('Invalid baseline Q1 modulus ratio')
  full=(1-2*r)*self.Kl+r*self.Kg;B=self.B
  if not classical(p):
   dk,db=corrections(self.n,p,self.L);full=full+dk;B=B+db
  return full[self.ufree][:,self.ufree].tocsc(),B[self.ufree].tocsr(),self.fu[self.ufree]
 def mechanics(self,p):
  active=tuple(p.get('_mechanics_active',()));keys=['r']+[d[4] for d in MANDATORY.values()]
  for k in active:
   d=OPTIONAL[k];keys.append(d['amplitude'])
   if d['order']:keys.append(d['order'])
  key=(active,tuple((k,float(p[k])) for k in keys))
  if key in self.cache:return self.cache[key]
  self.cache.clear();gc.collect()
  K,B,fg=self.matrices(p)
  if not np.isfinite(K.data).all() or not np.isfinite(B.data).all():raise NumericalFailure('Nonfinite matrix')
  try:lu=sla.splu(K)
  except RuntimeError as e:raise NumericalFailure('Anchored mechanics factorization: '+str(e)) from e
  value=(lu,B,K,fg);self.cache[key]=value;return value
 def forward(self,p,obs,substeps=2,direct_reference=False):
  pred,info=super().forward(p,obs,substeps,direct_reference)
  info.update(operator_audit(p));info['spatial_discretization']='Stabilized Q1 poroelasticity with weak fractional mechanics corrections'
  return pred,info
