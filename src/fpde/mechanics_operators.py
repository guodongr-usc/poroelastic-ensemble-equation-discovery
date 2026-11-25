# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from functools import lru_cache
import math
import numpy as np
import scipy.sparse as sp
ORDER_BOUNDS=(0.1,3.0)
HORIZON=0.1

MANDATORY={
 'U1_U1_x':(0,0,'even',0,'ord_U1_U1_x',2.),
 'U1_U1_y':(0,0,'even',1,'ord_U1_U1_y',2.),
 'U1_U2_xy':(0,1,'mixed',None,'ord_U1_U2_xy',2.),
 'U1_P_x':(0,2,'gradient',0,'ord_U1_P_x',1.),
 'U2_U2_x':(1,1,'even',0,'ord_U2_U2_x',2.),
 'U2_U2_y':(1,1,'even',1,'ord_U2_U2_y',2.),
 'U2_U1_xy':(1,0,'mixed',None,'ord_U2_U1_xy',2.),
 'U2_P_y':(1,2,'gradient',1,'ord_U2_P_y',1.)}
OPTIONAL={}
for eq in (0,1):
 own=eq;other=1-eq;prefix=f'U{eq+1}'
 def add(label,field,family,axis):
  name=prefix+'_'+label
  OPTIONAL[name]=dict(eq=eq,field=field,family=family,axis=axis,amplitude='amp_'+name,
                     order=None if family=='zero' else 'ord_'+name)
 for a,l in enumerate(('x','y')):add(f'U{own+1}_{l}_odd',own,'gradient',a)
 add(f'U{own+1}_xy',own,'mixed',None)
 for fam in ('even','odd'):
  for a,l in enumerate(('x','y')):add(f'U{other+1}_{l}_{fam}',other,'gradient' if fam=='odd' else 'even',a)
 add('P_'+('y' if eq==0 else 'x')+'_odd',2,'gradient',1-eq)
 for a,l in enumerate(('x','y')):add(f'P_{l}_even',2,'even',a)
 for field,l in ((own,f'U{own+1}'),(other,f'U{other+1}'),(2,'P')):add(l+'_0',field,'zero',None)
DEFAULTS={d[4]:d[5] for d in MANDATORY.values()}
BOUNDS={k:ORDER_BOUNDS for k in DEFAULTS}
for d in OPTIONAL.values():
 DEFAULTS[d['amplitude']]=0.;BOUNDS[d['amplitude']]=(-30.,30.)
 if d['order']:
  DEFAULTS[d['order']]=1. if d['family']=='gradient' else 2.
  BOUNDS[d['order']]=ORDER_BOUNDS

def block_keys(name):
 d=OPTIONAL[name];return [d['amplitude']]+([d['order']] if d['order'] else [])
def classical(p):
 return not p.get('_mechanics_active',()) and all(float(p[d[4]])==d[5] for d in MANDATORY.values())

@lru_cache(maxsize=96)
def line_gradient(n,order,side='left'):
 
 if not .05<=float(order)<=3.:raise ValueError('auxiliary order outside 0.05--3')
 if side not in ('left','right'):raise ValueError(side)
 h=1./(n-1);kmax=min(n-1,max(1,int(math.ceil(HORIZON/h))))
 w=np.ones(kmax+1)
 for k in range(1,kmax+1):w[k]=w[k-1]*(k-1-order)/k
 rows=[];cols=[];vals=[]
 for i in range(n):
  acc={}
  for k in range(1,kmax+1):
   if w[k]==0.:continue
   j=max(0,i-k) if side=='left' else min(n-1,i+k)
   acc[j]=acc.get(j,0.)+w[k];acc[i]=acc.get(i,0.)-w[k]
  for j,v in acc.items():
   if v:rows.append(i);cols.append(j);vals.append(v*h**(-order))
 return sp.csr_matrix((vals,(rows,cols)),shape=(n,n))
@lru_cache(maxsize=8)
def quadrature(n):
 h=1./(n-1);q=np.full(n,h);q[[0,-1]]*=.5
 return sp.diags(np.kron(q,q),format='csr')
def grad2(n,axis,order,side='left'):
 g=line_gradient(n,float(order),side);I=sp.eye(n,format='csr')
 return sp.kron(I,g,format='csr') if axis==0 else sp.kron(g,I,format='csr')
def weak_even(n,axis,order):
 W=quadrature(n);a=grad2(n,axis,float(order)/2,'left');b=grad2(n,axis,float(order)/2,'right')
 return (.5*(a.T@W@a+b.T@W@b)).tocsr()
def weak_mixed_parts(n,order):
 W=quadrature(n);x=grad2(n,0,float(order)/2);y=grad2(n,1,float(order)/2)
 a=(x.T@W@y).tocsr();return a,a.T.tocsr()
def weak_gradient(n,axis,order):

 return (grad2(n,axis,float(order)).T@quadrature(n)).tocsr()
def auxiliary(n,family,axis,order):
 if family=='even':return weak_even(n,axis,order)
 if family=='gradient':return weak_gradient(n,axis,order)
 if family=='mixed':
  a,b=weak_mixed_parts(n,order);return .5*(a+b)
 if family=='zero':return quadrature(n)
 raise ValueError(family)
def embed(A,row,field,N):
 a=A.tocoo()
 if field==2:return sp.csr_matrix((a.data,(2*a.row+row,a.col)),shape=(2*N,N))
 return sp.csr_matrix((a.data,(2*a.row+row,2*a.col+field)),shape=(2*N,2*N))

def corrections(n,p,L):
 
 N=n*n;r=float(p['r']);lam=1-2*r
 K=sp.csr_matrix((2*N,2*N));B=sp.csr_matrix((2*N,N))
 for name,(eq,field,fam,axis,key,ref) in MANDATORY.items():
  value=float(p[key])
  if value==ref:continue
  if fam=='even':
   coef=1. if name in ('U1_U1_x','U2_U2_y') else r
   d=coef*(weak_even(n,axis,value)-weak_even(n,axis,ref))
  elif fam=='gradient':d=L*(weak_gradient(n,axis,value)-weak_gradient(n,axis,ref))
  else:
   a,b=weak_mixed_parts(n,value);a0,b0=weak_mixed_parts(n,ref)

   d=lam*(a-a0)+r*(b-b0) if eq==0 else lam*(b-b0)+r*(a-a0)
  if field==2:B=B+embed(d,eq,2,N)
  else:K=K+embed(d,eq,field,N)
 for name in p.get('_mechanics_active',()):
  q=OPTIONAL[name];amp=float(p[q['amplitude']])
  if amp==0.:continue
  order=None if q['order'] is None else float(p[q['order']])
  d=amp*auxiliary(n,q['family'],q['axis'],order)
  if q['field']==2:B=B+embed(L*d,q['eq'],2,N)
  else:K=K+embed(d,q['eq'],q['field'],N)
 K.eliminate_zeros();B.eliminate_zeros();return K.tocsr(),B.tocsr()
def operator_audit(p):
 return dict(mechanics_anchor='Q1 plane-strain elasticity and pressure coupling integrated on the problem mesh',
  mechanics_mandatory_orders={k:float(p[d[4]]) for k,d in MANDATORY.items()},
  mechanics_order_bounds=list(ORDER_BOUNDS),mechanics_selected_blocks=list(p.get('_mechanics_active',())),
  mechanics_classical_exact=classical(p),mechanics_horizon_dimensionless=HORIZON,
  mechanics_operator_definition='Q1 + auxiliary weak(order) - auxiliary weak(reference order)',
  mechanics_auxiliary_definition='finite-horizon GL-gradient energies and weak test gradients',
  mechanics_boundary='Q1 natural traction at classical orders; constant extension in auxiliary stencil at noninteger orders',
  mechanics_global_dynamic_stability_guaranteed=False,mechanics_time_model='quasistatic; no U_t/U_tt',
  mechanics_physics_limit='phenomenological discrete family; exact classical recovery is not noninteger continuum/objectivity proof')
