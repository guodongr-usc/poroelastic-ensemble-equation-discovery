# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

import math
from mechanics_operators import MANDATORY,OPTIONAL as MECH,DEFAULTS as MD,BOUNDS as MB,block_keys
CORE=('P_caputo','U1_x_time','U2_y_time','P_x_gamma','P_y_gamma','Q')
PRESSURE_OPTIONAL=('P_x_beta','P_y_beta','P_xx','P_yy','P_x','P_y','P','U1_y_time','U2_x_time')
MAP={'U1_x_time':'bxt','U2_y_time':'byt','P_x_gamma':'dx','P_y_gamma':'dy','Q':'cq','P_x_beta':'vxb','P_y_beta':'vyb','P_xx':'cxx','P_yy':'cyy','P_x':'cx','P_y':'cy','P':'reaction','U1_y_time':'b1y','U2_x_time':'b2x'}
BOUNDS={'alpha':(.05,1.),'gammax':(1.01,2.),'gammay':(1.01,2.),'betax':(.05,.999),'betay':(.05,.999),'qx':(0.,1.),'qy':(0.,1.),'alpha_B':(.01,1.),'r':(.03,.72),'g':(1e-5,200.),'bxt':(1e-8,100.),'byt':(1e-8,100.),'dx':(1e-8,1e4),'dy':(1e-8,1e4),'cq':(1e-8,1e7),'vxb':(-100.,100.),'vyb':(-100.,100.),'cxx':(-100.,0.),'cyy':(-100.,0.),'cx':(-100.,100.),'cy':(-100.,100.),'reaction':(-100.,100.),'b1y':(-100.,100.),'b2x':(-100.,100.),**MB}
def linear(d):
 if d['type']==2:return {d['node']:1.}
 if d['type']==3:return {'constant':float(d['node'])}
 A,B=linear(d['left']),linear(d['right']);op=d['node']
 if op in ('+','-'):
  for k,v in B.items():A[k]=A.get(k,0)+(v if op=='+' else -v)
  return A
 if op=='*' and set(A)=={'constant'}:return {k:A['constant']*v for k,v in B.items()}
 if op=='*' and set(B)=={'constant'}:return {k:B['constant']*v for k,v in A.items()}
 raise ValueError('Only linear derivative combinations supported')

def pressure_active(c):return tuple(k for k in PRESSURE_OPTIONAL if k in linear(c['tree']))

def mechanics_active(c):return tuple(k for k in MECH if k in c.get('mechanics_active',()))

def parse(c):
 if c.get('schema')!='Q1_ANCHORED_R4_V1':raise ValueError('Incompatible fractional PDE parameter schema')
 t=linear(c['tree'])
 if set(CORE)-set(t) or set(t)-set(CORE)-set(PRESSURE_OPTIONAL) or t['P_caputo']!=1:raise ValueError('Pressure grammar')
 if set(c.get('mechanics_active',()))-set(MECH):raise ValueError('Mechanics grammar')
 p={**MD,**c['orders'],**c['mechanics']}
 for name in MECH:
  if name not in mechanics_active(c):
   for k in block_keys(name):p[k]=MD[k]
 for name,key in MAP.items():p[key]=float(t.get(name,0))*(-1 if key in ('dx','dy','cq') else 1)
 for k,(lo,hi) in BOUNDS.items():
  if k not in p or not math.isfinite(float(p[k])) or not lo<=p[k]<=hi:raise ValueError(f'Out-of-bounds {k}: {p.get(k)}')
 p['_mechanics_active']=mechanics_active(c);return p
