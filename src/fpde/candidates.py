# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.
# Tree representation follows the node types in Jinwoo Im's GPfSI software
# (Copyright (C) 2021 Jinwoo Im, https://github.com/Jinwoousc/GPSI). This module
# stores the poroelastic candidates as dictionaries and defines their allowed terms.
# GPfSI method: J. Im, C. B. Rizzo, F. P. J. de Barros and S. F. Masri (2021),
# Nonlinear Dynamics 104, 1781–1800. https://doi.org/10.1007/s11071-021-06335-0

import math
import numpy as np
from mechanics_operators import MANDATORY,OPTIONAL as MECH,DEFAULTS as MD,BOUNDS as MB,block_keys,operator_audit
CORE=('P_caputo','U1_x_time','U2_y_time','P_x_gamma','P_y_gamma','Q')
PRESSURE_OPTIONAL=('P_x_beta','P_y_beta','P_xx','P_yy','P_x','P_y','P','U1_y_time','U2_x_time')
MAP={'U1_x_time':'bxt','U2_y_time':'byt','P_x_gamma':'dx','P_y_gamma':'dy','Q':'cq','P_x_beta':'vxb','P_y_beta':'vyb','P_xx':'cxx','P_yy':'cyy','P_x':'cx','P_y':'cy','P':'reaction','U1_y_time':'b1y','U2_x_time':'b2x'}
BOUNDS={'alpha':(.05,1.),'gammax':(1.01,2.),'gammay':(1.01,2.),'betax':(.05,.999),'betay':(.05,.999),'qx':(0.,1.),'qy':(0.,1.),'alpha_B':(.01,1.),'r':(.03,.72),'g':(1e-5,200.),'bxt':(1e-8,100.),'byt':(1e-8,100.),'dx':(1e-8,1e4),'dy':(1e-8,1e4),'cq':(1e-8,1e7),'vxb':(-100.,100.),'vyb':(-100.,100.),'cxx':(-100.,0.),'cyy':(-100.,0.),'cx':(-100.,100.),'cy':(-100.,100.),'reaction':(-100.,100.),'b1y':(-100.,100.),'b2x':(-100.,100.),**MB}
LOG={'g','bxt','byt','dx','dy','cq'}
ORDER_KEYS=tuple(d[4] for d in MANDATORY.values())
ALL_OPTIONAL=PRESSURE_OPTIONAL+tuple(MECH)
MECHANICS_OPTIONAL=tuple(MECH)
def leaf(t,v):return dict(type=t,node=v,left=None,right=None)
def branch(op,a,b):return dict(type=0,node=op,left=a,right=b)
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
def build(p,active=()):
 chosen=set(active)
 if chosen-set(ALL_OPTIONAL):raise ValueError('Unknown support')
 terms=[leaf(2,'P_caputo')]
 for name,key in MAP.items():
  if name not in CORE and name not in chosen:continue
  val=float(p.get(key,0))*(-1 if key in ('dx','dy','cq') else 1)
  terms.append(branch('*',leaf(3,val),leaf(2,name)))
 while len(terms)>1:terms=[branch('+',terms[i],terms[i+1]) if i+1<len(terms) else terms[i] for i in range(0,len(terms),2)]
 mech={k:float(p[k]) for k in ('r','g','alpha_B')};mech.update({k:float(p.get(k,MD[k])) for k in ORDER_KEYS})
 for name in MECH:
  if name in chosen:
   for k in block_keys(name):mech[k]=float(p.get(k,MD[k]))
 return dict(schema='Q1_ANCHORED_R4_V1',tree=terms[0],orders={k:float(p[k]) for k in ('alpha','gammax','gammay','betax','betay','qx','qy')},mechanics=mech,mechanics_active=[k for k in MECH if k in chosen])
def pressure_active(c):return tuple(k for k in PRESSURE_OPTIONAL if k in linear(c['tree']))
def mechanics_active(c):return tuple(k for k in MECH if k in c.get('mechanics_active',()))
def active(c):return pressure_active(c)+mechanics_active(c)
def parse(c):
 if c.get('schema')!='Q1_ANCHORED_R4_V1':raise ValueError('Unsupported equation representation for these operators')
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

def initial_candidate(dataset_path, known):
    with np.load(dataset_path, allow_pickle=False) as dataset:
        lame = float(np.mean(dataset['mean_lambda_elem'], dtype=np.float64))
        shear = float(np.mean(dataset['mean_G_elem'], dtype=np.float64))
        biot = float(np.mean(dataset['mean_alpha_elem'], dtype=np.float64))
        storage = float(np.mean(dataset['mean_S_elem'], dtype=np.float64))
        mobility = float(np.mean(dataset['mean_mobility_elem'], dtype=np.float64))
    modulus = lame + 2.0 * shear
    pressure, ux, uy = np.asarray(known['scales'], dtype=float)
    length = float(np.ptp(known['x_m']))
    time = float(known['time_scale_s'])
    source = float(known['q_scale'])
    positive = np.array([modulus, shear, biot, storage, mobility, pressure, ux, uy,
                         length, time, source])
    if not np.all(np.isfinite(positive) & (positive > 0)):
        raise ValueError('Invalid material or normalization scales for Biot initialization')
    if not np.isclose(length, np.ptp(known['y_m'])):
        raise ValueError('The FPDE solver requires a square reservoir')
    diffusion = mobility * time / (storage * length**2)
    parameters = dict(
        alpha=1.0, gammax=2.0, gammay=2.0, betax=0.5, betay=0.5, qx=0.5, qy=0.5,
        alpha_B=biot, r=shear/modulus, g=biot*pressure*length/(modulus*ux),
        bxt=biot*ux/(storage*pressure*length), byt=biot*uy/(storage*pressure*length),
        dx=diffusion, dy=diffusion, cq=source*time/(storage*pressure),
    )
    candidate = build(parameters)
    parse(candidate)
    return candidate
def nodes(c):
 def count(d):return 1+(count(d['left']) if d['left'] else 0)+(count(d['right']) if d['right'] else 0)
 return count(c['tree'])+26+len(ORDER_KEYS)+sum(4+2*len(block_keys(k)) for k in mechanics_active(c))
def mutate(c,rng,structural=True,scale=1.):
 p=parse(c);a=set(active(c))
 if structural and rng.random()<.65:
  name=str(rng.choice(ALL_OPTIONAL))
  if name in a:a.remove(name)
  else:
   a.add(name)
   if name in MAP:
    key=MAP[name];p[key]=float(np.clip(rng.normal(0,.08),*BOUNDS[key]))
   else:
    d=MECH[name];p[d['amplitude']]=float(rng.normal(0,.025))
    if d['order']:p[d['order']]=float(rng.uniform(.1,3))
 keys=['r','g','alpha_B','alpha','gammax','gammay','qx','qy','bxt','byt','dx','dy','cq']+list(ORDER_KEYS)
 for name in sorted(a):
  keys+=([MAP[name]] if name in MAP else block_keys(name))
  if name=='P_x_beta':keys.append('betax')
  if name=='P_y_beta':keys.append('betay')
 keys=list(dict.fromkeys(keys))
 for key in rng.choice(keys,min(len(keys),int(rng.integers(1,5))),replace=False):
  lo,hi=BOUNDS[key];v=float(p[key])
  if key in LOG:v*=math.exp(float(rng.normal(0,.28*scale)))
  elif key.startswith('ord_'):
   v+=float(rng.normal(0,.12*scale))
   if rng.random()<.07:v=float(rng.uniform(lo,hi))
  elif key.startswith('amp_'):v+=float(rng.normal(0,max(.015,abs(v)*.3)*scale))
  else:v+=float(rng.normal(0,.025*(hi-lo)*scale))
  p[key]=float(np.clip(v,lo,hi))
 return build(p,a)
def cross(a,b,rng,structural=True):
 pa,pb=parse(a),parse(b);aa,ab=set(active(a)),set(active(b))
 for key in BOUNDS:
  if rng.random()<.5:pa[key],pb[key]=pb[key],pa[key]
 if structural:
  for name in ALL_OPTIONAL:
   if rng.random()<.5:
    va,vb=name in aa,name in ab;aa.discard(name);ab.discard(name)
    if vb:aa.add(name)
    if va:ab.add(name)
 return build(pa,aa),build(pb,ab)
def equations(c,scales):
 p=parse(c);s=float(scales[2]/scales[1]);r=p['r'];g=p['g']
 lhs=f"Caputo_theta^{p['alpha']:.12g} P + {p['bxt']:.12g} d2U1/(dxi dtheta) + {p['byt']:.12g} d2U2/(deta dtheta)"
 for name in pressure_active(c):lhs+=f" {p[MAP[name]]:+.12g} {name}"
 pressure=lhs+f" = {p['dx']:.12g} FluxDiv_xi(gamma={p['gammax']:.12g},q={p['qx']:.12g}; P,gravity) + {p['dy']:.12g} FluxDiv_eta(gamma={p['gammay']:.12g},q={p['qy']:.12g}; P,gravity) + {p['cq']:.12g} Q"
 def term(name,coef):return f"{coef:+.10g} Q1_anchored_weak[{name}](order={p[MANDATORY[name][4]]:.12g})"
 u1=' '.join([term('U1_U1_x',1),term('U1_U1_y',r),term('U1_U2_xy',(1-r)*s),term('U1_P_x',-g)])
 u2=' '.join([term('U2_U2_x',r),term('U2_U2_y',1),term('U2_U1_xy',(1-r)/s),term('U2_P_y',-g/s)])
 for name in mechanics_active(c):
  d=MECH[name];eq=d['eq'];field=d['field'];amp=p[d['amplitude']]
  coef=(-g*(1 if eq==0 else 1/s)*amp if field==2 else amp*float(scales[field+1]/scales[eq+1]))
  txt=f" {coef:+.10g} auxiliary_weak[{name}](order={p[d['order']]:.12g})" if d['order'] else f" {coef:+.10g} auxiliary_mass[{name}]"
  if eq==0:u1+=txt
  else:u2+=txt
 return dict(P=pressure,U1=u1+' = normalized Q1 gravity load',U2=u2+' = normalized Q1 gravity load',
  weak_system='K4 U_abs=(f_g+alpha_B*P_scale*B4 P_normalized)/C; C=alpha_B*P_scale*L/(g*U1_scale)',
  mixed_definition='separate lambda and G weak orientations retained; displayed mixed coefficient is the interior strong-form coefficient',
  coordinates='xi=x/500, eta=y/500, theta=t/train_end; P=P_abs/P_scale, Ui=Ui_abs/Ui_scale; no offset subtraction',
  mechanics_selection=list(mechanics_active(c)),mechanics_orders=operator_audit(p)['mechanics_mandatory_orders'],mechanics_order_bounds=[.1,3.],
  mechanics_operator='Q1 + auxiliary weak(order) - auxiliary weak(reference order)',
  pressure_order_bounds='time 0.05--1; high spatial 1.01--2; low optional spatial 0.05--0.999',
  initialization='P0 exact; candidate mechanics computes U0; mismatch reported; NO B0',alpha_B=p['alpha_B'],r=r,g=g)
