# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.
# For the L1 discretization of the Caputo time derivative on a nonuniform mesh,
# see M. Stynes, E. O'Riordan and J. L. Gracia (2017), SIAM J. Numer. Anal. 55,
# 1057–1079. https://doi.org/10.1137/16M1082329 (method used in l1_weights).

from __future__ import annotations
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.special import gamma
from generator_q1 import Geometry
from q1_pressure import flux_parts
from spatial_kernels import beta_left_matrix,_classic_line_matrix

class NumericalFailure(RuntimeError):
    pass

def time_grid(times,substeps,extra_times=None):
    times=np.asarray(times,float);s=int(substeps)
    if s<1 or times[0]!=0 or np.any(np.diff(times)<=0):
        raise ValueError('Invalid causal time grid')
    grid=[0.]
    for lo,hi in zip(times[:-1],times[1:]):
        pts=np.linspace(lo,hi,s+1)[1:];pts[-1]=hi;grid.extend(pts)
    grid=np.asarray(grid)
    if extra_times is not None:
        extra=np.asarray(extra_times,float);extra=extra[(extra>0)&(extra<times[-1])]
        extra=extra[~np.any(np.isclose(extra[:,None],grid[None,:],rtol=2e-13,atol=1e-16),axis=1)]
        grid=np.unique(np.r_[grid,extra])
    ids=np.searchsorted(grid,times)
    if not np.array_equal(grid[ids],times):raise ValueError('Lost exact observation times')
    return grid,ids

def l1_weights(t,n,alpha):
    dt=np.diff(t[:n+1])
    if abs(alpha-1)<1e-13:
        w=np.zeros(n);w[-1]=1/dt[-1];return w
    if not 0 < alpha <= 1:raise ValueError('Invalid Caputo order')
    left=t[n]-t[:n];ratio=np.clip(dt/left,0,1);power=1-alpha
    diff=np.ones(n);m=ratio<1;diff[m]=-np.expm1(power*np.log1p(-ratio[m]))
    return left**power*diff/(dt*gamma(2-alpha))

def manufactured_errors(obs,substeps):
    t,_=time_grid(obs,substeps);out={}
    for a in (.3,.63,.95):
        v=l1_weights(t,len(t)-1,a)@np.diff(t*t);ex=2/gamma(3-a)*t[-1]**(2-a)
        out[str(a)]=float(abs(v-ex)/abs(ex))
    return out

class PoroelasticSolver(ABC):
    def __init__(self,known,grid=101,rtol=1e-8,max_seconds=1800):
        n=int(grid)
        if n<21 or 100%(n-1):raise ValueError('Use nested 21,51,101 node grids; production default is 101')
        step=100//(n-1);ii=np.arange(0,101,step)
        self.ids=(ii[:,None]*101+ii[None,:]).ravel();self.n=n;self.N=n*n;N=self.N
        self.known=known;self.sc=np.asarray(known['scales'],float);self.rtol=float(rtol);self.max_seconds=float(max_seconds)
        self.L=float(known['x_m'][-1]-known['x_m'][0])
        if not np.isclose(self.L,500) or not np.isclose(known['y_m'][-1]-known['y_m'][0],500):raise ValueError('Only uploaded 500m square geometry is supported')
        self.geom=Geometry(n-1,n-1,lx=self.L,ly=self.L,q_total=float(known['source_total_m2_s']))
        g=self.geom;r=g.ref
        def matrix(local,rows,cols,shape):
            a=sp.coo_matrix((np.tile(local.ravel(),g.ne),(rows,cols)),shape=shape).tocsr();a.sum_duplicates();return a
        self.Kl=matrix(r['Kl'],g.rows_k,g.cols_k,(g.nu,g.nu))
        self.Kg=matrix(r['Kg'],g.rows_k,g.cols_k,(g.nu,g.nu))
        self.B=matrix(r['B'],g.rows_b,g.cols_b,(g.nu,g.np))
        self.M0=matrix(r['M'],g.rows_p,g.cols_p,(N,N))/self.L**2
        self.H=matrix(r['H'],g.rows_p,g.cols_p,(N,N))
        self.rho_b=float(known['rho_b']);self.rho_f=float(known['rho_f']);self.gravity=np.asarray(known['gravity'],float)
        fel=self.rho_b*(self.gravity[0]*r['Fbx']+self.gravity[1]*r['Fby'])
        self.fu=np.zeros(g.nu);np.add.at(self.fu,g.udofs.ravel(),np.tile(fel,g.ne))
        self.masks={}
        for key in ('p_mask','u1_mask','u2_mask'):
            src=np.flatnonzero(known[key]);yy=np.rint(src//101/step).astype(int);xx=np.rint(src%101/step).astype(int)
            mask=np.zeros((n,n),bool);mask[yy,xx]=True;self.masks[key]=mask
        self.pmask=self.masks['p_mask'].ravel();self.pfree=np.flatnonzero(~self.pmask)
        uf=np.zeros((N,2),bool);uf[:,0]=self.masks['u1_mask'].ravel();uf[:,1]=self.masks['u2_mask'].ravel()
        self.umask=uf.ravel();self.ufree=np.flatnonzero(~self.umask)
        if not np.array_equal(self.ufree,g.u_free):raise ValueError('Mechanical DOF map differs from generator')
        self.pbc=np.zeros(N);self.pbc[self.pmask]=float(known['p_boundary_positive'])

        self.qphysical=np.zeros(N)
        src=np.asarray(known['q_nodal_m2_s'],float);yy,xx=np.nonzero(src.reshape(101,101))
        np.add.at(self.qphysical,(np.rint(yy/step).astype(int)*n+np.rint(xx/step).astype(int)),src[yy*101+xx])
        if not np.isclose(self.qphysical.sum(),known['source_total_m2_s'],rtol=1e-13):raise ValueError('Well total changed')
        self.q=self.qphysical/(float(known['q_scale'])*self.L**2)
        self.p0=known['initial_normalized'][0,self.ids].copy()
        self.u0_observed_physical=np.stack([known['initial_absolute'][1,self.ids],known['initial_absolute'][2,self.ids]],axis=1).ravel()
        self.uscale=np.tile(self.sc[1:],N);self.usf=self.uscale[self.ufree]
        self.internal_theta=np.asarray(known.get('internal_theta',[]),float)

        self.Wx=self.B[0::2].T/self.L;self.Wy=self.B[1::2].T/self.L
        h=1/(n-1);I=sp.eye(n,format='csr');d1=sp.csr_matrix(_classic_line_matrix(n,h,1,True,True))
        self.dx=sp.kron(I,d1,format='csr');self.dy=sp.kron(d1,I,format='csr')
        self.cache=OrderedDict()

    @abstractmethod
    def mechanics(self, parameters):
        raise NotImplementedError

    def modulus(self,p):
        C=p['alpha_B']*self.sc[0]*self.L/(p['g']*self.sc[1])
        if not np.isfinite(C) or C<=0:raise NumericalFailure('Invalid constrained modulus')
        return float(C)

    def initial_equilibrium(self,p):
        C=self.modulus(p);lu,B,K,bf=self.mechanics(p)
        rhs=(bf+p['alpha_B']*self.sc[0]*(B@self.p0))/C
        u=lu.solve(rhs);obs=self.u0_observed_physical[self.ufree]
        errors=[]
        for field in (0,1):
            mask=(self.ufree%2)==field
            errors.append(float(np.linalg.norm(u[mask]-obs[mask])/max(np.linalg.norm(obs[mask]),1e-30)))
        alg=float(np.linalg.norm(K@u-rhs)/max(np.linalg.norm(rhs),1e-30))
        observed_force_residual=float(np.linalg.norm(K@obs-rhs)/max(np.linalg.norm(rhs),1e-30))
        return u,dict(initial_mechanics_field_relL2=errors,
            initial_mechanics_relative_mismatch=float(np.sqrt(np.mean(np.asarray(errors)**2))),
            initial_observed_force_relative_residual=observed_force_residual,
            initial_computed_equilibrium_residual=alg,constrained_modulus_pa=C)

    def pressure_parts(self,p):
        gh=self.rho_f*self.gravity*self.L/self.sc[0]
        Kx,Gx=flux_parts(self.n,0,p['gammax'],p['qx'],gh[0]);Ky,Gy=flux_parts(self.n,1,p['gammay'],p['qy'],gh[1])
        Ap=(p['dx']*Kx+p['dy']*Ky).tocsr();fg=p['dx']*Gx+p['dy']*Gy
        if p.get('cxx',0):
            K,G=flux_parts(self.n,0,2.,.5,gh[0]);Ap-=p['cxx']*K;fg-=p['cxx']*G
        if p.get('cyy',0):
            K,G=flux_parts(self.n,1,2.,.5,gh[1]);Ap-=p['cyy']*K;fg-=p['cyy']*G
        for axis,key,order in [(0,'vxb','betax'),(1,'vyb','betay')]:
            if p.get(key,0):
                F=sp.csr_matrix(beta_left_matrix(self.n,1/(self.n-1),p[order],True));I=sp.eye(self.n,format='csr')
                F2=sp.kron(I,F,format='csr') if axis==0 else sp.kron(F,I,format='csr')
                Ap+=p[key]*(self.M0@F2)
        if p.get('cx',0):Ap+=p['cx']*(self.M0@self.dx)
        if p.get('cy',0):Ap+=p['cy']*(self.M0@self.dy)
        if p.get('reaction',0):Ap+=p['reaction']*self.M0
        return Ap.tocsr(),fg

    def forward(self,p,obs,substeps=2,direct_reference=False):
        start=time.perf_counter();n=self.n;N=self.N;lu,B,K,bf=self.mechanics(p);C=self.modulus(p)
        u0,ic=self.initial_equilibrium(p)


        Sref=float(self.known['time_scale_s'])*float(self.known['q_scale'])/(p['cq']*self.sc[0])
        if not np.isfinite(Sref) or Sref<=0:raise NumericalFailure('Invalid positive reference storage')
        tau=(1/(n-1))**2/(12*C*Sref)
        M=(self.M0+tau*self.H).tocsr()

        Dx=p['bxt']*self.Wx+p.get('b1y',0)*self.Wy
        Dy=p['byt']*self.Wy+p.get('b2x',0)*self.Wx
        coox=Dx.tocoo();cooy=Dy.tocoo()
        D=sp.coo_matrix((np.r_[coox.data,cooy.data],(np.r_[coox.row,cooy.row],np.r_[2*coox.col,2*cooy.col+1])),shape=(N,2*N)).tocsr()[:,self.ufree]

        Dphys=(D@sp.diags(1/self.usf)).tocsr()
        Ap,fg=self.pressure_parts(p);aP=p['alpha_B']*self.sc[0]/C
        ubase=lu.solve(bf/C)
        grid,indices=time_grid(obs,substeps,self.internal_theta);nt=len(grid)
        output_at={int(v):j for j,v in enumerate(indices)}
        ph=np.empty((nt,N));ph[0]=self.p0;out=np.empty((3,len(obs),N));out[0,0]=self.p0
        def put_u(u,j):
            full=np.zeros(2*N);full[self.ufree]=u/self.usf
            out[1,j]=full[0::2];out[2,j]=full[1::2]
        put_u(u0,0)
        pnow=self.p0.copy();u_prev=u0.copy();lift=self.pbc
        ulift=ubase+aP*lu.solve(B@lift);Mlift=M@lift;Aplift=Ap@lift
        Mff=M[self.pfree][:,self.pfree].tocsr();Apff=Ap[self.pfree][:,self.pfree].tocsr()
        pre_cache=OrderedDict();resids=[];mres=[];iters=[]
        for j in range(1,nt):
            if time.perf_counter()-start>self.max_seconds:raise NumericalFailure('Candidate walltime limit')
            w=l1_weights(grid,j,p['alpha']);an=float(w[-1]);dt=float(grid[j]-grid[j-1]);eta=1/(an*dt)
            hist=w[:-1]@np.diff(ph[:j],axis=0) if j>1 else np.zeros(N)
            rhs=(p['cq']*self.q+fg-M@hist)/an+M@pnow+eta*(Dphys@u_prev)
            rf=(rhs-Mlift-Aplift/an-eta*(Dphys@ulift))[self.pfree]
            def mv(v):
                vfull=np.zeros(N);vfull[self.pfree]=v
                u=aP*lu.solve(B@vfull)
                return Mff@v+Apff@v/an+eta*(Dphys@u)[self.pfree]
            key=(round(an,11),round(eta,11));count=[0]
            if key not in pre_cache:
                diag=max(0.,.5*p['g']*(abs(p['bxt'])+abs(p['byt'])*self.sc[1]/self.sc[2])*eta)
                pre=(Mff+diag*self.M0[self.pfree][:,self.pfree]+Apff/an).tocsc()
                try:plu=sla.spilu(pre,drop_tol=1e-4,fill_factor=6.)
                except RuntimeError as exc:raise NumericalFailure('Pressure ILU: '+str(exc)) from exc
                pre_cache[key]=plu
                if len(pre_cache)>2:pre_cache.popitem(last=False)
            plu=pre_cache[key]
            op=sla.LinearOperator((len(self.pfree),len(self.pfree)),matvec=mv,dtype=float)
            prec=sla.LinearOperator(op.shape,matvec=plu.solve,dtype=float)
            sol,flag=sla.gmres(op,rf,x0=pnow[self.pfree],M=prec,rtol=self.rtol,atol=1e-14,
                restart=80,maxiter=30,callback=lambda _:count.__setitem__(0,count[0]+1),callback_type='pr_norm')
            residual=float(np.linalg.norm(mv(sol)-rf)/max(np.linalg.norm(rf),1e-30))
            if not np.isfinite(residual) or residual>max(1e-7,10*self.rtol):raise NumericalFailure(f'Pressure residual {residual}; flag={flag}')
            pnow=lift.copy();pnow[self.pfree]=sol;u=ubase+aP*lu.solve(B@pnow)
            urhs=bf/C+aP*(B@pnow);mr=float(np.linalg.norm(K@u-urhs)/max(np.linalg.norm(urhs),1e-30))
            if not np.isfinite(mr) or mr>1e-8 or not np.isfinite(pnow).all() or not np.isfinite(u).all():raise NumericalFailure(f'Mechanics residual {mr}')
            ph[j]=pnow;u_prev=u;resids.append(residual);mres.append(mr);iters.append(count[0])
            if j in output_at:
                idx=output_at[j];out[0,idx]=pnow;put_u(u,idx)
        return out,dict(**ic,elapsed_seconds=time.perf_counter()-start,nxy=n,time_states=nt,substeps=int(substeps),
            max_true_pressure_residual=max(resids,default=0.),max_true_mechanics_residual=max(mres,default=0.),
            mean_gmres_iterations=float(np.mean(iters)) if iters else 0.,future_observations_used=False,
            absolute_state=True,initial_pressure_exact=bool(np.array_equal(out[0,0],self.p0)),
            initial_displacement_computed_from_equilibrium=True,initial_observed_displacement_overwritten=False,
            initial_compensation=False,alpha_B=float(p['alpha_B']),reference_storage_pa_inv=Sref,
            Q1_stabilization_tau_dimensionless=tau,body_force_density_N_m3=(self.rho_b*self.gravity).tolist(),
            pressure_gravity_from_initial_field=False,well_total_m2_s=float(self.qphysical.sum()),
            spatial_discretization='stabilized consistent-mass Q1-Q1, generator-matched integer limit')
