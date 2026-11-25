# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# Governing equations: M. A. Biot (1941), "General Theory of Three-Dimensional
# Consolidation", Journal of Applied Physics 12, 155-164.
# https://doi.org/10.1063/1.1712886
# This module assembles the coupled operators for this study's 2D mesh.

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from . import q1 as core

class LinearFactor:
    
    def __init__(self,matrix,columns,symmetric_row_transform=None):
        self.matrix=matrix.tocsr();self.columns=np.asarray(columns,float)
        if symmetric_row_transform is None:
            scaled=self.matrix.multiply(self.columns[None,:]).tocsr()
            norms=np.asarray(abs(scaled).max(axis=1).toarray()).ravel()
            if np.any(norms<=0) or not np.isfinite(norms).all():raise ValueError('Bad matrix row')
            self.rows=1/norms
            self.scaled=scaled.multiply(self.rows[:,None]).tocsc()
            self.lu=splu(self.scaled)
        else:



            transform=np.asarray(symmetric_row_transform,float)
            diagonal=np.abs(transform*self.matrix.diagonal())
            if np.any(diagonal<=0) or not np.isfinite(diagonal).all():raise ValueError('Bad saddle diagonal')
            self.columns=1/np.sqrt(diagonal);self.rows=transform*self.columns
            self.scaled=self.matrix.multiply(self.columns[None,:]).multiply(self.rows[:,None]).tocsc()
            self.lu=splu(self.scaled,permc_spec='MMD_AT_PLUS_A')

    def solve(self,rhs):
        b=self.rows[:,None]*np.asarray(rhs)
        z=self.lu.solve(np.asfortranarray(b))
        for _ in range(1):
            proposal=z+self.lu.solve(np.asfortranarray(b-self.scaled@z))
            good=np.linalg.norm(b-self.scaled@proposal,axis=0)<np.linalg.norm(b-self.scaled@z,axis=0)
            z[:,good]=proposal[:,good]
        residual=np.linalg.norm(self.scaled@z-b,axis=0)/np.maximum(np.linalg.norm(b,axis=0),1e-300)
        if not np.isfinite(z).all() or residual.max()>1e-7:
            raise np.linalg.LinAlgError(f'Forward residual gate: {residual.max()}')
        return self.columns[:,None]*z,float(residual.max())



@dataclass
class Operators:
    K: object
    Q: object
    C: object
    H: object
    gravity: np.ndarray
    tau: np.ndarray
    coefficients: np.ndarray


class FEM:
    def __init__(self,metadata,stride=1):
        self.metadata=metadata;self.stride=int(stride)
        if self.stride!=stride or 100%self.stride:raise ValueError('Stride must divide 100')
        self.x=np.asarray(metadata['x_m'])[::self.stride]
        self.y=np.asarray(metadata['y_m'])[::self.stride]
        self.geometry=core.build_geometry(self.x,self.y)
        self.nn=self.geometry.nn;self.nu=2*self.nn
        self.times=np.asarray(metadata['times_s'],float)
        self.profile=np.asarray(metadata['forcing_temporal_profile'],float)
        self.conn=self.geometry.conn;self.uconn=self.geometry.uconn
        self.xy=self.geometry.coords
        self.element_xy=self.xy[self.conn].mean(axis=1)
        self.hx=float(self.x[1]-self.x[0]);self.hy=float(self.y[1]-self.y[0])
        fine_xy=np.asarray(metadata['node_xy_m'])
        native_nx=len(metadata['x_m'])
        self.native_node_ids=(np.arange(0,len(metadata['y_m']),self.stride)[:,None]*native_nx+
                              np.arange(0,native_nx,self.stride)[None,:]).ravel()
        fixed1=core.nearest_coarse_nodes(fine_xy[metadata['fixed_U1_node_ids']],self.geometry)
        fixed2=core.nearest_coarse_nodes(fine_xy[metadata['fixed_U2_node_ids']],self.geometry)
        self.fixed_u=np.unique(np.r_[2*fixed1,2*fixed2+1])
        self.free_u=np.setdiff1d(np.arange(self.nu),self.fixed_u)
        self.fixed_p,self.p_boundary=core.pressure_boundary_vector(metadata,self.geometry,fine_xy,self.stride!=1)
        self.free_p=np.setdiff1d(np.arange(self.nn),self.fixed_p)
        self.fixed=np.r_[self.fixed_u,self.nu+self.fixed_p]
        self.free=np.setdiff1d(np.arange(self.nu+self.nn),self.fixed)
        self.free_ids={'P':self.free_p,'U1':np.setdiff1d(np.arange(self.nn),fixed1),
                       'U2':np.setdiff1d(np.arange(self.nn),fixed2)}
        keys=('mean_lambda_elem_pa','mean_G_elem_pa','mean_alpha_elem','mean_S_elem_pa_inv',
              'mean_mobility_elem_m2_pa_s','mean_mobility_elem_m2_pa_s')
        self.base=np.column_stack([core.block_mean(np.asarray(metadata[k]),self.stride).ravel() for k in keys])
        self.local=core.element_operators(self.geometry)
        self.gradient_local=self.local.Hx+self.local.Hy
        self.body,_=core.total_load_vectors(self.geometry,np.ones(self.geometry.ne))
        self.gravity_local=np.tile(-9810*self.hx/2*np.asarray([-1.,-1.,1.,1.]),(self.geometry.ne,1))
        self.p0=np.asarray(metadata['initial_pressure_pa'])[self.native_node_ids].copy()
        self.columns=np.r_[np.ones(self.nu),np.full(self.nn,1e7)]

    def sources(self,case_ids):
        case_ids=np.asarray(case_ids,int)
        if not np.isin(case_ids,self.metadata['case_id']).all():raise ValueError('Unknown case ID')
        fine=np.zeros((len(self.metadata['node_xy_m']),len(case_ids)))
        for j,case_id in enumerate(case_ids):
            index=np.flatnonzero(self.metadata['case_id']==case_id).item()
            fine[np.asarray(self.metadata['case_source_node_ids'][index],int),j]=self.metadata['case_q_per_source_node_m2_s'][index]
        source=core.conservative_project_nodal_load(np.asarray(self.metadata['node_xy_m']),fine,self.geometry)
        if np.any(source[self.fixed_p]):raise ValueError('Projected source lies on pressure constraint')
        return source

    def assemble(self,coefficients):
        c=np.asarray(coefficients,float)
        if c.shape!=self.base.shape or not np.isfinite(c).all() or np.any(c<=0):
            raise ValueError('All six coefficient maps must be positive finite')
        lam,G,alpha,S,mx,my=c.T
        if np.any(alpha>=1):raise ValueError('alpha must stay below 1 in this model family')
        l=self.local
        K=core.assemble(lam[:,None,None]*l.K_lambda+G[:,None,None]*l.K_G,self.uconn,self.uconn,(self.nu,self.nu))
        Q=core.assemble(alpha[:,None,None]*l.Q,self.uconn,self.conn,(self.nu,self.nn))
        tau=self.hx*self.hy/(12*(lam+2*G))
        C=core.assemble(S[:,None,None]*l.M+tau[:,None,None]*self.gradient_local,self.conn,self.conn,(self.nn,self.nn))
        H=core.assemble(mx[:,None,None]*l.Hx+my[:,None,None]*l.Hy,self.conn,self.conn,(self.nn,self.nn))
        gravity=np.zeros(self.nn);np.add.at(gravity,self.conn,(my[:,None]*self.gravity_local))
        return Operators(K,Q,C,H,gravity,tau,c)
