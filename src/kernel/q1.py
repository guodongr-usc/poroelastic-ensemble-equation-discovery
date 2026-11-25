# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****

from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

@dataclass
class Geometry:
    x: np.ndarray
    y: np.ndarray
    coords: np.ndarray
    conn: np.ndarray
    uconn: np.ndarray
    @property
    def nn(self):return len(self.coords)
    @property
    def ne(self):return len(self.conn)


@dataclass
class ElementOperators:
    K_lambda: np.ndarray
    K_G: np.ndarray
    Q: np.ndarray
    M: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray


def build_geometry(x,y):
    x,y=np.asarray(x),np.asarray(y)
    xx,yy=np.meshgrid(x,y,indexing='xy');nx=len(x)
    bl=(np.arange(len(y)-1)[:,None]*nx+np.arange(nx-1)[None,:]).ravel()
    conn=np.column_stack([bl,bl+1,bl+nx+1,bl+nx])
    uconn=np.empty((len(conn),8),int);uconn[:,0::2]=2*conn;uconn[:,1::2]=2*conn+1
    return Geometry(x,y,np.c_[xx.ravel(),yy.ravel()],conn,uconn)


def element_operators(geometry):
    dx,dy=np.diff(geometry.x),np.diff(geometry.y)
    if not np.allclose(dx,dx[0]) or not np.allclose(dy,dy[0]):raise ValueError('Uniform grids only')
    Kl=np.zeros((8,8));Kg=Kl.copy();Q=np.zeros((8,4));M=np.zeros((4,4));Hx=M.copy();Hy=M.copy()
    Dlambda=np.array([[1.,1.,0.],[1.,1.,0.],[0.,0.,0.]])
    DG=np.diag([2.,2.,1.]);gp=1/np.sqrt(3);det=dx[0]*dy[0]/4
    for xi,eta in ((-gp,-gp),(gp,-gp),(gp,gp),(-gp,gp)):
        N=.25*np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])
        nx=.5/dx[0]*np.array([-(1-eta),1-eta,1+eta,-(1+eta)])
        ny=.5/dy[0]*np.array([-(1-xi),-(1+xi),1+xi,1-xi])
        B=np.zeros((3,8));B[0,0::2]=nx;B[1,1::2]=ny;B[2,0::2]=ny;B[2,1::2]=nx
        div=np.empty(8);div[0::2]=nx;div[1::2]=ny
        Kl+=B.T@Dlambda@B*det;Kg+=B.T@DG@B*det;Q+=np.outer(div,N)*det
        M+=np.outer(N,N)*det;Hx+=np.outer(nx,nx)*det;Hy+=np.outer(ny,ny)*det
    return ElementOperators(*(np.broadcast_to(a,(geometry.ne,)+a.shape) for a in (Kl,Kg,Q,M,Hx,Hy)))


def assemble(local,rows,cols,shape):
    rr=np.broadcast_to(rows[:,:,None],local.shape).ravel()
    cc=np.broadcast_to(cols[:,None,:],local.shape).ravel()
    matrix=sp.coo_matrix((local.ravel(),(rr,cc)),shape=shape).tocsr();matrix.sum_duplicates()
    return matrix


def block_mean(values,stride):
    ny,nx=values.shape
    return values.reshape(ny//stride,stride,nx//stride,stride).mean(axis=(1,3))


def nearest_coarse_nodes(coords,geometry):
    ix=abs(geometry.x[None,:]-coords[:,0,None]).argmin(axis=1)
    iy=abs(geometry.y[None,:]-coords[:,1,None]).argmin(axis=1)
    return np.unique(iy*len(geometry.x)+ix).astype(int)


def conservative_project_nodal_load(fine_xy,fine_load,geometry):
    output=np.zeros((geometry.nn,fine_load.shape[1]));nx=len(geometry.x)
    for node in np.flatnonzero(np.any(fine_load!=0,axis=1)):
        x,y=fine_xy[node]
        i=max(0,min(int(np.searchsorted(geometry.x,x,side='right'))-1,nx-2))
        j=max(0,min(int(np.searchsorted(geometry.y,y,side='right'))-1,len(geometry.y)-2))
        rx=(x-geometry.x[i])/(geometry.x[i+1]-geometry.x[i]);ry=(y-geometry.y[j])/(geometry.y[j+1]-geometry.y[j])
        ids=[j*nx+i,j*nx+i+1,(j+1)*nx+i+1,(j+1)*nx+i]
        weights=[(1-rx)*(1-ry),rx*(1-ry),rx*ry,(1-rx)*ry]
        for k,w in zip(ids,weights):output[k]+=w*fine_load[node]
    np.testing.assert_allclose(output.sum(axis=0),fine_load.sum(axis=0),rtol=1e-13,atol=1e-18)
    return output


def pressure_boundary_vector(metadata,geometry,fine_xy,coarse):
    if coarse:
        left=nearest_coarse_nodes(metadata['left_pressure_node_xy_m'],geometry)
        right=nearest_coarse_nodes(metadata['right_pressure_node_xy_m'],geometry)
    else:
        left=metadata['left_pressure_node_ids'];right=metadata['right_pressure_node_ids']
    ids=np.r_[left,right].astype(int)
    values=np.r_[np.full(len(left),float(metadata['common_left_pressure_pa'])),np.full(len(right),float(metadata['common_right_pressure_pa']))]
    if len(np.unique(ids))!=len(ids):raise ValueError('Duplicate pressure constraint')
    order=np.argsort(ids)
    return ids[order],values[order]


def total_load_vectors(geometry,mobility):
    hx=geometry.x[1]-geometry.x[0];hy=geometry.y[1]-geometry.y[0]
    body=np.zeros(2*geometry.nn);gravity=np.zeros(geometry.nn)
    np.add.at(body,geometry.uconn[:,1::2],np.full((geometry.ne,4),-2242.609582*9.81*hx*hy/4))
    local=-9810*hx/2*np.array([-1.,-1.,1.,1.])
    np.add.at(gravity,geometry.conn,np.asarray(mobility)[:,None]*local)
    return body,gravity
