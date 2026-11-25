# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****

from pathlib import Path
import zipfile
import numpy as np
from numpy.lib import format as fmt
from material_sim.base import fbm500_biot_core as b

DATA=Path(__file__).resolve().parents[2]/'data/443EA_CLEAN_ICSTRAIN0.1_P_U1_U2.npz'

def _header(stream):
    version = fmt.read_magic(stream)
    if version == (1, 0):
        return fmt.read_array_header_1_0(stream)
    if version == (2, 0):
        return fmt.read_array_header_2_0(stream)
    raise ValueError(f"Unsupported NPY version: {version}")


def _read_exact(stream, size):
    blocks = []
    remaining = size
    while remaining:
        block = stream.read(min(remaining, 1024 * 1024))
        if not block:
            raise IOError("Truncated NPY prefix")
        if len(block) > remaining:
            raise IOError("Stream returned more than requested prefix bytes")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def metadata():
    with np.load(DATA,allow_pickle=False) as z:
        m={k:z[k].copy() for k in z.files if not k.endswith('_EA') and not k.startswith('member_')}
    assert int(m['nreal'])==443 and float(m['fbm_H'])==.5 and float(m['fbm_logk_std'])==2.
    assert float(m['k_ref_md'])==1. and bool(m['constant_flow'])
    assert bool(m['total_fields_are_direct_active_solves'])
    assert not bool(m['zero_source_reference_solved']) and not bool(m['derived_response_fields_stored'])
    np.testing.assert_array_equal(m['times_s'],np.linspace(0,730*86400,49))
    np.testing.assert_array_equal(m['solver_times_s'][m['output_solver_step_indices']],m['times_s'])
    xy=m['node_xy_m']; x=m['x_m']; y=m['y_m']; nn=len(xy)
    np.testing.assert_array_equal(xy,np.column_stack([a.ravel() for a in np.meshgrid(x,y)]))


    b.K_REF_MD=1.
    c=b.material_from_mean_k(np.exp(m['mean_logk_elem']))
    m.update(mean_lambda_elem_pa=c['lam'],mean_G_elem_pa=c['G'],mean_alpha_elem=c['alpha_mech'],
             mean_S_elem_pa_inv=c['S'],mean_mobility_elem_m2_pa_s=c['mobility'])
    bottom=np.flatnonzero(xy[:,1]==0); left=np.flatnonzero(xy[:,0]==0)
    m['fixed_U1_node_ids']=np.union1d(bottom,left);m['fixed_U2_node_ids']=bottom
    target=np.asarray([165.,200.,235.,265.,300.,335.])
    m['left_pressure_node_ids']=np.asarray([np.flatnonzero((xy[:,0]==0)&(xy[:,1]==v)).item() for v in target])
    m['right_pressure_node_ids']=np.asarray([np.flatnonzero((xy[:,0]==500)&(xy[:,1]==v)).item() for v in target])
    m['common_left_pressure_pa']=np.asarray(1e7);m['common_right_pressure_pa']=np.asarray(1e7)
    m['initial_pressure_pa']=9.81e6+9810*(500-xy[:,1])
    m['case_q_total_m2_s']=np.asarray([float(m['q_total_m2_s'])])
    q=m['forcing_q_matrix_m2_s'][:,0]
    src=np.asarray(m['case_source_node_ids'][0],int)
    np.testing.assert_array_equal(np.flatnonzero(q),src)
    m['case_q_per_source_node_m2_s']=np.asarray([q[src]])
    m['forcing_temporal_profile']=np.r_[0.,np.ones(48)]
    m['tau_log1p_s']=np.log1p(m['times_s'])/np.log1p(m['times_s'][-1])
    m['output_positive_frame_count']=np.asarray(48);m['initial_frame_included']=np.asarray(True)
    m['output_states_are_solver_states']=np.asarray(True);m['temporal_interpolation_used']=np.asarray(False)
    m['internal_state_histories_stored']=np.asarray(False)
    return m


def read_prefix(last=38):
    assert isinstance(last,int) and 0<=last<=48
    values={}
    with zipfile.ZipFile(DATA) as z:
        for h in ('P','U1','U2'):
            with z.open(h+'_EA.npy') as f:
                shape,order,dtype=_header(f)
                assert shape==(1,49,10201) and not order and dtype==np.dtype('float64')
                count=(last+1)*10201*dtype.itemsize
                raw=_read_exact(f,count)
                values[h]=np.frombuffer(raw,dtype=dtype).reshape(1,last+1,10201).copy()
    return values


def free_ids(m):
    return {h:np.setdiff1d(np.arange(10201),fixed) for h,fixed in
        [('P',np.union1d(m['left_pressure_node_ids'],m['right_pressure_node_ids'])),
         ('U1',m['fixed_U1_node_ids']),('U2',m['fixed_U2_node_ids'])]}


def time_slice(m,last=38,internal=False):
    r={k:np.asarray(v).copy() for k,v in m.items()}
    t=m['solver_times_s'] if internal else m['times_s'];t=t[t<=m['times_s'][last]]
    r['times_s']=t;r['forcing_temporal_profile']=np.r_[0.,np.ones(len(t)-1)]
    r['tau_log1p_s']=np.log1p(t)/np.log1p(t[-1])
    return r
