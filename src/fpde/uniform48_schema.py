# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from __future__ import annotations
import ast, json, math, struct, zipfile
from pathlib import Path

DAY_S=86400.0
END_S=730.0*DAY_S
INTERVALS=48
STATES=49
NODES=10201
EXPECTED_SHAPE=(1,STATES,NODES)

def expected_times(): return [i*END_S/INTERVALS for i in range(STATES)]

def chronological_split(times, fraction=.8):
    n=len(times)
    if n!=STATES: raise ValueError(f'Expected {STATES} states including t0, got {n}')
    if not .5<=float(fraction)<=.9: raise ValueError('train_fraction must be in [0.5,0.9]')
    if any(not math.isfinite(float(v)) for v in times): raise ValueError('Nonfinite time metadata')
    exp=expected_times()
    if any(abs(float(a)-b)>1e-6 for a,b in zip(times,exp)):
        raise ValueError('Expected 0..730 days in 48 equal intervals (49 states)')
    k=int(math.floor(n*float(fraction)))
    if not 3<=k<n: raise ValueError('Invalid chronological split')
    return dict(total_snapshots=n,train_count=k,test_count=n-k,train_positive_count=k-1,
                train_indices=list(range(k)),test_indices=list(range(k,n)),fraction_requested=float(fraction),
                split_rule='floor(train_fraction * all_snapshots_including_t0)',
                train_end_s=float(times[k-1]),train_end_days=float(times[k-1]/DAY_S),
                first_test_s=float(times[k]),first_test_days=float(times[k]/DAY_S),
                last_test_s=float(times[-1]),last_test_days=float(times[-1]/DAY_S),
                output_interval_days=730.0/INTERVALS,
                train_physical_duration_fraction=float(times[k-1]/times[-1]))

def read_header(stream):
    if stream.read(6)!=b'\x93NUMPY': raise ValueError('Not an NPY stream')
    ver=tuple(stream.read(2)); nbytes=2 if ver==(1,0) else 4 if ver in ((2,0),(3,0)) else 0
    if not nbytes: raise ValueError(f'Unsupported NPY version {ver}')
    size=int.from_bytes(stream.read(nbytes),'little')
    if size>1000000: raise ValueError('Unreasonably large NPY header')
    raw=stream.read(size)
    if len(raw)!=size: raise IOError('Truncated NPY header')
    h=ast.literal_eval(raw.decode('utf-8' if ver==(3,0) else 'latin1').strip())
    if not isinstance(h,dict) or not {'shape','descr','fortran_order'}<=set(h): raise ValueError('Invalid NPY header')
    return h

def _scalar(z,key):
    with z.open(key+'.npy') as f:
        h=read_header(f)
        if tuple(h['shape'])!=(): raise ValueError(f'{key} must be scalar')
        d=h['descr']; endian='<' if d[0] in ('<','|','=') else '>'
        if d.endswith('f8'): return struct.unpack(endian+'d',f.read(8))[0]
        if d.endswith('i8'): return struct.unpack(endian+'q',f.read(8))[0]
        if d.endswith('b1'): return bool(f.read(1)[0])
        raise ValueError(f'Unsupported scalar dtype {key}: {d}')


def _small_array(z,key):
    with z.open(key+'.npy') as f:
        h=read_header(f);shape=tuple(h['shape']);d=h['descr']
        if len(shape)!=1 or shape[0]>500 or d not in ('<i8','>i8','=i8'):
            raise ValueError('Expected int64 seed array '+key)
        raw=f.read(shape[0]*8)
        if len(raw)!=shape[0]*8:raise IOError('Truncated seed array')
        endian='>' if d[0]=='>' else '<'
        return list(struct.unpack(endian+str(shape[0])+'q',raw))

def _text_scalar(z,key):
    with z.open(key+'.npy') as f:
        h=read_header(f);d=h['descr']
        if tuple(h['shape'])!=() or len(d)<3 or d[1]!='U':
            raise ValueError('Expected scalar Unicode '+key)
        n=int(d[2:])
        if n>100000:raise ValueError('Oversized text metadata')
        return f.read(n*4).decode('utf-32-be' if d[0]=='>' else 'utf-32-le').rstrip('\0')

def inspect_npz(path, train_fraction=.8):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    req=('times_s','times_days','x_m','y_m','node_xy_m','forcing_q_matrix_m2_s','case_source_node_ids',
         'case_source_center_xy_m','nreal','fbm_H','fbm_logk_std','k_ref_md','q_total_m2_s','constant_flow',
         'total_fields_are_direct_active_solves','zero_source_reference_solved','derived_response_fields_stored')
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist())
        missing=[k for k in req if k+'.npy' not in names]
        if missing: raise KeyError('Missing metadata '+str(missing))
        if any(k+'.npy' in names for k in ('P_reference_EA','U1_reference_EA','U2_reference_EA')):
            raise ValueError('Requires DIRECT no-reference archive')
        for key in ('P_EA','U1_EA','U2_EA'):
            with z.open(key+'.npy') as f:
                h=read_header(f)
            if tuple(h['shape'])!=EXPECTED_SHAPE or h['fortran_order']:
                raise ValueError(f'{key} header {h}; expected C-order {EXPECTED_SHAPE}')
            if h['descr'] not in ('<f8','<f4','>f8','>f4','=f8','=f4'):
                raise ValueError('Unexpected field dtype '+str(h['descr']))
        with z.open('times_s.npy') as f:
            h=read_header(f)
            if tuple(h['shape'])!=(STATES,) or h['descr'] not in ('<f8','>f8','=f8'):
                raise ValueError('Expected float64 times_s[49]')
            endian='<' if h['descr'][0] in ('<','=') else '>'
            raw=f.read(STATES*8)
            if len(raw)!=STATES*8: raise IOError('Truncated times_s')
            times=struct.unpack(endian+str(STATES)+'d',raw)
        metadata={k:_scalar(z,k) for k in ('nreal','fbm_H','fbm_logk_std','k_ref_md','q_total_m2_s','constant_flow',
                                                        'total_fields_are_direct_active_solves','zero_source_reference_solved','derived_response_fields_stored')}
    nreal=int(metadata['nreal'])
    if not 1 <= nreal <= 500: raise ValueError('Expected 1..500 retained realizations')
    filtered_metadata={}
    if nreal != 500:
        required_filter=('nreal_original','included_seeds','excluded_seeds','filtering_rule_json')
        with zipfile.ZipFile(path) as z:
            missing=[key for key in required_filter if key+'.npy' not in z.namelist()]
            if missing: raise ValueError('Filtered EA lacks provenance: '+str(missing))
            if int(_scalar(z,'nreal_original')) != 500: raise ValueError('Expected original nreal=500')
            included=_small_array(z,'included_seeds');excluded=_small_array(z,'excluded_seeds')
            if len(included)!=nreal or len(excluded)!=500-nreal:
                raise ValueError('Filtered seed counts disagree with nreal')
            if sorted(included+excluded)!=list(range(42,542)):
                raise ValueError('Included/excluded seeds must partition the original seeds 42..541 exactly')
            rule=json.loads(_text_scalar(z,'filtering_rule_json'))
            if rule.get('uses_future_solution_values', True) or rule.get('uses_fit_scores', True):
                raise ValueError('Filter must explicitly exclude future solution values and fit scores')
            filtered_metadata.update(nreal_original=500,included_seeds=included,
                excluded_seeds=excluded,filtering_rule=rule)

    if abs(float(metadata['fbm_H'])-.5)>1e-12 or abs(float(metadata['fbm_logk_std'])-2.)>1e-12 or abs(float(metadata['k_ref_md'])-1.)>1e-12:
        raise ValueError('Expected H=.5, std(logk)=2, k_ref=1mD')
    if not metadata['constant_flow'] or not metadata['total_fields_are_direct_active_solves'] or metadata['zero_source_reference_solved'] or metadata['derived_response_fields_stored']:
        raise ValueError('Expected direct active total fields; no reference/derived response')
    split=chronological_split(times,train_fraction)
    return dict(path=str(path.resolve()),filename=path.name,nreal=nreal,filtered=(nreal!=500),filtering_metadata=filtered_metadata,field_shape=list(EXPECTED_SHAPE),H=.5,logk_std=2.,k_ref_md=1.,
                q_total_m2_s=float(metadata['q_total_m2_s']),uniform_dt_days=730.0/48,
                uniform_time_metadata_verified=True,solution_values_read=False,**split)
