# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from __future__ import annotations
import math

FIELDS=('P','U1','U2')
DEFAULT_WEIGHTS=(.70,.15,.15)

def components(metrics, initial_error, complexity, *, weights=DEFAULT_WEIGHTS,
               ic_weight=.5, penalty=.0005):
    weights=tuple(float(w) for w in weights)
    if len(weights)!=3 or any(not math.isfinite(w) or w<=0 for w in weights):
        raise ValueError('Three finite positive field weights are required')
    if not math.isclose(sum(weights),1.,rel_tol=0.,abs_tol=1e-10):
        raise ValueError('Field weights must sum to one')
    errors=[1.-float(metrics[f+'_R2']) for f in FIELDS]
    if any(not math.isfinite(x) or x < -1e-12 for x in errors):
        raise ValueError('Undefined training R2: do not replace it by an arbitrary value')
    if not math.isfinite(initial_error) or initial_error<0:raise ValueError('Invalid IC error')
    terms={f+'_weighted_SSE_over_SST':w*max(e,0.) for f,w,e in zip(FIELDS,weights,errors)}
    terms.update(forward=sum(terms.values()),initial=float(ic_weight)*initial_error,
                 complexity=float(penalty)*complexity)
    return terms

def total(terms):return terms['forward']+terms['initial']+terms['complexity']
