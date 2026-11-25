# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****

import numpy as np

def absolute_metrics(truth,prediction,field):
    truth=np.asarray(truth,dtype=float);prediction=np.asarray(prediction,dtype=float)
    if truth.shape!=prediction.shape:raise ValueError('Shapes must match')
    if field not in ('P','U1','U2'):raise ValueError('Unknown field')
    if not np.isfinite(truth).all():raise ValueError('Truth must be finite')
    factor,unit=(1e-6,'MPa') if field=='P' else (1e3,'mm')
    result=dict(R2=None,L2_norm_error_SI=None,L2_norm_error=None,unit=unit,
        sample_count=truth.size,complete_prediction=bool(np.isfinite(prediction).all()))
    if not result['complete_prediction']:return result
    err=float(np.linalg.norm((prediction-truth).ravel()))
    centered=float(np.linalg.norm((truth-truth.mean()).ravel()))
    result.update(R2=float(1-(err/centered)**2) if centered else None,
                  L2_norm_error_SI=err,L2_norm_error=err*factor)
    return result


def score(y, p):
    error = np.asarray(p, float) - np.asarray(y, float)
    sse = float(np.sum(error ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    norm = float(np.linalg.norm(y))
    return dict(R2=1 - sse / sst if sst > 0 else None,
                absolute_L2=float(np.sqrt(sse)),
                relative_L2=float(np.sqrt(sse) / norm) if norm > 0 else None)
