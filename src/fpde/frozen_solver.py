# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

from pathlib import Path
import json,time
import numpy as np
from true_u0 import TruthU0Solver
from frozen_candidate import parse

FIELD_ORDER = np.asarray(['Ux','Uy','P'])

def run_forward(root:Path,output_path:Path,*,candidate_path=None,known_path=None)->dict:
    root=Path(root);output_path=Path(output_path)
    model_path=root/'frozen_candidate.json' if candidate_path is None else Path(candidate_path)
    known_path=root/'known_inputs.npz' if known_path is None else Path(known_path)
    candidate=json.loads(model_path.read_text())
    with np.load(known_path,allow_pickle=False) as z:
        known={key:np.array(z[key]) for key in z.files}
    parsed=parse(candidate)
    start=time.perf_counter()
    solver=TruthU0Solver(known,grid=101,rtol=1e-9,max_seconds=172800.0)
    pred_norm,info=solver.forward(parsed,known['theta'],substeps=16)
    if pred_norm.shape!=(3,49,10201):raise RuntimeError('Unexpected FPDE prediction shape')
    if not np.array_equal(pred_norm[:,0],known['initial_normalized']):
        raise RuntimeError('FPDE initial-normalized-state mismatch')
    physical=pred_norm*known['scales'][:,None,None]
    physical[:,0]=known['initial_absolute']
    if not np.array_equal(physical[:,0],known['initial_absolute']):
        raise RuntimeError('FPDE absolute initial-state mismatch')


    converted=physical[[1,2,0]].transpose(1,0,2)
    x,y=known['x_m'],known['y_m']
    coords=np.stack(np.meshgrid(x,y),axis=-1).reshape(-1,2)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,prediction_si=converted,field_order=FIELD_ORDER,
                        times_s=known['times_s'],node_xy_m=coords,
                        initial_truth_used=np.asarray(True),
                        n_train=known['n_train'],substeps=np.asarray(16))
    return dict(method='frozen_fpde_round4_true_U0',complete=True,output_states=49,
               solver_grid=101,substeps=16,field_order=FIELD_ORDER.tolist(),
               initial_three_fields_exact=True,initial_mechanical_projection=False,
               true_state_resets=False,future_truth_used=False,
               production_training_modules_imported=False,model_parameters_optimized=False,
               fixed_Biot_added_again=False,no_clipping=True,
               runtime_seconds=time.perf_counter()-start,solver_info=info)
