# *****
# Copyright Guodong Ren
# University of Southern California
# Contact: guodongr@usc.edu
# *****
# This notice covers release preparation and modifications, 2026-11-24.

import numpy as np
from solver_cpu import CPUSolver

class TruthU0Solver(CPUSolver):
    

    def initial_equilibrium(self, p):


        u_equil, eq = super().initial_equilibrium(p)
        u_truth = self.u0_observed_physical[self.ufree].copy()

        C = self.modulus(p)
        lu, B, K, bf = self.mechanics(p)
        rhs = bf / C + p['alpha_B'] * self.sc[0] * (B @ self.p0) / C
        truth_res = float(np.linalg.norm(K @ u_truth - rhs) / max(np.linalg.norm(rhs), 1e-30))



        info = dict(
            initial_mechanics_field_relL2=[0.0, 0.0],
            initial_mechanics_relative_mismatch=0.0,
            initial_observed_force_relative_residual=truth_res,
            initial_computed_equilibrium_residual=truth_res,
            constrained_modulus_pa=float(eq['constrained_modulus_pa']),
            truth_U0_seeded=True,
            candidate_equilibrium_initial_field_relL2=list(eq['initial_mechanics_field_relL2']),
            candidate_equilibrium_initial_relative_mismatch=float(eq['initial_mechanics_relative_mismatch']),
            candidate_equilibrium_force_residual=float(eq['initial_computed_equilibrium_residual']),
            truth_U0_force_residual=truth_res,
        )
        return u_truth, info

    def forward(self, p, obs, substeps=2, direct_reference=False):
        pred, info = super().forward(p, obs, substeps, direct_reference)





        truth0 = self.known['initial_normalized'][:, self.ids]
        if not np.array_equal(pred[0, 0], truth0[0]):
            raise RuntimeError('P(t0) is not exactly the stored truth IC')
        if not np.array_equal(pred[1, 0], truth0[1]) or not np.array_equal(pred[2, 0], truth0[2]):
            raise RuntimeError('Injected Ux/Uy(t0) do not exactly reproduce the stored truth IC')
        pred[1, 0] = truth0[1]
        pred[2, 0] = truth0[2]

        info.update(
            initial_displacement_computed_from_equilibrium=False,
            initial_displacement_truth_seeded=True,
            initial_observed_displacement_overwritten=True,
            initial_compensation=False,
            initial_truth_exact=True,
            initialization='P0, Ux0, Uy0 are observed truth; t>0 is candidate forward solution',
        )
        return pred, info
