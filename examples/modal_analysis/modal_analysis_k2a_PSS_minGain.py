import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt
1
import os
import sys
from scipy.optimize import minimize

script_dir = os.path.dirname(os.path.abspath(__file__))  # current file's directory
root_dir = os.path.abspath(os.path.join(script_dir, '../../src/tops/ps_models'))
sys.path.append(root_dir)


import k2a_base_case_with_AVRs_and_GOVs_PSS_HYDRO as model_data
model = model_data.load()
ps = dps.PowerSystemModel(model=model)
ps.init_dyn_sim()

def obj_function(K):
    return np.linalg.norm(K)


def non_linear_constraint(vars):

    # Assign gain to the PSS
    for i in range(2):
        ps.pss['STAB1'].par[i]['K'] = vars[i]

    # Perform system linearization
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)

    # Find transfer function residuals
    get_eigs=False
    t0=0
    x0=np.array([])
    input_description = "PSS_design"
    output_description = "PSS_design"
    ps_lin.linearize(get_eigs, None, t0, x0, input_description, output_description)

    # ps_lin.linearize()
    ps_lin.eigenvalue_decomposition()

    damplim = 0.05 # Damping threshold that we want to achieve
    mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=damplim)  # Get indices of modes from specified criteria.

    # Selecting mode shapes to print
    critmode = mode_idx[0]

    # Compute residues
    residues = ps_lin.residues(critmode)

    # Compute the residues angle and the compensation angle
    angle = np.angle(residues)
    compensation_angle = np.pi - angle

    # Compute the time constants of the PSS
    alpha = (1 - np.sin(compensation_angle/2))/(1 + np.sin(compensation_angle/2))
    T_3 = 1/(np.sqrt(alpha)*ps_lin.eigs.imag[critmode])
    T_1= alpha*T_3

    # Write the found time constants in the ps model
    for i in range(ps.pss['STAB1'].n_units):
        ps.pss['STAB1'].par[i]['T_1'] = T_1[i,i]
        ps.pss['STAB1'].par[i]['T_3'] = T_3[i,i]

    ps_lin_2 = dps_mdl.PowerSystemModelLinearization(ps)

    ps_lin_2.linearize()
    ps_lin_2.eigenvalue_decomposition()

    min_damp = np.min(ps_lin_2.damping)  # Get indices of modes from specified criteria.

    return min_damp - damplim

constraints = [
    {'type': 'ineq', 'fun': non_linear_constraint},
]

initial_guess = [10, 10]

results = minimize(obj_function, initial_guess, constraints=constraints)

print(results)