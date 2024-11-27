import tops.dynamic as dps
import tops.modal_analysis as dps_mdl
import tops.plotting as dps_plt
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  # current file's directory
root_dir = os.path.abspath(os.path.join(script_dir, '../../src/tops/ps_models'))
sys.path.append(root_dir)


import k2a_base_case_with_AVRs_and_GOVs_PSS_HYDRO as model_data
model = model_data.load()
ps = dps.PowerSystemModel(model=model)
ps.init_dyn_sim()

# Perform system linearization
ps_lin = dps_mdl.PowerSystemModelLinearization(ps)


# Find transfer function residuals
get_eigs=False
t0=0
x0=np.array([])
ps_lin.linearize(get_eigs, None, t0, x0, input_description, output_description)

b = np.zeros(ps.n_states)
b[ps.gen['GEN'].state_idx_global['e_q_t']] = 1/ps.gen['GEN'].par['T_d0_t']

c = np.zeros(ps.n_states)
c[ps.gen['GEN'].state_idx_global['speed']] = 1

# ps_lin.linearize()
ps_lin.eigenvalue_decomposition()

# Plot eigenvalues
dps_plt.plot_eigs(ps_lin.eigs)
plt.show()

print(' ')
print('state description: ')
print(ps.state_desc)
print(' ')
print('Eigenvalues = ')
print(ps_lin.eigs)
print(' ')
print('Speed states: ')
speedstates = ps.gen['GEN'].state_idx_global['speed']
print(speedstates)

# Get mode shape for electromechanical modes
print(' ')
damplim = 0.0
dampl = input('Specify damping threshold (%) : ')
damplim = 0.01* float(dampl)
mode_idx = ps_lin.get_mode_idx(['em'], damp_threshold=damplim)  # Get indices of modes from specified criteria.
print('Mode indices with damping less than', dampl,'% :',mode_idx)
# mode_idx = [14, 12, 10]
rev = ps_lin.rev # Right eigenvectors
# Selecting mode shapes to print
maxmode=0.0
Gennumber=0
tellermax=0
critmode = int(input('Specify mode index for printing generator speed MODE SHAPES : '))
for tellerx in speedstates:
    if abs(rev[tellerx, critmode]) > maxmode:
        maxmode = abs(rev[tellerx, critmode])
        tellermax=tellerx
        Gennumber= int((tellerx-speedstates[0]+6)/6)

print(' ')
print('Selected eigenvalue = ')
print(ps_lin.eigs[critmode])
print(' ')
print('Gen with highest mode shape = ', Gennumber)
print(ps.state_desc[tellermax])
print('Mode shape magnitude = ', maxmode )
# printing eigenvectors
print('Right eigenvectors = ')
gen_number = 0
for tellerx in speedstates:
    gen_number =gen_number + 1
    print('mode ', critmode, ' gen',gen_number,' =', rev[tellerx, critmode] / (maxmode))
# Plotting selected mode shape
# mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], mode_idx)]
mode_shape = rev[np.ix_(ps.gen['GEN'].state_idx_global['speed'], [critmode, critmode+1])]
fig, ax = plt.subplots(1, mode_shape.shape[1], subplot_kw={'projection': 'polar'})
for ax_, ms in zip(ax, mode_shape.T):
    dps_plt.plot_mode_shape(ms, ax=ax_, normalize=True)

plt.show()


