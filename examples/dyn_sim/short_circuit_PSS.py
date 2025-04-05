import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))  # current file's directory
root_dir = os.path.abspath(os.path.join(script_dir, '../../src/tops/ps_models'))
sys.path.append(root_dir)
root_dir = os.path.abspath(os.path.join(script_dir, '../../src/tops'))
sys.path.append(root_dir)

# Load model
import k2a_base_case_with_AVRs_and_GOVs_PSS_HYDRO as model_data
import k2a_base_case_with_AVRs_and_GOVs_PSS_HYDRO_minGain as model_data_minGain

model = model_data.load()
model_minGain = model_data_minGain.load()

# Power system model
ps = dps.PowerSystemModel(model=model)
ps.init_dyn_sim() # initialize power flow on the network
ps_minGain = dps.PowerSystemModel(model=model_minGain)
ps_minGain.init_dyn_sim() # initialize power flow on the network
# print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

t_end = 40 # simulation time
t_0 = time.time()
x_0 = ps.x_0.copy() # set the initial state as the one computes before
x_0_minGain = ps_minGain.x_0.copy() # set the initial state as the one computes before

# Solver
sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)
sol_minGain = dps_sol.ModifiedEulerDAE(ps_minGain.state_derivatives, ps_minGain.solve_algebraic, 0, x_0_minGain, t_end, max_step=5e-3)

t = 0
res = defaultdict(list) # store the results
res_minGain = defaultdict(list) # store the results

sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][1] # get the bus index of the generator where the short circuit occurs
sc_bus_idx_minGain = ps_minGain.gen['GEN'].bus_idx_red['terminal'][1] # get the bus index of the generator where the short circuit occurs
# ps.gen['GEN'] -> access all the generators in the network
# ps.gen['GEN'].bus_idx_red['terminal'][0] -> get the bus index of the generator where the short circuit occurs

while t < t_end:
  sys.stdout.write("\r%d%%" % (t/(t_end)*100)) # print the percentage of the simulation completed

  # Implement the short circuit in the bus where the generator is connected
  if 1 < t < 1.05:
    ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6 # set the admittance of the bus where the short circuit occurs to 1e6
    ps_minGain.y_bus_red_mod[(sc_bus_idx_minGain,) * 2] = 1e6 # set the admittance of the bus where the short circuit occurs to 1e6
  else:
    ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0 # set the admittance of the bus where the short circuit occurs to 0
    ps_minGain.y_bus_red_mod[(sc_bus_idx_minGain,) * 2] = 0 # set the admittance of the bus where the short circuit occurs to 0

  # Simulate next step
  result = sol.step() # integrate the system one step
  # Extract the information from the solution
  x = sol.y # state variables
  v = sol.v # complex node voltage
  t = sol.t

  result_minGain = sol_minGain.step() # integrate the system one step
  # Extract the information from the solution
  x_minGain = sol_minGain.y # state variables
  v_minGain = sol_minGain.v # complex node voltage
  t_minGain = sol_minGain.t

  # dx = ps.ode_fun(0, ps.x_0) # compute the derivative of the state variables (in case they are needed)
  # dx_minGain = ps_minGain.ode_fun(0, ps_minGain.x_0) # compute the derivative of the state variables (in case they are needed)

  # Store result
  res['t'].append(t)
  res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy()) # extract the 

  res_minGain['t'].append(t)
  res_minGain['gen_speed'].append(ps_minGain.gen['GEN'].speed(x_minGain, v_minGain).copy()) # extract the 

print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

# plt.figure()
# plt.plot(res['t'], res['gen_speed'])
# plt.xlabel('Time [s]')
# plt.ylabel('Gen. speed')

# plt.figure()
# plt.plot(res_minGain['t'], res_minGain['gen_speed'])
# plt.xlabel('Time [s]')
# plt.ylabel('Gen. speed')

stacked = np.stack(res['gen_speed'])      # Shape: (3, 3)
stacked_minGain = np.stack(res_minGain['gen_speed'])      # Shape: (3, 3)


output_path = 'assigments/PSS_design/figures/speed_comparison.pdf'
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()  # Makes it easier to index in a 1D way
for i in range(4):
    axes[i].plot(res['t'], stacked[:, i], label='NO PSS')
    axes[i].plot(res_minGain['t'], stacked_minGain[:, i], linestyle='--', color='orange', label='With PSS')
    axes[i].set_title(f'Generator {i+1}')
    axes[i].set_xlabel('Time [s]')  # Add x-axis label
    axes[i].set_ylabel('Speed')     # Add y-axis label
    axes[i].grid(True)

axes[3].legend()  # Add legend to each subplot
plt.tight_layout()
fig.savefig(output_path, format='pdf', bbox_inches='tight')
plt.show()