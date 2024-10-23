import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np

# Load model
import tops.ps_models.ieee39 as model_data

model = model_data.load()

# Power system model
ps = dps.PowerSystemModel(model=model)
ps.init_dyn_sim() # initialize power flow on the network
print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

t_end = 50 # simulation time
t_0 = time.time()
x_0 = ps.x_0.copy() # set the initial state as the one computes before

# Solver
sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

t = 0
res = defaultdict(list) # store the results

# sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0] # get the bus index of the generator where the short circuit occurs
event_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][1] # get the bus index of the generator where the short circuit occurs
all_load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'] # index of all the buses

# ps.gen['GEN'] -> access all the generators in the network
# ps.gen['GEN'].bus_idx_red['terminal'][0] -> get the bus index of the generator where the short circuit occurs
s_const = ps.loads['Load'].par['P'] + 1j * ps.loads['Load'].par['Q'] # "old" apparent power
v_old = ps.v_0[all_load_bus_idx]
y_old = abs(v_old)**2*np.conj(s_const) # admittance of the load
        
while t < t_end:
  sys.stdout.write("\r%d%%" % (t/(t_end)*100)) # print the percentage of the simulation completed

  # Implement the short circuit in the bus where the generator is connected
  #if event:
  #  s_const[idx_event] += 1e2

  # Simulate next step
  result = sol.step() # integrate the system one step
  # Extract the information from the solution
  x = sol.y # state variables
  v = sol.v # complex node voltage
  t = sol.t

  # Constant power loads: update the modified admittance matrix
  v_load = v[all_load_bus_idx]
  y_new = abs(v_load)**2*np.conj(s_const) # new admittance of the load
  ps.y_bus_red_mod[(all_load_bus_idx,) * 2] = y_new - y_old

  dx = ps.ode_fun(0, ps.x_0) # compute the derivative of the state variables (in case they are needed)

  # Store result
  res['t'].append(t)
  res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy()) # extract the speed of the generators
  

H = ps.gen['GEN'].par['H'] # Inertia of the generators
COI = res['gen_speed']@H/np.sum(H)

print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

plt.figure(1)
plt.plot(res['t'], res['gen_speed'])
plt.xlabel('Time [s]')
plt.ylabel('Gen. speed')
plt.legend(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'])

plt.figure(2)
plt.plot(res['t'], COI)
plt.xlabel('Time [s]')
plt.ylabel('COI freq')

plt.show()
