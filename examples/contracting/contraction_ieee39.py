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

t_end = 30 # simulation time
t_event = 1 # time of the load step occurance
t_0 = time.time()
x_0 = ps.x_0.copy() # set the initial state as the one computes before

# Solver
sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

t = 0
res = defaultdict(list) # store the results

event_load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][1] # bus index of the generator where the P step
all_load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'] # index of all the loads

s_const_old = (ps.loads['Load'].par['P'] + 1j * ps.loads['Load'].par['Q'])/ps.s_n # "old" apparent power
v_old = ps.v_0[all_load_bus_idx]
y_old = np.conj(s_const_old)/abs(v_old)**2 # admittance of the load
v = ps.v_0
        
while t < t_end:
  sys.stdout.write("\r%d%%" % (t/(t_end)*100)) # print the percentage of the simulation completed

  # Implement the short circuit in the bus where the generator is connected
  if t > t_event and t < t_event +5e-3:
   s_const_old[event_load_bus_idx] += 1e2/ps.s_n 
 
  # Simulate next step
  result = sol.step() # integrate the system one step
  # Extract the information from the solution
  x = sol.y # state variables
  v = sol.v # complex node voltage
  t = sol.t

  # Constant power loads: update the modified admittance matrix
  # tmp = ps.y_bus_red_full[3,2]
  # Igen_43 = ps.y_bus_red_full[3,2]*(v[3] - v[2])
  # Igen_45 = ps.y_bus_red_full[3,4]*(v[3] - v[4])
  # Igen_4_14 = ps.y_bus_red_full[3,13]*(v[3] - v[13])
  # s_43 = v[3]*np.conj(Igen_43 + Igen_45)
  v_load = v[all_load_bus_idx]
  y_new = np.conj(s_const_old)/abs(v_load)**2 # new admittance of the load
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
