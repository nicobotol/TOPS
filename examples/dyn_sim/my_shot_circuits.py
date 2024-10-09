import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol

# Load model
import tops.ps_models.sm_ib as model_data

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

sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0] # get the bus index of the generator where the short circuit occurs
# ps.gen['GEN'] -> access all the generators in the network
# ps.gen['GEN'].bus_idx_red['terminal'][0] -> get the bus index of the generator where the short circuit occurs

while t < t_end:
  sys.stdout.write("\r%d%%" % (t/(t_end)*100)) # print the percentage of the simulation completed

  # Implement the short circuit in the bus where the generator is connected
  if 1 < t < 1.05:
    ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 1e6 # set the admittance of the bus where the short circuit occurs to 1e6
  else:
    ps.y_bus_red_mod[(sc_bus_idx,) * 2] = 0 # set the admittance of the bus where the short circuit occurs to 0

  # Simulate next step
  result = sol.step() # integrate the system one step
  # Extract the information from the solution
  x = sol.y # state variables
  v = sol.v # complex node voltage
  t = sol.t

  dx = ps.ode_fun(0, ps.x_0) # compute the derivative of the state variables (in case they are needed)

  # Store result
  res['t'].append(t)
  res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy()) # extract the 

print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

plt.figure()
plt.plot(res['t'], res['gen_speed'])
plt.xlabel('Time [s]')
plt.ylabel('Gen. speed')
plt.show()