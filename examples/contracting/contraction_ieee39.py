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

t_end = 30            # simulation time
t_event = 1             # time of the load step occurance
event_true = True       # boolean to activate
power_unbanlance = 1e2  # power unbalance in the generator bus [MW]
t_0 = time.time()
x_0 = ps.x_0.copy() # set the initial state as the one computes before

# Solver
sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

t = 0
res = defaultdict(list) # store the results

# event_load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'][5] # bus index of the generator where the P step
event_load_bus_idx = 1
all_load_bus_idx = ps.loads['Load'].bus_idx_red['terminal'] # index of all the loads

s_const_old = (ps.loads['Load'].par['P'] + 1j * ps.loads['Load'].par['Q'])/ps.s_n # "old" apparent power
v_old = ps.v_0[all_load_bus_idx]
y_old = np.conj(s_const_old)/abs(v_old)**2 # admittance of the load
v = ps.v_0


while t < t_end:
  sys.stdout.write("\r%d%%" % (t/(t_end)*100)) # print the percentage of the simulation completed

  # Implement the short circuit in the bus where the generator is connected
  if t > t_event and  event_true:
    s_const_old[event_load_bus_idx] += power_unbanlance/ps.s_n
    event_true = False

  # Simulate next step
  result = sol.step() # integrate the system one step
  # Extract the information from the solution
  x = sol.y # state variables
  v = sol.v # complex node voltage
  t = sol.t

  # Constant power loads: update the modified admittance matrix
  v_load = v[all_load_bus_idx]
  y_new = np.conj(s_const_old)/abs(v_load)**2 # new admittance of the load
  ps.y_bus_red_mod[(all_load_bus_idx,) * 2] = y_new - y_old

  # Compute the power of the load 4
  I_4_3  = ps.y_bus_red_full[3,2]*(v[3] - v[2])
  I_4_5  = ps.y_bus_red_full[3,4]*(v[3] - v[4])
  I_4_14 = ps.y_bus_red_full[3,13]*(v[3] - v[13])
  s_4 = v[3]*np.conj(I_4_3 + I_4_5 + I_4_14)

  # Compute power of the generator 4
  I_33_19 = ps.y_bus_red_full[32,18]*(v[32] - v[18]) # current from bus 33 to 19
  V_33 = v[32] # voltage of bus 33
  s_gen_4 = -V_33*np.conj(I_33_19) # apparent power of the generator 4

  dx = ps.ode_fun(0, ps.x_0) # compute the derivative of the state variables (in case they are needed)

  # Store result
  res['t'].append(t)
  res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy()) # extract the speed of the generators
  res['P_load_4'].append(np.real(s_4)*ps.s_n)                    # computed active power of the load 4
  res['P_load_4_setpoint'].append( np.real(s_const_old[event_load_bus_idx]*ps.s_n) ) # extract the apparent power of the load 4
  res['P_loads'].append(np.real(s_const_old)*ps.s_n)
  res['P_gen_4'].append(np.real(s_gen_4)*ps.s_n)            # computed active power of the load 4
  res['P_e'].append(ps.gen['GEN'].P_e(x, v).copy())         # power of the generators
  

H = ps.gen['GEN'].par['H'] # Inertia of the generators
COI = res['gen_speed']@H/np.sum(H)
RoCoF = np.diff(COI)/np.diff(res['t']) # Rate of Change of Frequency
droop_4 = (res['P_gen_4'][-1] - res['P_gen_4'][0])/(res['gen_speed'][4][-1] - res['gen_speed'][4][0]) 
print('Droop of the generator 4: {:.2f}'.format(droop_4))

print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

# Speed of all the generators
plt.figure(1)
plt.plot(res['t'], res['gen_speed'])
plt.xlabel('Time [s]')
plt.ylabel('Gen. speed [pu]')
plt.legend(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10'])
plt.title('Speed of the generators')

# Center of Inertia (COI) frequency
fig, ax1=plt.subplots()
ax1.plot(res['t'], COI, color='b', label='Freq')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('COI freq [pu]', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Center of Inertia (COI) and ROCOF')
ax2 = ax1.twinx()
ax2.plot(res['t'][:-1], RoCoF, color='r', label='RoCoF')
ax2.set_ylabel('RoCoF', color='r')
ax2.tick_params(axis='y', labelcolor='r')
fig.tight_layout()

# Power of the load 4
plt.figure(4)
plt.plot(res['t'], res['P_load_4'])
plt.plot(res['t'], res['P_load_4_setpoint'])
plt.xlabel('Time [s]')
plt.ylabel('p4 [MW]')
plt.legend(['Computed power', 'Set point'])
plt.title('Power of the load 4')

# Power of all the load
plt.figure(5)
plt.plot(res['t'], res['P_loads'])
plt.xlabel('Time [s]')
plt.ylabel('Load active power')
plt.legend(['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 'L15', 'L16', 'L17', 'L18', 'L19' ])
plt.title('Power of all the loads')

# power of generator 4
plt.figure(6)
plt.plot(res['t'], res['P_gen_4'])
plt.axhline(y=ps.gen['GEN'].par['P'][3], color='r', linestyle='--', linewidth=1.5, label='y=0.5')
plt.xlabel('Time [s]')
plt.ylabel('[MW]')
plt.legend(['Computed power', 'Rated power'])
plt.title('Power at generator 4')

# power of the generators
plt.figure(7)
plt.plot(res['t'], res['P_e'])
# plt.axhline(y=ps.gen['GEN'].par['P'][3], color='r', linestyle='--', linewidth=1.5)  # rated power of the generator 4
plt.xlabel('Time [s]')
plt.ylabel('[MW]')
plt.legend(['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10' ])
plt.title('Power at generator ')



plt.show()
