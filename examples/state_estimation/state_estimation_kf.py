import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import tops.modal_analysis as dps_mdl
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))  # current file's directory
root_dir = os.path.abspath(os.path.join(script_dir, '../../src/tops/ps_models'))
sys.path.append(root_dir)

if __name__ == '__main__':

  # Load model
  import gs4 as model_data
  model = model_data.load()

  # Power system model
  ps = dps.PowerSystemModel(model=model)
  ps.init_dyn_sim()
  print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

  t_end = 10
  iter = 100
  x_0 = ps.x_0.copy()

  # Initialize simulation
  t = 0
  res = defaultdict(list)
  t_0 = time.time()

  # Model of the sensor
  H_mag   = np.eye(ps.n_bus) # Identity matrix
  H_angle = np.eye(ps.n_bus) # Identity matrix
  H = np.block([[H_angle, np.zeros_like(H_angle)], [np.zeros_like(H_angle), H_mag]])
  
  # Noise
  R_angle     = 1e3*np.eye(ps.n_bus)  # Identity matrix
  R_mag       = 1e3*np.eye(ps.n_bus)  # Identity matrix
  R_pred      = np.block([[R_angle, np.zeros_like(R_angle)], [np.zeros_like(R_angle), R_mag]])

  Q = 0*np.eye(ps.n_states) # covariance of the noise of the dynamic
  
  error_v = 1e-1
  var_r  = (error_v/3)**2
  R_Rv  = var_r*np.eye(2*ps.n_bus) # covariance of the noise of the measures

  # Initial state
  x_v_angles  = 0*np.ones(ps.n_bus)
  x_v_mag     = 0*np.ones(ps.n_bus)
  X_pred      = np.concatenate((x_v_angles, x_v_mag), axis=0).reshape(-1, 1)
  P_pred      = np.linalg.inv(H.T @ np.linalg.inv(R_pred) @ H)

  for i in range(t_end): 
    #   _   _       _          
    #  | \ | | ___ (_)___  ___ 
    #  |  \| |/ _ \| / __|/ _ \
    #  | |\  | (_) | \__ \  __/
    #  |_| \_|\___/|_|___/\___|

    # Noise on the state
    nu_x  = np.random.multivariate_normal(mean=0, cov=Q, size=1)

    # Noise on the measures
    epsilon_Rv  = np.random.multivariate_normal(mean=0, cov=R_Rv, size=1)
    
    
    ps_lin = dps_mdl.PowerSystemModelLinearization(ps)
    ps_lin.linearize(x0 = x_hat)
    
    A = ps_lin.a # df/dx
    G = np.zeros(ps.n_states, ps.n_states) # df/dnu

    #   ____               _ _      _   _             
    #  |  _ \ _ __ ___  __| (_) ___| |_(_) ___  _ __  
    #  | |_) | '__/ _ \/ _` | |/ __| __| |/ _ \| '_ \ 
    #  |  __/| | |  __/ (_| | | (__| |_| | (_) | | | |
    #  |_|   |_|  \___|\__,_|_|\___|\__|_|\___/|_| |_|
                                                    
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)
    result = sol.step()
    x_hat_pred = sol.y  # x_hat_k+1_pred state of the system
    z = sol.v           # bus voltages
    t = sol.t           # time

    P_pred = A @ P_pred @ A.T + G @ Q @ G.T # covariance of the state

    #   _   _           _       _       
    #  | | | |_ __   __| | __ _| |_ ___ 
    #  | | | | '_ \ / _` |/ _` | __/ _ \
    #  | |_| | |_) | (_| | (_| | ||  __/
    #   \___/| .__/ \__,_|\__,_|\__\___|
    #        |_|                        

    H = # dh/dx
    h = # measure using the state x_hat_pred

    S = H @ P_pred @ H.T + R_pred
    W = P_pred @ H.T @ np.linalg.inv(S)
    x_hat = x_hat_pred + W @ (z - h) # new estimation
    P_hat = P_pred - W @ H @ P_pred           # new covariance

    res['x_hat'].append(x_hat)
    res['P_hat'].append(P_hat)


  
  names = list(range(1, iter+1))
  plt.figure()
  plt.plot(names, v_tmp_angle, 'o')
  plt.plot(names, X_angle_tmp, 'x')
  plt.xlabel('Iteration')
  plt.ylabel('Angle [deg]')

  plt.figure()
  plt.plot(names, v_tmp_mag, 'o')
  plt.plot(names, X_mag_tmp, 'x')
  plt.xlabel('Iteration')
  plt.ylabel('Voltage [pu]')
  plt.show()