import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np
from scipy.linalg import block_diag

script_dir = os.path.dirname(os.path.abspath(__file__))  # current file's directory
root_dir = os.path.abspath(os.path.join(script_dir, '../../src/tops/ps_models'))
sys.path.append(root_dir)

if __name__ == '__main__':

  #   ____       _
  #  / ___|  ___| |_   _   _ _ __
  #  \___ \ / _ \ __| | | | | '_ \
  #   ___) |  __/ |_  | |_| | |_) |
  #  |____/ \___|\__|  \__,_| .__/
  #                         |_|

  # Load model
  # import gs4 as model_data
  import k2a as model_data
  model = model_data.load()

  # Power system model
  ps = dps.PowerSystemModel(model=model)
  ps.init_dyn_sim()

  t_end = 10
  iter = 1
  x_0 = ps.x_0.copy()

  # Initialize simulation
  t = 0
  res = defaultdict(list)
  t_0 = time.time()

  # build incidence matrix of the power system
  ps.n_lines = ps.lines['Line'].n_units
  incidence = np.zeros((ps.n_bus, ps.n_lines))
  for i in range(ps.n_lines):
    incidence[ps.lines['Line'].idx_from[i], i] = 1
    incidence[ps.lines['Line'].idx_to[i], i] = -1

  #   ____        _       _   _               ____  _____
  #  / ___|  ___ | |_   _| |_(_) ___  _ __   |  _ \|  ___|
  #  \___ \ / _ \| | | | | __| |/ _ \| '_ \  | |_) | |_
  #   ___) | (_) | | |_| | |_| | (_) | | | | |  __/|  _|
  #  |____/ \___/|_|\__,_|\__|_|\___/|_| |_| |_|   |_|

  sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)
  result = sol.step()

  #   ____            _              _                 
  #  |  _ \ ___  __ _| | __   ____ _| |_   _  ___  ___ 
  #  | |_) / _ \/ _` | | \ \ / / _` | | | | |/ _ \/ __|
  #  |  _ <  __/ (_| | |  \ V / (_| | | |_| |  __/\__ \
  #  |_| \_\___|\__,_|_|   \_/ \__,_|_|\__,_|\___||___/
                                                     
  x = sol.v
  i_bus = ps.y_bus_lf @ x # bus injected current
  i_line = np.diag(ps.lines['Line'].admittance) @ incidence.T @ x # line current
  x_Rv = np.real(x)     # real part of the voltage
  x_Iv = np.imag(x)     # imaginary part of the voltage
  x_Ri_bus = np.real(i_bus) # real part of the bus current
  x_Ii_bus = np.imag(i_bus) # imaginary part of the bus current
  x_Ri_line = np.real(i_line) # real part of the line current
  x_Ii_line = np.imag(i_line) # imaginary part of the line current
  
  X    = np.hstack([x_Rv, x_Iv]) # vector of real states

  #   _   _       _          
  #  | \ | | ___ (_)___  ___ 
  #  |  \| |/ _ \| / __|/ _ \
  #  | |\  | (_) | \__ \  __/
  #  |_| \_|\___/|_|___/\___|
                           
  mu_v  = 0*np.ones(ps.n_bus)
  error_v = 1e-6
  error_i_bus = 1e-6
  error_i_line = 1e-6
  var_r  = (error_v/3)**2
  mu_i_bus  = 0*np.ones(ps.n_bus)
  var_i_bus  = (error_i_bus/3)**2
  R_Rv  = var_r*np.eye(ps.n_bus)  # Covariance on the measure on the real part of v
  R_Iv  = var_r*np.eye(ps.n_bus)  # Covariance on the measure on the imaginary part of v
  R_v   = np.block([[R_Rv, np.zeros_like(R_Rv)], [np.zeros_like(R_Iv), R_Iv]])
  R_Ri_bus  = var_i_bus*np.eye(ps.n_bus)  # Covariance on the measure on the real part of i
  R_Ii_bus  = var_i_bus*np.eye(ps.n_bus)  # Covariance on the measure on the imaginary part of i
  R_i_bus   = np.block([[R_Ri_bus, np.zeros_like(R_Ri_bus)], [np.zeros_like(R_Ii_bus), R_Ii_bus]])
  R     = block_diag(R_v, R_i_bus) # Covariance matrix of all the measures

  epsilon_Rv  = np.random.multivariate_normal(mean=mu_v, cov=R_Rv, size=1)
  epsilon_Iv  = np.random.multivariate_normal(mean=mu_v, cov=R_Iv, size=1)
  epsilon_Ri_bus  = np.random.multivariate_normal(mean=mu_i_bus, cov=R_Ri_bus, size=1)
  epsilon_Ii_bus  = np.random.multivariate_normal(mean=mu_i_bus, cov=R_Ii_bus, size=1)
  epsilon     = np.hstack([epsilon_Rv, epsilon_Iv, epsilon_Ri_bus, epsilon_Ii_bus])

  #   ____                            
  #  / ___|  ___ _ __  ___  ___  _ __ 
  #  \___ \ / _ \ '_ \/ __|/ _ \| '__|
  #   ___) |  __/ | | \__ \ (_) | |   
  #  |____/ \___|_| |_|___/\___/|_|   

  # Measure v                                  
  H_Rv    = np.eye(ps.n_bus) # Identity matrix
  H_Iv    = np.eye(ps.n_bus) # Identity matrix
  H_v     = np.block([[H_Rv, np.zeros([ps.n_bus,ps.n_bus])], [np.zeros([ps.n_bus,ps.n_bus]), H_Iv]])

  # Measure i_bus
  H_Ri_busRv  = np.real(ps.y_bus_lf)
  H_Ri_busIv  = -np.imag(ps.y_bus_lf)
  H_Ii_busRv  = np.imag(ps.y_bus_lf)
  H_Ii_busIv  = np.real(ps.y_bus_lf)
  H_i_bus     = np.block([[H_Ri_busRv, H_Ri_busIv], [H_Ii_busRv, H_Ii_busIv]])
  
  # Combine measures
  H       = np.vstack([H_v, H_i_bus]) # Measures angles and magnitudes

  #   __  __                               
  #  |  \/  | ___  __ _ ___ _   _ _ __ ___ 
  #  | |\/| |/ _ \/ _` / __| | | | '__/ _ \
  #  | |  | |  __/ (_| \__ \ |_| | | |  __/
  #  |_|  |_|\___|\__,_|___/\__,_|_|  \___|
                                         
  z = H @ X + epsilon
  z_v = z[0][0:ps.n_bus] + 1j*z[0][ps.n_bus:2*ps.n_bus]
  z_i_bus = z[0][2*ps.n_bus:3*ps.n_bus] + 1j*z[0][3*ps.n_bus:4*ps.n_bus]

  #   _____     _   _                 _   _             
  #  | ____|___| |_(_)_ __ ___   __ _| |_(_) ___  _ __  
  #  |  _| / __| __| | '_ ` _ \ / _` | __| |/ _ \| '_ \ 
  #  | |___\__ \ |_| | | | | | | (_| | |_| | (_) | | | |
  #  |_____|___/\__|_|_| |_| |_|\__,_|\__|_|\___/|_| |_|

  # estimate v using all the measures                                                    
  G = H.T @ np.linalg.inv(R) @ H
  x_hat = np.linalg.inv(G) @ H.T @ np.linalg.inv(R) @ z.T
  x_hat_comb = x_hat[0:ps.n_bus] + 1j*x_hat[ps.n_bus:2*ps.n_bus]

  # estimate v measuring only v
  G_v = H_v.T @ np.linalg.inv(R_v) @ H_v
  x_hat_v = np.linalg.inv(G_v) @ H_v.T @ np.linalg.inv(R_v) @ z[0][0:2*ps.n_bus].T
  x_hat_comb_v = x_hat_v[0:ps.n_bus] + 1j*x_hat_v[ps.n_bus:2*ps.n_bus]

  # estimate v measuring only i_bus
  G_vmi_bus = H_i_bus.T @ np.linalg.inv(R_i_bus) @ H_i_bus 
  x_hat_vmi_bus = np.linalg.inv(G_vmi_bus) @ H_i_bus.T @ np.linalg.inv(R_i_bus) @ z[0][2*ps.n_bus:4*ps.n_bus].T
  x_hat_comb_vmi_bus = x_hat_vmi_bus[0:ps.n_bus] + 1j*x_hat_vmi_bus[ps.n_bus:2*ps.n_bus]

  #  ____  _       _
  # |  _ \| | ___ | |_
  # | |_) | |/ _ \| __|
  # |  __/| | (_) | |_
  # |_|   |_|\___/ \__|

  names = list(range(1, ps.n_bus+1))
  plt.figure()
  plt.plot(names, np.abs(x), 'o', label='Real values')
  plt.plot(names, np.abs(x_hat_comb), 'x', label='Estimated values')
  plt.plot(names, np.abs(z_v), 'v', label='v measure from v')     # v measuring v
  plt.plot(names, np.abs(x_hat_comb_vmi_bus), '1',  label='v measure from i_bus') # v measuring i
  plt.xlabel('Bus number')
  plt.ylabel('Mag [-]')
  plt.legend()

  plt.figure()
  plt.plot(names, np.arctan2(np.imag(x),np.real(x)), 'o', label='Real values')
  plt.plot(names, np.arctan2(np.imag(x_hat_comb),np.real(x_hat_comb)), 'x', label='Estimated values')
  plt.plot(names, np.arctan2(np.imag(z_v),np.real(z_v)), 'v', label='v measure from v')     # v measuring v
  plt.plot(names, np.arctan2(np.imag(x_hat_comb_vmi_bus),np.real(x_hat_comb_vmi_bus)), '1',  label='v measure from i_bus') # v measuring i
  plt.xlabel('Bus number')
  plt.ylabel('Angle [-]')
  plt.legend()

  plt.figure()
  plt.plot(names, np.real(x), 'o', label='Real values')
  plt.plot(names, np.real(x_hat_comb), 'x', label='Estimated values')
  plt.plot(names, np.real(z_v), 'v', label='v measure from v')     # v measuring v
  plt.plot(names, np.real(x_hat_comb_vmi_bus), '1',  label='v measure from i_bus') # v measuring i
  plt.xlabel('Bus number')
  plt.ylabel('Real [-]')
  plt.legend()

  plt.figure()
  plt.plot(names, np.imag(x), 'o', label='Real values')
  plt.plot(names, np.imag(x_hat_comb), 'x', label='Estimated values')
  plt.plot(names, np.imag(z_v), 'v', label='v measure from v')     # v measuring v
  plt.plot(names, np.imag(x_hat_comb_vmi_bus), '1',  label='v measure from i_bus') # v measuring i
  plt.xlabel('Bus number')
  plt.ylabel('Imag [-]')
  plt.legend()

  plt.show()