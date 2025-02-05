import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))  # current file's directory
root_dir = os.path.abspath(os.path.join(script_dir, '../../src/tops/ps_models'))
sys.path.append(root_dir)

if __name__ == '__main__':

  # Load model
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

  # Simulate next step
  sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)
  result = sol.step()

  # Real values of bus voltage and line currents
  x = sol.v
  i_bus = ps.y_bus_lf @ x  
  x_Rv = np.real(x)     # real part of the voltage
  x_Iv = np.imag(x)     # imaginary part of the voltage
  x_Ri = np.real(i_bus) # real part of the current
  x_Ii = np.imag(i_bus) # imaginary part of the current
  X    = np.hstack([x_Rv, x_Iv]) # vector of all the measures

  # Noise
  mu_v  = 0*np.ones(ps.n_bus)
  sigma_v  = 0.01
  mu_i  = 0*np.ones(ps.n_bus)
  sigma_i  = 1e-2*0.01
  R_Rv  = sigma_v*np.eye(ps.n_bus)  # Covariance on the measure on the real part of v
  R_Iv  = sigma_v*np.eye(ps.n_bus)  # Covariance on the measure on the imaginary part of v
  R_v   = np.block([[R_Rv, np.zeros_like(R_Rv)], [np.zeros_like(R_Iv), R_Iv]])
  R_Ri  = sigma_i*np.eye(ps.n_bus)  # Covariance on the measure on the real part of i
  R_Ii  = sigma_i*np.eye(ps.n_bus)  # Covariance on the measure on the imaginary part of i
  R_i   = np.block([[R_Ri, np.zeros_like(R_Ri)], [np.zeros_like(R_Ii), R_Ii]])
  R     = np.block([[R_v, np.zeros_like(R_v)], [np.zeros_like(R_i), R_i]]) # Covariance matrix of all the measures

  epsilon_Rv  = np.random.multivariate_normal(mean=mu_v, cov=R_Rv, size=1)
  epsilon_Iv  = np.random.multivariate_normal(mean=mu_v, cov=R_Iv, size=1)
  epsilon_Ri  = np.random.multivariate_normal(mean=mu_i, cov=R_Ri, size=1)
  epsilon_Ii  = np.random.multivariate_normal(mean=mu_i, cov=R_Ii, size=1)
  epsilon     = np.hstack([epsilon_Rv, epsilon_Iv, epsilon_Ri, epsilon_Ii])
  
  # Model of the sensor
  H_Rv    = np.eye(ps.n_bus) # Identity matrix
  H_Iv    = np.eye(ps.n_bus) # Identity matrix
  H_v     = np.block([[H_Rv, np.zeros([ps.n_bus,ps.n_bus])], [np.zeros([ps.n_bus,ps.n_bus]), H_Iv]])
  H_RiRv  = np.real(ps.y_bus_lf)
  H_RiIv  = -np.imag(ps.y_bus_lf)
  H_IiRv  = np.imag(ps.y_bus_lf)
  H_IiIv  = np.real(ps.y_bus_lf)
  H_i     = np.block([[H_RiRv, H_RiIv], [H_IiRv, H_IiIv]])
  H       = np.vstack([H_v, H_i]) # Measures angles and magnitudes

  # Measures
  z = H @ X + epsilon
  z_v = z[0][0:ps.n_bus] + 1j*z[0][ps.n_bus:2*ps.n_bus]
  z_i = z[0][2*ps.n_bus:3*ps.n_bus] + 1j*z[0][3*ps.n_bus:4*ps.n_bus]

  # Estimate the state
  G = H.T @ np.linalg.inv(R) @ H
  x_hat = np.linalg.inv(G) @ H.T @ np.linalg.inv(R) @ z.T
  x_hat_comb = x_hat[0:ps.n_bus] + 1j*x_hat[ps.n_bus:2*ps.n_bus]

  G_vmi = H_i.T @ np.linalg.inv(R_i) @ H_i # estimate v measuring i
  x_hat_vmi = np.linalg.inv(G_vmi) @ H_i.T @ np.linalg.inv(R_i) @ z[0][2*ps.n_bus:4*ps.n_bus].T
  x_hat_comb_vmi = x_hat_vmi[0:ps.n_bus] + 1j*x_hat_vmi[ps.n_bus:2*ps.n_bus]

  # Plot
  names = list(range(1, ps.n_bus+1))
  plt.figure()
  plt.plot(names, np.abs(x), 'o', label='Real values') 
  plt.plot(names, np.abs(x_hat_comb), 'x', label='Estimated values')
  plt.plot(names, np.abs(z_v), 'v', label='v measure from v')     # v measuring v
  plt.plot(names, np.abs(x_hat_comb_vmi), '1',  label='v measure from i') # v measuring i
  plt.xlabel('Bus number')  
  plt.ylabel('Mag [-]')
  plt.legend()
  
  plt.figure()
  plt.plot(names, np.arctan2(np.imag(x),np.real(x)), 'o', label='Real values') 
  plt.plot(names, np.arctan2(np.imag(x_hat_comb),np.real(x_hat_comb)), 'x', label='Estimated values')
  plt.plot(names, np.arctan2(np.imag(z_v),np.real(z_v)), 'v', label='v measure from v')     # v measuring v
  plt.plot(names, np.arctan2(np.imag(x_hat_comb_vmi),np.real(x_hat_comb_vmi)), '1',  label='v measure from i') # v measuring i
  plt.xlabel('Bus number')  
  plt.ylabel('Angle [-]')
  plt.legend()


  plt.show()