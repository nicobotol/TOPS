import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np



if __name__ == '__main__':

  # Load model
  import tops.ps_models.ieee39 as model_data
  model = model_data.load()

  # Power system model
  ps = dps.PowerSystemModel(model=model)
  ps.init_dyn_sim()
  print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

  t_end = 10
  x_0 = ps.x_0.copy()


  epsilon_angle_mean = 0*np.ones(ps.n_bus)
  epsilon_angle_std = 1e-6*0.01
  epsilon_mag_mean = 0*np.ones(ps.n_bus)
  epsilon_mag_std = 1e-6*0.01

  # Model of the sensor
  H_mag   = np.eye(ps.n_bus) # Identity matrix
  H_angle = np.eye(ps.n_bus) # Identity matrix
  H = np.block([[H_angle, np.zeros_like(H_angle)], [np.zeros_like(H_angle), H_mag]])

  R_angle = epsilon_angle_std*np.eye(ps.n_bus)  # Identity matrix
  R_mag = epsilon_mag_std*np.eye(ps.n_bus)      # Identity matrix
  R = np.block([[R_angle, np.zeros_like(R_angle)], [np.zeros_like(R_angle), R_mag]])

  # Solver
  sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)

  # Initialize simulation
  t = 0
  res = defaultdict(list)
  t_0 = time.time()

  sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]

  # Simulate next step
  result = sol.step()
  x = sol.y
  v = sol.v
  t = sol.t

  # True angles and magnitudes
  x_v_angles = np.arctan2(np.imag(v), np.real(v)) # Bus angles
  x_v_mag = np.abs(v) # Bus voltage magnitudes
  X = np.concatenate((x_v_angles, x_v_mag), axis=0).reshape(-1, 1)

  # Measures angles and magnitudes
  epsilon_angle = np.random.multivariate_normal(mean=epsilon_angle_mean, cov=R_angle, size=1)
  epsilon_mag = np.random.multivariate_normal(mean=epsilon_mag_mean, cov=R_mag, size=1)
  epsilon = np.concatenate((epsilon_angle.T, epsilon_mag.T), axis=0).reshape(-1, 1)
  tmp = H@X
  z = tmp + epsilon

  x_hat = np.linalg.inv(H.T @ np.linalg.inv(R) @ H) @ H.T @ np.linalg.inv(R) @ z

  names = list(range(1, ps.n_bus + 1))
  plt.figure()
  plt.plot(names, x_v_angles*180/np.pi, 'o')
  plt.plot(names, x_hat[0:ps.n_bus]*180/np.pi, 'x')
  plt.xlabel('Generator')
  plt.ylabel('Angle [deg]')
  plt.show()

  plt.figure()
  plt.plot(names, x_v_mag, 'o')
  plt.plot(names, x_hat[ps.n_bus:2*ps.n_bus], 'x')
  plt.xlabel('Generator')
  plt.ylabel('Voltage [pu]')
  plt.show()