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
  R_prev      = np.block([[R_angle, np.zeros_like(R_angle)], [np.zeros_like(R_angle), R_mag]])
  # Initial state
  x_v_angles  = 0*np.ones(ps.n_bus)
  x_v_mag     = 0*np.ones(ps.n_bus)
  X_prev      = np.concatenate((x_v_angles, x_v_mag), axis=0).reshape(-1, 1)
  P_prev      = np.linalg.inv(H.T @ np.linalg.inv(R_prev) @ H)

  for i in range(iter): 
    # Simulate next step
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=5e-3)
    result = sol.step()
    x = sol.y
    v = sol.v
    t = sol.t

    # Noise
    epsilon_angle_mean = 0*np.ones(ps.n_bus)
    epsilon_angle_std = 1e-6*0.01
    epsilon_mag_mean = 0*np.ones(ps.n_bus)
    epsilon_mag_std = 1e-6*0.01
    R_angle = epsilon_angle_std*np.eye(ps.n_bus)  # Identity matrix
    R_mag = epsilon_mag_std*np.eye(ps.n_bus)      # Identity matrix
    R = np.block([[R_angle, np.zeros_like(R_angle)], [np.zeros_like(R_angle), R_mag]])

    # True angles and magnitudes
    x_v_angles = np.arctan2(np.imag(v), np.real(v)) # Bus angles
    x_v_mag = np.abs(v) # Bus voltage magnitudes
    X = np.concatenate((x_v_angles, x_v_mag), axis=0).reshape(-1, 1)

    # Measures angles and magnitudes
    epsilon_angle = np.random.multivariate_normal(mean=epsilon_angle_mean, cov=R_angle, size=1)
    epsilon_mag = np.random.multivariate_normal(mean=epsilon_mag_mean, cov=R_mag, size=1)
    epsilon = np.concatenate((epsilon_angle.T, epsilon_mag.T), axis=0).reshape(-1, 1)
    z = H@X + epsilon

    x_hat = np.linalg.inv(H.T @ np.linalg.inv(R) @ H) @ H.T @ np.linalg.inv(R) @ z

    S = H@P_prev@H.T + R_prev
    W = P_prev@H.T*np.linalg.inv(S)
    X_hat_tmp = X_prev + W@(z - H@X_prev) # new estimation
    P_hat_tmp = P_prev - W@H@P_prev           # new covariance

    res['X_hat'].append(X_hat_tmp)
    res['P_hat'].append(P_hat_tmp)
    res['v'].append(v)

    X_prev = X_hat_tmp
    P_prev = P_hat_tmp

  X_angle_tmp = np.concatenate([X_hat[0] for X_hat in res['X_hat']])
  X_angle_tmp = X_angle_tmp*180/np.pi
  X_mag_tmp = np.concatenate([X_hat[39] for X_hat in res['X_hat']])
  v_tmp = [v[0] for v in res['v']]
  v_tmp_mag = np.abs(v_tmp)
  v_tmp_angle = np.arctan2(np.imag(v_tmp), np.real(v_tmp))*180/np.pi

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