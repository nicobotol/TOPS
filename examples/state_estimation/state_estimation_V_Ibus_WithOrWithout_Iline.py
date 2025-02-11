import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib as mpl

# Use LaTeX for text rendering
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']

import time
import os
import tops.dynamic as dps
import tops.solvers as dps_sol
import numpy as np
from scipy.linalg import block_diag
np.random.seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))  # current file's directory
root_dir = os.path.abspath(os.path.join(script_dir, '../../src/tops/ps_models'))
sys.path.append(root_dir)

# Write a function that wrotes in a latex table the MSE and J values
def write_latex_table(case_names, MSE_errors, J_errors, filename):
    filepath = f"../assigments/system_estimation/results/{filename}.tex"
    label = f"tab:{filename}"
    with open(filepath, 'w') as f:
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\columnwidth}{!}{\n")
        # f.write("\\caption{MSE and J values for different cases}\n")
        f.write("\\begin{tabular}{ccc}\n")
        f.write("\\toprule\n")
        f.write("Case & MSE $\\left[\\si{\\square\\volt}\\right]$ & J $\\left[\\si{\\square\\volt}\\right]$  \\\\\n")
        f.write("\\midrule\n")
        for i in range(len(case_names)):
            f.write(f"{case_names[i]} & {MSE_errors[i]:.2e} & {J_errors[i]:.2e} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        # f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")

if __name__ == '__main__':

  #   ____       _
  #  / ___|  ___| |_   _   _ _ __
  #  \___ \ / _ \ __| | | | | '_ \
  #   ___) |  __/ |_  | |_| | |_) |
  #  |____/ \___|\__|  \__,_| .__/
  #                         |_|

  # Load model
  import gs4 as model_data
  # import k2a as model_data
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
  Ibus = ps.y_bus_red_full @ x # bus injected current
  Iline = np.diag(ps.lines['Line'].admittance) @ incidence.T @ x # line current
  x_Rv = np.real(x)     # real part of the voltage
  x_Iv = np.imag(x)     # imaginary part of the voltage
  x_RIbus = np.real(Ibus) # real part of the bus current
  x_IIbus = np.imag(Ibus) # imaginary part of the bus current
  x_RIline = np.real(Iline) # real part of the line current
  x_IIline = np.imag(Iline) # imaginary part of the line current
  
  X    = np.hstack([x_Rv, x_Iv]).reshape(-1,1) # vector of real states

  #   _   _       _          
  #  | \ | | ___ (_)___  ___ 
  #  |  \| |/ _ \| / __|/ _ \
  #  | |\  | (_) | \__ \  __/
  #  |_| \_|\___/|_|___/\___|
                           
  mu_v  = 0*np.ones(ps.n_bus)
  error_v = 5e-1
  error_Ibus = 1e-1
  error_Iline = 1e-1
  var_r  = (error_v/3)**2
  mu_Ibus  = 0*np.ones(ps.n_bus)
  var_Ibus  = (error_Ibus/3)**2
  mu_Iline  = 0*np.ones(ps.n_lines)
  var_Iline  = (error_Ibus/3)**2
  R_Rv  = var_r*np.eye(ps.n_bus)  # Covariance on the measure on the real part of v
  R_Iv  = var_r*np.eye(ps.n_bus)  # Covariance on the measure on the imaginary part of v
  R_v   = np.block([[R_Rv, np.zeros_like(R_Rv)], [np.zeros_like(R_Iv), R_Iv]])
  R_RIbus  = var_Ibus*np.eye(ps.n_bus)  # Covariance on the measure on the real part of i
  R_IIbus  = var_Ibus*np.eye(ps.n_bus)  # Covariance on the measure on the imaginary part of i
  R_Ibus   = np.block([[R_RIbus, np.zeros_like(R_RIbus)], [np.zeros_like(R_IIbus), R_IIbus]])
  R_RIline  = var_Iline*np.eye(ps.n_lines)  # Covariance on the measure on the real part of i
  R_IIline  = var_Iline*np.eye(ps.n_lines)  # Covariance on the measure on the imaginary part of i
  R_Iline   = np.block([[R_RIline, np.zeros_like(R_RIline)], [np.zeros_like(R_IIline), R_IIline]])
  R     = block_diag(R_v, R_Ibus, R_Iline) # Covariance matrix of all the measures
  R_V_Ibus   = block_diag(R_v, R_Ibus) # Covariance matrix of all the measures

  epsilon_Rv  = np.random.multivariate_normal(mean=mu_v, cov=R_Rv, size=1)
  epsilon_Iv  = np.random.multivariate_normal(mean=mu_v, cov=R_Iv, size=1)
  epsilon_RIbus  = np.random.multivariate_normal(mean=mu_Ibus, cov=R_RIbus, size=1)
  epsilon_IIbus  = np.random.multivariate_normal(mean=mu_Ibus, cov=R_IIbus, size=1)
  epsilon_RIline  = np.random.multivariate_normal(mean=mu_Iline, cov=R_RIline, size=1)
  epsilon_IIline  = np.random.multivariate_normal(mean=mu_Iline, cov=R_IIline, size=1)
  epsilon     = np.hstack([epsilon_Rv, epsilon_Iv, epsilon_RIbus, epsilon_IIbus, epsilon_RIline, epsilon_IIline])
  epsilon_V_Ibus   = np.hstack([epsilon_Rv, epsilon_Iv, epsilon_RIbus, epsilon_IIbus])

  #   ____                            
  #  / ___|  ___ _ __  ___  ___  _ __ 
  #  \___ \ / _ \ '_ \/ __|/ _ \| '__|
  #   ___) |  __/ | | \__ \ (_) | |   
  #  |____/ \___|_| |_|___/\___/|_|   

  # Measure v                                  
  H_Rv    = np.eye(ps.n_bus) # Identity matrix
  H_Iv    = np.eye(ps.n_bus) # Identity matrix
  H_v     = np.block([[H_Rv, np.zeros([ps.n_bus,ps.n_bus])], [np.zeros([ps.n_bus,ps.n_bus]), H_Iv]])

  # Measure Ibus
  H_RIbusRv  = np.real(ps.y_bus_red_full)
  H_RIbusIv  = -np.imag(ps.y_bus_red_full)
  H_IIbusRv  = np.imag(ps.y_bus_red_full)
  H_IIbusIv  = np.real(ps.y_bus_red_full)
  H_Ibus     = np.block([[H_RIbusRv, H_RIbusIv], [H_IIbusRv, H_IIbusIv]])

  # Measure Iline
  J = np.diag(ps.lines['Line'].admittance) @ incidence.T 
  H_RIlineRv  = np.real(J)
  H_RIlineIv  = -np.imag(J)
  H_IIlineRv  = np.imag(J)
  H_IIlineIv  = np.real(J)
  H_Iline     = np.block([[H_RIlineRv, H_RIlineIv], [H_IIlineRv, H_IIlineIv]])
  
  # Combine measures
  H       = np.vstack([H_v, H_Ibus, H_Iline]) # V + Ibus + Iline
  H_V_Ibus= np.vstack([H_v, H_Ibus])           # V + Ibus

  #   __  __                               
  #  |  \/  | ___  __ _ ___ _   _ _ __ ___ 
  #  | |\/| |/ _ \/ _` / __| | | | '__/ _ \
  #  | |  | |  __/ (_| \__ \ |_| | | |  __/
  #  |_|  |_|\___|\__,_|___/\__,_|_|  \___|
                                         
  z = (H @ X).reshape(-1,1) + epsilon.T
  z_v = z[0:2*ps.n_bus]
  z_v_comb = z_v[0:ps.n_bus] + 1j*z_v[ps.n_bus:2*ps.n_bus]
  z_Ibus = z[2*ps.n_bus:4*ps.n_bus]
  z_Ibus_comb = z_Ibus[0:ps.n_bus] + 1j*z_Ibus[ps.n_bus:2*ps.n_bus]
  z_Iline = z[4*ps.n_bus:4*ps.n_bus+2*ps.n_lines]
  z_Iline_comb = z_Iline[0:ps.n_lines] + 1j*z_Iline[ps.n_lines:2*ps.n_lines]

  z_V_Ibus = np.vstack([z_v, z_Ibus])

  #   _____     _   _                 _   _             
  #  | ____|___| |_(_)_ __ ___   __ _| |_(_) ___  _ __  
  #  |  _| / __| __| | '_ ` _ \ / _` | __| |/ _ \| '_ \ 
  #  | |___\__ \ |_| | | | | | | (_| | |_| | (_) | | | |
  #  |_____|___/\__|_|_| |_| |_|\__,_|\__|_|\___/|_| |_|

  # estimate v using V + Ibus + Iline                                                    
  G = H.T @ np.linalg.inv(R) @ H
  x_hat = np.linalg.inv(G) @ H.T @ np.linalg.inv(R) @ z
  x_hat_comb = x_hat[0:ps.n_bus] + 1j*x_hat[ps.n_bus:2*ps.n_bus]

  # estimate v using V + Ibus
  G_V_Ibus = H_V_Ibus.T @ np.linalg.inv(R_V_Ibus) @ H_V_Ibus
  x_hat_V_Ibus = np.linalg.inv(G_V_Ibus) @ H_V_Ibus.T @ np.linalg.inv(R_V_Ibus) @ z_V_Ibus
  x_hat_comb_V_Ibus = x_hat_V_Ibus[0:ps.n_bus] + 1j*x_hat_V_Ibus[ps.n_bus:2*ps.n_bus]

  # estimate v measuring only v
  G_v = H_v.T @ np.linalg.inv(R_v) @ H_v
  x_hat_v = np.linalg.inv(G_v) @ H_v.T @ np.linalg.inv(R_v) @ z_v
  x_hat_comb_v = x_hat_v[0:ps.n_bus] + 1j*x_hat_v[ps.n_bus:2*ps.n_bus]

  # estimate v measuring only Ibus
  G_Ibus = H_Ibus.T @ np.linalg.inv(R_Ibus) @ H_Ibus 
  x_hat_Ibus = np.linalg.inv(G_Ibus) @ H_Ibus.T @ np.linalg.inv(R_Ibus) @ z_Ibus
  x_hat_comb_Ibus = x_hat_Ibus[0:ps.n_bus] + 1j*x_hat_Ibus[ps.n_bus:2*ps.n_bus]

  # estimate v measuring only Iline
  G_Iline = H_Iline.T @ np.linalg.inv(R_Iline) @ H_Iline 
  x_hat_Iline = np.linalg.inv(G_Iline) @ H_Iline.T @ np.linalg.inv(R_Iline) @ z_Iline
  x_hat_comb_Iline = x_hat_Iline[0:ps.n_bus] + 1j*x_hat_Iline[ps.n_bus:2*ps.n_bus]
  e_x_Iline = X - x_hat_Iline
  e_x_Iline_comb = e_x_Iline[0:ps.n_bus] + 1j*e_x_Iline[ps.n_bus:2*ps.n_bus]
  print(f"Error Iline: {e_x_Iline_comb}")

  #   _____     _                                
  #  | ____|___| |_      ___ _ __ _ __ ___  _ __ 
  #  |  _| / __| __|    / _ \ '__| '__/ _ \| '__|
  #  | |___\__ \ |_ _  |  __/ |  | | | (_) | |   
  #  |_____|___/\__(_)  \___|_|  |_|  \___/|_|   

  # Estimation error using V + Ibus + Iline 
  error_hat = z - H @ x_hat
  J_hat = error_hat.T @ np.linalg.inv(R) @ error_hat # min value of the cost function    
  MSE_hat = np.mean(np.abs(X - x_hat)**2) # mean square error

  # Estimation error using V + Ibus
  error_hat_V_Ibus = z_V_Ibus - H_V_Ibus @ x_hat_V_Ibus      
  J_hat_V_Ibus = error_hat_V_Ibus.T @ np.linalg.inv(R_V_Ibus) @ error_hat_V_Ibus # min value of the cost function
  MSE_hat_V_Ibus = np.mean(np.abs(X - x_hat_V_Ibus)**2) # mean square error

  # Estimation error using only v
  error_hat_v = z_v - H_v @ x_hat_v
  J_hat_v = error_hat_v.T @ np.linalg.inv(R_v) @ error_hat_v # min value of the cost function
  MSE_hat_v = np.mean(np.abs(X - x_hat_v)**2) # mean square error

  # Estimation error using only Ibus
  error_hat_Ibus = z_Ibus - H_Ibus @ x_hat_Ibus
  J_hat_Ibus = error_hat_Ibus.T @ np.linalg.inv(R_Ibus) @ error_hat_Ibus # min value of the cost function
  MSE_hat_Ibus = np.mean(np.abs(X - x_hat_Ibus)**2) # mean square error                           

  # Estimation error using only Iline
  error_hat_Iline = z_Iline - H_Iline @ x_hat_Iline
  J_hat_Iline = error_hat_Iline.T @ np.linalg.inv(R_Iline) @ error_hat_Iline # min value of the cost function
  MSE_hat_Iline = np.mean(np.abs(X - x_hat_Iline)**2) # mean square error

  #  ____  _       _
  # |  _ \| | ___ | |_
  # | |_) | |/ _ \| __|
  # |  __/| | (_) | |_
  # |_|   |_|\___/ \__|

  case_names = ['V + Ibus + Iline', 'V + Ibus', 'V', 'Ibus', 'Iline']
  J_errors = [J_hat, J_hat_V_Ibus, J_hat_v, J_hat_Ibus, J_hat_Iline]
  J_errors = np.array([x.item() for x in J_errors])
  MSE_errors = [MSE_hat, MSE_hat_V_Ibus, MSE_hat_v, MSE_hat_Ibus, MSE_hat_Iline]
  names = list(range(1, ps.n_bus+1))

  # plt.figure()
  # plt.plot(names, np.abs(x), 'o', label='Real values')
  # plt.plot(names, np.abs(x_hat_comb), 'x', label=case_names[0])
  # plt.plot(names, np.abs(x_hat_comb_V_Ibus), '1',  label=case_names[1]) # v measuring i
  # plt.plot(names, np.abs(z_v_comb), 'v', label=case_names[2])     # v measuring v
  # plt.plot(names, np.abs(x_hat_comb_Ibus), '2',  label=case_names[3]) # v measuring i
  # plt.plot(names, np.abs(x_hat_comb_Iline), '3',  label=case_names[4]) # v measuring i
  # plt.xlabel('Bus number')
  # plt.ylabel('Mag [-]')
  # plt.legend()

  # plt.figure()
  # plt.plot(names, np.arctan2(np.imag(x),np.real(x)), 'o', label='Real values')
  # plt.plot(names, np.arctan2(np.imag(x_hat_comb),np.real(x_hat_comb)), 'x', label=case_names[0])
  # plt.plot(names, np.arctan2(np.imag(x_hat_comb_V_Ibus),np.real(x_hat_comb_V_Ibus)), '1',  label=case_names[1]) # v measuring i
  # plt.plot(names, np.arctan2(np.imag(z_v_comb),np.real(z_v_comb)), 'v', label=case_names[2])     # v measuring v
  # plt.plot(names, np.arctan2(np.imag(x_hat_comb_Ibus),np.real(x_hat_comb_Ibus)), '2',  label=case_names[3]) # v measuring i
  # plt.plot(names, np.arctan2(np.imag(x_hat_comb_Iline),np.real(x_hat_comb_Iline)), '3',  label=case_names[4]) # v measuring i
  # plt.xlabel('Bus number')
  # plt.ylabel('Angle [-]')
  # plt.legend()

  # # Plot the graphs side by side
  # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
  # # First plot
  # ax1.plot(names, np.real(x), 'o', label='Real values')
  # ax1.plot(names, np.real(x_hat_comb), 'x', label=case_names[0])
  # ax1.plot(names, np.real(x_hat_comb_V_Ibus), '1', label=case_names[1])  # v measuring i
  # ax1.plot(names, np.real(z_v_comb), 'v', label=case_names[2])  # v measuring v
  # ax1.plot(names, np.real(x_hat_comb_Ibus), '2', label=case_names[3])  # v measuring i
  # ax1.plot(names, np.real(x_hat_comb_Iline), '3', label=case_names[4])  # v measuring i
  # ax1.set_xlabel('Bus number', fontsize=20)
  # ax1.set_ylabel('Real [-]', fontsize=20)
  # ax1.set_title('Real part', fontsize=20)
  # ax1.tick_params(axis='both', which='major', labelsize=15)
  # ax1.legend(fontsize=15)
  # # Second plot
  # ax2.plot(names, np.imag(x), 'o', label='Real values')
  # ax2.plot(names, np.imag(x_hat_comb), 'x', label=case_names[0])
  # ax2.plot(names, np.imag(x_hat_comb_V_Ibus), '1', label=case_names[1])  # v measuring i
  # ax2.plot(names, np.imag(z_v_comb), 'v', label=case_names[2])  # v measuring v
  # ax2.plot(names, np.imag(x_hat_comb_Ibus), '2', label=case_names[3])  # v measuring i
  # ax2.plot(names, np.imag(x_hat_comb_Iline), '3', label=case_names[4])  # v measuring
  # ax2.set_xlabel('Bus number', fontsize=20)
  # ax2.set_ylabel('Imag [-]', fontsize=20)
  # ax2.set_title('Imaginary part', fontsize=20)
  # ax2.tick_params(axis='both', which='major', labelsize=15)
  # # Save the figure as a PDF
  # plt.tight_layout()
  # plt.savefig('../assigments/system_estimation/results/combined_plots.pdf')
  # plt.show()

  
  # fig, ax1 = plt.subplots()
  # line1,= ax1.plot(case_names, MSE_errors, 'o', label='MSE')
  # ax1.set_xlabel("Cases")
  # ax1.set_ylabel("MSE Error")
  # ax2 = ax1.twinx()
  # line2, = ax2.plot(case_names, J_errors, 'x', label='J')
  # ax2.set_ylabel("J")
  # lines = [line1, line2]  # Combine both lines
  # labels = [line.get_label() for line in lines]  # Extract labels
  # ax1.legend(lines, labels, loc="right")  # Set legend

  # plt.show()

  # Write the results in a latex table
  write_latex_table(case_names, MSE_errors, J_errors, 'basic_case_increased_error_voltage')

