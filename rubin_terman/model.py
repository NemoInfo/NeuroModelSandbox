from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def rubin_terman():
  N_gpe, N_stn = 10, 1
  dt = 0.01  #ms
  T = int(5e3 / dt)  #ms

  v_gpe = np.zeros((T, N_gpe))

  # These initialisations can be very important
  v_stn = np.ones((T, N_stn)) * -60
  n_stn = np.zeros((T, N_stn))
  h_stn = np.zeros((T, N_stn))
  r_stn = np.zeros((T, N_stn))
  Ca_stn = np.ones((T, N_stn)) * 0.5

  # Parameters
  g_L_stn = 2.25  #  nS/um^2
  g_K_stn = 45.  #   nS/um^2
  g_Na_stn = 37.5  # nS/um^2
  g_T_stn = .5  #    nS/um^2
  g_Ca_stn = .5  #   nS/um^2
  g_AHP_stn = 9.0  # nS/um^2

  v_L_stn = -60.  #  mV
  v_K_stn = -80.  #  mV
  v_Na_stn = 55.  #  mV
  v_Ca_stn = 140.  # mV

  k_1 = 15.
  k_Ca = 22.5
  eps = 3.75e-5  # ms^-1

  phi_h_stn = .75
  phi_n_stn = .75
  phi_r_stn = 0.2

  tht_m_stn = -30.
  tht_h_stn = -39.
  tht_n_stn = -32.
  tht_r_stn = -67.
  tht_a_stn = -63.
  tht_b_stn = .4
  tht_s_stn = -39.

  sig_m_stn = 15.
  sig_h_stn = -3.1
  sig_n_stn = 8.
  sig_r_stn = -2.0
  sig_a_stn = 7.8
  sig_b_stn = -.1
  sig_s_stn = 8.

  tau_h_1_stn = 500.  # ms
  tau_n_1_stn = 100.  # ms
  tau_r_1_stn = 17.5  # ms
  tau_h_0_stn = 1.  #   ms
  tau_n_0_stn = 1.  #   ms
  tau_r_0_stn = 40.  #  ms

  tht_h_T_stn = -57.
  tht_n_T_stn = -80.
  tht_r_T_stn = 68.

  sig_h_T_stn = -3.
  sig_n_T_stn = -26.
  sig_r_T_stn = -2.2

  b_const = 1 / (1 + np.exp(-tht_b_stn / sig_b_stn))

  for t in tqdm(range(T - 1), leave=False):
    # Update STN neurons
    for i, (v, n, h, r, Ca) in enumerate(zip(v_stn[t], n_stn[t], h_stn[t], r_stn[t], Ca_stn[t])):
      n_inf = x_inf(v, tht_n_stn, sig_n_stn)
      m_inf = x_inf(v, tht_m_stn, sig_m_stn)
      h_inf = x_inf(v, tht_h_stn, sig_h_stn)
      a_inf = x_inf(v, tht_a_stn, sig_a_stn)
      r_inf = x_inf(v, tht_r_stn, sig_r_stn)
      s_inf = x_inf(v, tht_s_stn, sig_s_stn)
      b_inf = x_inf(r, tht_b_stn, sig_b_stn) - b_const

      tau_n = tau_x(v, tau_n_0_stn, tau_n_1_stn, tht_n_T_stn, sig_n_T_stn)
      tau_h = tau_x(v, tau_h_0_stn, tau_h_1_stn, tht_h_T_stn, sig_h_T_stn)
      tau_r = tau_x(v, tau_r_0_stn, tau_r_1_stn, tht_r_T_stn, sig_r_T_stn)

      n = n_stn[t + 1, i] = n + dt * phi_n_stn * (n_inf - n) / tau_n
      h = h_stn[t + 1, i] = h + dt * phi_h_stn * (h_inf - h) / tau_h
      r = r_stn[t + 1, i] = r + dt * phi_r_stn * (r_inf - r) / tau_r

      I_L = g_L_stn * (v - v_L_stn)
      I_Na = g_Na_stn * m_inf**3 * h * (v - v_Na_stn)
      I_K = g_K_stn * n**4 * (v - v_K_stn)
      I_T = g_T_stn * a_inf**3 * b_inf**2 * (v - v_Ca_stn)
      I_Ca = g_Ca_stn * s_inf**2 * (v - v_Ca_stn)

      Ca = Ca_stn[t + 1, i] = Ca + dt * eps * (-I_Ca - I_T - k_Ca * Ca)
      I_AHP = g_AHP_stn * (v - v_Ca_stn) * Ca / (Ca + k_1)

      v_stn[t + 1, i] = v + dt * (-I_L - I_K - I_Na - I_T - I_Ca - I_AHP)

      # print()
      # _ = input()


#  plt.plot(v_stn[:int(1e3 // dt), 0])
  plt.plot(v_stn[:, 0])
  plt.show()


def x_inf(v, tht_x, sig_x):
  return 1 / (1 + np.exp((tht_x - v) / sig_x))


def tau_x(v, tau_x_0, tau_x_1, tht_x_T, sig_x_T):
  return tau_x_0 + tau_x_1 / (1 + np.exp((tht_x_T - v) / sig_x_T))


if __name__ == "__main__":
  rubin_terman()
