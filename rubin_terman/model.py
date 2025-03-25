from tqdm import tqdm
import numpy as np

def rubin_terman(N_gpe, N_stn, I_ext_stn=lambda t, n: 0, \
    I_ext_gpe=lambda t, n: 0, I_app_gpe=lambda t, n: 0 , dt=0.01, T=5, \
    c_G_S=None, c_G_G=None, c_S_G=None):

  T = int(T * 1e3 / dt)  #ms

  v_gpe = np.zeros((T, N_gpe))
  v_stn = np.zeros((T, N_stn))

  n_stn = np.zeros((T, N_stn))
  h_stn = np.zeros((T, N_stn))
  r_stn = np.zeros((T, N_stn))
  Ca_stn = np.zeros((T, N_stn))
  n_gpe = np.zeros((T, N_gpe))
  h_gpe = np.zeros((T, N_gpe))
  r_gpe = np.zeros((T, N_gpe))
  Ca_gpe = np.zeros((T, N_gpe))

  # STN Parameters
  g_L_stn = 2.25  #  nS/um^2
  g_K_stn = 45.  #  nS/um^2
  g_Na_stn = 37.5  # nS/um^2
  g_T_stn = 0.5  #   nS/um^2
  g_Ca_stn = 0.5  #  nS/um^2
  g_AHP_stn = 9.  # nS/um^2

  v_L_stn = -60.  #  mV
  v_K_stn = -80.  #  mV
  v_Na_stn = 55.  #  mV
  v_Ca_stn = 140.  # mV
  v_G_S = -85.  # mV

  tau_h_1_stn = 500.  # ms
  tau_n_1_stn = 100.  # ms
  tau_r_1_stn = 17.5  # ms
  tau_h_0_stn = 1.  #   ms
  tau_n_0_stn = 1.  #   ms
  tau_r_0_stn = 40.  #  ms

  phi_h_stn = 0.75
  phi_n_stn = 0.75
  phi_r_stn = 0.2

  k_1_stn = 15.
  k_Ca_stn = 22.5
  eps_stn = 3.75e-5  # ms^-1

  tht_m_stn = -30.
  tht_h_stn = -39.
  tht_n_stn = -32.
  tht_r_stn = -67.
  tht_a_stn = -63.
  tht_b_stn = 0.4
  tht_s_stn = -39.

  tht_h_T_stn = -57.
  tht_n_T_stn = -80.
  tht_r_T_stn = 68.

  tht_g_H_stn = -39.
  tht_g_stn = 30.
  alpha_stn = 5.  # ms^-1

  sig_m_stn = 15.
  sig_h_stn = -3.1
  sig_n_stn = 8.
  sig_r_stn = -2.0
  sig_a_stn = 7.8
  sig_b_stn = -0.1
  sig_s_stn = 8.

  sig_h_T_stn = -3.
  sig_n_T_stn = -26.
  sig_r_T_stn = -2.2
  sig_g_H_stn = 8.
  beta_stn = 1.  # ms^-1

  b_const = 1 / (1 + np.exp(-tht_b_stn / sig_b_stn))

  # GPe Parameters
  g_L_gpe = 0.1  #   nS/um^2
  g_K_gpe = 30.0  #  nS/um^2
  g_Na_gpe = 120.  # nS/um^2
  g_T_gpe = 0.5  #   nS/um^2
  g_Ca_gpe = 0.15  # nS/um^2
  g_AHP_gpe = 30.  # nS/um^2

  v_L_gpe = -55.  #  mV
  v_K_gpe = -80.  #  mV
  v_Na_gpe = 55.  #  mV
  v_Ca_gpe = 120.  # mV
  v_G_G = -100.  #   mV
  v_S_G = 0.  #      mV

  tau_h_1_gpe = 0.27  # ms
  tau_n_1_gpe = 0.27  # ms
  tau_h_0_gpe = 0.05  # ms
  tau_n_0_gpe = 0.05  # ms
  tau_r_gpe = 30.  #    ms

  phi_h_gpe = 0.05
  phi_n_gpe = 0.05
  phi_r_gpe = 1.0

  k_1_gpe = 30.
  k_Ca_gpe = 20.
  eps_gpe = 1e-4  # ms^-1

  tht_m_gpe = -37.
  tht_h_gpe = -58.
  tht_n_gpe = -50.
  tht_r_gpe = -70.
  tht_a_gpe = -57.
  tht_s_gpe = -35.

  tht_h_T_gpe = -40.
  tht_n_T_gpe = -40.
  tht_g_H_gpe = -57.
  tht_g_gpe = 20.
  alpha_gpe = 2.0  # ms^-1

  sig_m_gpe = 10.
  sig_h_gpe = -12.
  sig_n_gpe = 14.
  sig_r_gpe = -2.
  sig_a_gpe = 2.
  sig_s_gpe = 2.

  sig_h_T_gpe = -12.
  sig_n_T_gpe = -12.
  sig_g_H_gpe = 2.
  beta_gpe = .08  # ms^-1

  # Synaptic Parameters
  g_G_S = 4.5  #   nS/um^2
  g_S_G = 0.56  #  nS/um^2
  g_G_G = 0.06  #  nS/um^2

  # Initilase Variables
  v_stn[0] = -65.  # mV
  n_stn[0] = x_inf(v_stn[0], tht_n_stn, sig_n_stn)
  #h_stn[0] = np.linspace(0, 0.8, N_stn)
  h_stn[0] = np.random.uniform(low=0, high=x_inf(v_stn[0, 0], tht_h_stn, sig_h_stn), size=N_stn)
  r_stn[0] = x_inf(v_stn[0], tht_r_stn, sig_r_stn)
  Ca_stn[0] = 0.05

  v_gpe[0] = -65.  # mV
  n_gpe[0] = x_inf(v_gpe[0], tht_n_gpe, sig_n_gpe)
  # h_gpe[0] = np.linspace(0, 0.8, N_gpe)
  h_gpe[0] = x_inf(v_gpe[0], tht_h_gpe, sig_h_gpe)
  r_gpe[0] = x_inf(v_gpe[0], tht_r_gpe, sig_r_gpe)
  Ca_gpe[0] = 0.0

  # Create data arrays
  I_L_stn = np.zeros((T, N_stn))
  I_K_stn = np.zeros((T, N_stn))
  I_Na_stn = np.zeros((T, N_stn))
  I_T_stn = np.zeros((T, N_stn))
  I_Ca_stn = np.zeros((T, N_stn))
  I_AHP_stn = np.zeros((T, N_stn))
  I_G_S = np.zeros((T, N_stn))
  s_stn = np.zeros((T, N_stn))

  v_stn[0] = [-77., -77., -53.2, -53.2] * 2
  h_stn[0] = [0.19, 0.19, 0.1, 0.1] * 2

  n_stn[0, [0, 1, 4, 5]] = 0.15
  n_stn[0, [2, 3, 6, 7]] = 0.45

  r_stn[0, [0, 1, 4, 5]] = 0.23
  r_stn[0, [2, 3, 6, 7]] = 0.6

  Ca_stn[0, [0, 1, 4, 5]] = 0.06
  Ca_stn[0, [2, 3, 6, 7]] = 0.12

  s_stn[0, [0, 1, 4, 5]] = .0
  s_stn[0, [2, 3, 6, 7]] = .44

  if N_stn:
    I_ext_stn = np.fromfunction(np.vectorize(I_ext_stn), (T, N_stn))

  I_L_gpe = np.zeros((T, N_gpe))
  I_K_gpe = np.zeros((T, N_gpe))
  I_Na_gpe = np.zeros((T, N_gpe))
  I_T_gpe = np.zeros((T, N_gpe))
  I_Ca_gpe = np.zeros((T, N_gpe))
  I_AHP_gpe = np.zeros((T, N_gpe))
  I_S_G = np.zeros((T, N_gpe))
  I_G_G = np.zeros((T, N_gpe))
  s_gpe = np.zeros((T, N_gpe))

  v_gpe[0, 0:2] = -95.
  n_gpe[0, 0:2] = .04
  h_gpe[0, 0:2] = .95
  Ca_gpe[0, 0:2] = 0.06
  r_gpe[0, 0:2] = .9
  v_gpe[0, 2:4] = -77.
  n_gpe[0, 2:4] = .78
  h_gpe[0, 2:4] = .2
  Ca_gpe[0, 2:4] = 0.035
  r_gpe[0, 2:4] = 0.9
  s_gpe[0, 0:2] = 0.09
  s_gpe[0, 2:4] = .5
  v_gpe[0, 4:6] = -95.
  n_gpe[0, 4:6] = .04
  h_gpe[0, 4:6] = .95
  Ca_gpe[0, 4:6] = 0.06
  r_gpe[0, 4:6] = .9
  v_gpe[0, 6:8] = -77.
  n_gpe[0, 6:8] = .78
  h_gpe[0, 6:8] = .2
  Ca_gpe[0, 6:8] = 0.035
  r_gpe[0, 6:8] = 0.9
  s_gpe[0, 4:6] = 0.09
  s_gpe[0, 6:8] = .5

  if N_gpe:
    I_app_gpe = np.fromfunction(np.vectorize(I_app_gpe), (T, N_gpe))
    I_ext_gpe = np.fromfunction(np.vectorize(I_ext_gpe), (T, N_gpe))

  if c_G_S is None: c_G_S = np.zeros((N_gpe, N_stn), dtype=np.bool)
  if c_S_G is None: c_S_G = np.zeros((N_stn, N_gpe), dtype=np.bool)
  if c_G_G is None: c_G_G = np.zeros((N_gpe, N_gpe), dtype=np.bool)

  for t in tqdm(range(T - 1), leave=False):
    # Update STN neurons
    v, n, h, r, Ca = v_stn[t], n_stn[t], h_stn[t], r_stn[t], Ca_stn[t]
    n_inf = x_inf(v, tht_n_stn, sig_n_stn)
    m_inf = x_inf(v, tht_m_stn, sig_m_stn)
    h_inf = x_inf(v, tht_h_stn, sig_h_stn)
    a_inf = x_inf(v, tht_a_stn, sig_a_stn)
    r_inf = x_inf(v, tht_r_stn, sig_r_stn)
    s_inf = x_inf(v, tht_s_stn, sig_s_stn)
    b_inf = 1 / (1 + np.exp((r - tht_b_stn) / sig_b_stn)) - b_const

    tau_n = tau_x(v, tau_n_0_stn, tau_n_1_stn, tht_n_T_stn, sig_n_T_stn)
    tau_h = tau_x(v, tau_h_0_stn, tau_h_1_stn, tht_h_T_stn, sig_h_T_stn)
    tau_r = tau_x(v, tau_r_0_stn, tau_r_1_stn, tht_r_T_stn, sig_r_T_stn)

    I_L_stn[t] = g_L_stn * (v - v_L_stn)
    I_K_stn[t] = g_K_stn * n**4 * (v - v_K_stn)
    I_Na_stn[t] = g_Na_stn * m_inf**3 * h * (v - v_Na_stn)
    I_T_stn[t] = g_T_stn * a_inf**3 * b_inf**2 * (v - v_Ca_stn)
    I_Ca_stn[t] = g_Ca_stn * s_inf**2 * (v - v_Ca_stn)
    I_AHP_stn[t] = g_AHP_stn * (v - v_K_stn) * Ca / (Ca + k_1_stn)
    I_G_S[t] = g_G_S * (v - v_G_S) * (c_G_S.T @ s_gpe[t])

    v_stn[t + 1] = v + dt * (-I_L_stn[t] - I_K_stn[t] - I_Na_stn[t] - I_T_stn[t] - I_Ca_stn[t] - I_AHP_stn[t] -
                             I_G_S[t] - I_ext_stn[t])
    n_stn[t + 1] = n + dt * phi_n_stn * (n_inf - n) / tau_n
    h_stn[t + 1] = h + dt * phi_h_stn * (h_inf - h) / tau_h
    r_stn[t + 1] = r + dt * phi_r_stn * (r_inf - r) / tau_r
    Ca_stn[t + 1] = Ca + dt * eps_stn * ((-I_Ca_stn[t] - I_T_stn[t]) - k_Ca_stn * Ca)

    # STN synapses
    H_inf = x_inf(v - tht_g_stn, tht_g_H_stn, sig_g_H_stn)
    s_stn[t + 1] = s_stn[t] + dt * (alpha_stn * H_inf * (1 - s_stn[t]) - beta_stn * s_stn[t])

    # Update GPe neurons
    v, n, h, r, Ca = v_gpe[t], n_gpe[t], h_gpe[t], r_gpe[t], Ca_gpe[t]
    n_inf = x_inf(v, tht_n_gpe, sig_n_gpe)
    m_inf = x_inf(v, tht_m_gpe, sig_m_gpe)
    h_inf = x_inf(v, tht_h_gpe, sig_h_gpe)
    a_inf = x_inf(v, tht_a_gpe, sig_a_gpe)
    r_inf = x_inf(v, tht_r_gpe, sig_r_gpe)
    s_inf = x_inf(v, tht_s_gpe, sig_s_gpe)

    tau_n = tau_x(v, tau_n_0_gpe, tau_n_1_gpe, tht_n_T_gpe, sig_n_T_gpe)
    tau_h = tau_x(v, tau_h_0_gpe, tau_h_1_gpe, tht_h_T_gpe, sig_h_T_gpe)

    I_L_gpe[t] = g_L_gpe * (v - v_L_gpe)
    I_K_gpe[t] = g_K_gpe * n**4 * (v - v_K_gpe)
    I_Na_gpe[t] = g_Na_gpe * m_inf**3 * h * (v - v_Na_gpe)
    I_T_gpe[t] = g_T_gpe * a_inf**3 * r * (v - v_Ca_gpe)
    I_Ca_gpe[t] = g_Ca_gpe * s_inf**2 * (v - v_Ca_gpe)
    I_AHP_gpe[t] = g_AHP_gpe * (v - v_K_gpe) * Ca / (Ca + k_1_gpe)
    I_G_G[t] = g_G_G * (v - v_G_G) * (c_G_G.T @ s_gpe[t])
    I_S_G[t] = g_S_G * (v - v_S_G) * (c_S_G.T @ s_stn[t])

    v_gpe[t + 1] = v + dt * (-I_L_gpe[t] - I_K_gpe[t] - I_Na_gpe[t] - I_T_gpe[t] - I_Ca_gpe[t] - I_AHP_gpe[t] -
                             I_ext_gpe[t] - I_G_G[t] - I_S_G[t] + I_app_gpe[t])
    n_gpe[t + 1] = n + dt * phi_n_gpe * (n_inf - n) / tau_n
    h_gpe[t + 1] = h + dt * phi_h_gpe * (h_inf - h) / tau_h
    r_gpe[t + 1] = r + dt * phi_r_gpe * (r_inf - r) / tau_r_gpe
    Ca_gpe[t + 1] = Ca + dt * eps_gpe * ((-I_Ca_gpe[t] - I_T_gpe[t]) - k_Ca_gpe * Ca)

    # GPe -> X synapses
    H_inf = x_inf(v - tht_g_gpe, tht_g_H_gpe, sig_g_H_gpe)
    s_gpe[t + 1] = s_gpe[t] + dt * (alpha_gpe * H_inf * (1 - s_gpe[t]) - beta_gpe * s_gpe[t])

  return {
      "I_L_stn": I_L_stn,
      "I_K_stn": I_K_stn,
      "I_Na_stn": I_Na_stn,
      "I_T_stn": I_T_stn,
      "I_Ca_stn": I_Ca_stn,
      "I_AHP_stn": I_AHP_stn,
      "I_ext_stn": I_ext_stn,
      "Ca_stn": Ca_stn,
      "v_stn": v_stn,
      "I_L_gpe": I_L_gpe,
      "I_K_gpe": I_K_gpe,
      "I_Na_gpe": I_Na_gpe,
      "I_T_gpe": I_T_gpe,
      "I_Ca_gpe": I_Ca_gpe,
      "I_AHP_gpe": I_AHP_gpe,
      "I_ext_gpe": I_ext_gpe,
      "I_app_gpe": I_app_gpe,
      "Ca_gpe": Ca_gpe,
      "v_gpe": v_gpe,
      "I_G_S": I_G_S,
      "I_G_G": I_G_G,
      "I_S_G": I_S_G,
      "s_stn": s_stn,
      "s_gpe": s_gpe,
  }


def x_inf(v, tht_x, sig_x):
  return 1 / (1 + np.exp((tht_x - v) / sig_x))


def tau_x(v, tau_x_0, tau_x_1, tht_x_T, sig_x_T):
  return tau_x_0 + tau_x_1 / (1 + np.exp((tht_x_T - v) / sig_x_T))


if __name__ == "__main__":
  np.random.seed(69)

  N_gpe, N_stn = 3, 3
  dt = 0.01
  I_app = lambda t, n: -1.2

  c_G_S = np.array([[0, 0, 0,], \
                    [0, 0, 0,], \
                    [0, 0, 0,]], dtype=np.bool)

  c_S_G = np.array([[1, 1, 0,], \
                    [0, 1, 1,], \
                    [1, 0, 1,]], dtype=np.bool)

  c_G_G = np.array([[0, 0, 0,], \
                    [0, 0, 0,], \
                    [0, 0, 0,]], dtype=np.bool)

  _ = rubin_terman(N_gpe, N_stn, I_app_gpe=I_app, dt=dt, T=2, c_G_S=c_G_S, c_G_G=c_G_G, c_S_G=c_S_G)
