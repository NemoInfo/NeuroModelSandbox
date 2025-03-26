from tqdm import tqdm
import numpy as np
from parameters import *
from dataclasses import dataclass


class STN_Population:

  def __init__(self, T: int, N: int, paramters: STN_Parameters = STN_Parameters()):
    self.p = paramters

    self.v = np.zeros((T, N))

    self.n = np.zeros((T, N))
    self.h = np.zeros((T, N))
    self.r = np.zeros((T, N))
    self.Ca = np.zeros((T, N))

    # Currents
    self.I_L = np.zeros((T, N))
    self.I_K = np.zeros((T, N))
    self.I_Na = np.zeros((T, N))
    self.I_T = np.zeros((T, N))
    self.I_Ca = np.zeros((T, N))
    self.I_AHP = np.zeros((T, N))
    self.I_G_S = np.zeros((T, N))
    self.s = np.zeros((T, N))

  def step(self, t):
    dt = 0.01      # [TEMP]


#    v, n, h, r, Ca = self.v[t], self.n[t], self.h[t], self.r[t], self.Ca[t]
#    n_inf = x_inf(v, self.p.tht_n, self.p.sig_n)
#    m_inf = x_inf(v, self.p.tht_m, self.p.sig_m)
#    h_inf = x_inf(v, self.p.tht_h, self.p.sig_h)
#    a_inf = x_inf(v, self.p.tht_a, self.p.sig_a)
#    r_inf = x_inf(v, self.p.tht_r, self.p.sig_r)
#    s_inf = x_inf(v, self.p.tht_s, self.p.sig_s)
#    b_inf = x_inf(r, self.p.tht_b, -self.p.sig_b) - self.p.b_const # [!]
#
#    tau_n = tau_x(v, self.p.tau_n_0, self.p.tau_n_1, self.p.tht_n_T, self.p.sig_n_T)
#    tau_h = tau_x(v, self.p.tau_h_0, self.p.tau_h_1, self.p.tht_h_T, self.p.sig_h_T)
#    tau_r = tau_x(v, self.p.tau_r_0, self.p.tau_r_1, self.p.tht_r_T, self.p.sig_r_T)
#
#    self.I_L[t] = self.p.g_L * (v - self.p.v_L)
#    self.I_K[t] = self.p.g_K * n**4 * (v - self.p.v_K)
#    self.I_Na[t] = self.p.g_Na * m_inf**3 * h * (v - self.p.v_Na)
#    self.I_T[t] = self.p.g_T * a_inf**3 * b_inf**2 * (v - self.p.v_Ca)
#    self.I_Ca[t] = self.p.g_Ca * s_inf**2 * (v - self.p.v_Ca)
#    self.I_AHP[t] = self.p.g_AHP * (v - self.p.v_K) * Ca / (Ca + self.p.k_1)
#    self.I_G_S[t] = self.p.g_G_S * (v - self.p.v_G_S) * (c_G_S.T @ s_gpe[t])
#
#    v[t + 1] = v + dt * (-self.I_L[t] - self.I_K[t] - self.I_Na[t] - self.I_T[t] - self.I_Ca[t] - self.I_AHP[t] -
#                         self.I_G_S[t] - self.I_ext[t])
#    n[t + 1] = n + dt * self.p.phi_n * (n_inf - n) / tau_n
#    h[t + 1] = h + dt * self.p.phi_h * (h_inf - h) / tau_h
#    r[t + 1] = r + dt * self.p.phi_r * (r_inf - r) / tau_r
#    Ca[t + 1] = Ca + dt * self.p.eps * ((-self.I_Ca[t] - self.I_T[t]) - self.p.k_Ca * Ca)
#
#    # STN synapses
#    H_inf = x_inf(v - self.p.tht_g, self.p.tht_g_H, self.p.sig_g_H)
#    s[t + 1] = s[t] + dt * (self.p.alpha * H_inf * (1 - self.s[t]) - self.p.beta * self.s[t])

# class GPe_Population(Neuron_Population):
#
#   def __init__(self, T: int, N: int, paramters: GPe_Parameters = GPe_Parameters()):
#     self.p = paramters
#     Neuron_Population.__init__(self, T, N)

# neuron params
# neuron state
# ^
# |
# v
# neuron currents


def rubin_terman(N_gpe, N_stn, I_ext_stn=lambda t, n: 0, \
    I_ext_gpe=lambda t, n: 0, I_app_gpe=lambda t, n: 0 , dt=0.01, T=5, \
    c_G_S=None, c_G_G=None, c_S_G=None, stn=STN_Parameters(), gpe=GPe_Parameters()):

  T = int(T * 1e3 / dt) # ms

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

  # Create data arrays
  I_L_stn = np.zeros((T, N_stn))
  I_K_stn = np.zeros((T, N_stn))
  I_Na_stn = np.zeros((T, N_stn))
  I_T_stn = np.zeros((T, N_stn))
  I_Ca_stn = np.zeros((T, N_stn))
  I_AHP_stn = np.zeros((T, N_stn))
  I_G_S = np.zeros((T, N_stn))
  s_stn = np.zeros((T, N_stn))

  I_L_gpe = np.zeros((T, N_gpe))
  I_K_gpe = np.zeros((T, N_gpe))
  I_Na_gpe = np.zeros((T, N_gpe))
  I_T_gpe = np.zeros((T, N_gpe))
  I_Ca_gpe = np.zeros((T, N_gpe))
  I_AHP_gpe = np.zeros((T, N_gpe))
  I_S_G = np.zeros((T, N_gpe))
  I_G_G = np.zeros((T, N_gpe))
  s_gpe = np.zeros((T, N_gpe))

  if N_stn:
    I_ext_stn = np.fromfunction(np.vectorize(I_ext_stn), (T, N_stn))
  if N_gpe:
    I_app_gpe = np.fromfunction(np.vectorize(I_app_gpe), (T, N_gpe))
    I_ext_gpe = np.fromfunction(np.vectorize(I_ext_gpe), (T, N_gpe))

  if c_G_S is None: c_G_S = np.zeros((N_gpe, N_stn), dtype=np.bool)
  if c_S_G is None: c_S_G = np.zeros((N_stn, N_gpe), dtype=np.bool)
  if c_G_G is None: c_G_G = np.zeros((N_gpe, N_gpe), dtype=np.bool)

  #  v_stn[0] = [-77., -77., -53.2, -53.2] * 2
  #  h_stn[0] = [0.19, 0.19, 0.1, 0.1] * 2
  #  n_stn[0] = [0.15, 0.15, 0.45, 0.45] * 2
  #  r_stn[0] = [0.23, 0.23, 0.6, 0.6] * 2
  #  Ca_stn[0] = [0.06, 0.06, 0.12, 0.12] * 2
  #  s_stn[0] = [.0, .0, .44, .44] * 2
  #
  #  v_gpe[0] = [-95., -95., -77., -77.] * 2
  #  n_gpe[0] = [.04, .04, .78, .78] * 2
  #  h_gpe[0] = [.95, .95, .2, .2] * 2
  #  Ca_gpe[0] = [0.06, 0.06, 0.035, 0.035] * 2
  #  r_gpe[0] = [.9, .9, 0.9, 0.9] * 2

  v_stn[0] = [
      -59.62828421888404, -61.0485669306943, -59.9232859246653, -58.70506521874258, -59.81316532105502,
      -60.41737514151719, -60.57000688576042, -60.77581472006873, -59.72163362685856, -59.20177081754847
  ]
  h_stn[0] = [
      0.5063486245631907, 0.2933274739456392, 0.4828268896903307, 0.5957938758715363, 0.4801708406464686,
      0.397555659151211, 0.3761635970127477, 0.3316364917935809, 0.4881964058107033, 0.5373898124788108
  ]
  n_stn[0] = [
      0.0301468039831072, 0.04412485475791555, 0.02936940165051648, 0.03307223867110721, 0.02961425249063069,
      0.02990618866753074, 0.03096707115136645, 0.03603641291454053, 0.02983123244237023, 0.03137696787429014
  ]
  r_stn[0] = [
      0.0295473069771012, 0.07318677802595788, 0.03401991571903244, 0.01899268957583912, 0.0322092810112401,
      0.04490215539151968, 0.0496024428039565, 0.05982606979469521, 0.03078507359379932, 0.02403333448524015
  ]
  Ca_stn[0] = [
      0.2994323366425385, 0.4076730264403847, 0.3271760563827424, 0.2456039126383157, 0.3090126869287847,
      0.3533066857313201, 0.3668697913124569, 0.3777575381495549, 0.3008309498107221, 0.2631312497961643
  ]
  s_stn[0] = [
      0.008821617722180833, 0.007400276913597601, 0.00850582621763913, 0.009886276645187469, 0.00862235586166425,
      0.008001611992658621, 0.007851916739337694, 0.007654426383227644, 0.008720434017133022, 0.009298664650592724
  ]

  v_gpe[0] = [
      -67.82599080345415, -67.93189025010138, -67.71998331113508, -67.642675227932, -67.84176237771105,
      -68.20005240297162, -68.25008741948682, -67.96902996444675, -67.94128590225782, -67.90220199361714
  ]
  n_gpe[0] = [
      0.2185706578168535, 0.2172726252685865, 0.2198825985895934, 0.2208386365344371, 0.218376554433867,
      0.2139920161919114, 0.2133851309251945, 0.2168122919020363, 0.2171513971203313, 0.2176303246979667
  ]
  h_gpe[0] = [
      0.6941693982604106, 0.6960525186649159, 0.6922676967009093, 0.6908829096175084, 0.694450894176868,
      0.7008187424839692, 0.701701517982433, 0.6967207207137638, 0.6962284698338927, 0.6955334306358607
  ]
  r_gpe[0] = [
      0.2573659658746555, 0.267633438957204, 0.2467211480536698, 0.2393102961633564, 0.25894821881624, 0.29699756425809,
      0.3026979450078091, 0.2718222561955378, 0.2692289296543874, 0.265227594196562
  ]
  Ca_gpe[0] = [
      0.009931221391373412, 0.01105798337560691, 0.008618198042586294, 0.007842254082952149, 0.01009947181199855,
      0.01439779182000786, 0.01499734401485086, 0.01153249037272892, 0.01131645351818554, 0.01090747666531076
  ]
  s_gpe[0] = [
      5.005384170945523e-06, 4.742408605511108e-06, 5.285062432830586e-06, 5.496855711727898e-06, 4.965314879070837e-06,
      4.134036862962592e-06, 4.035693901984886e-06, 4.652783831094474e-06, 4.718489668449042e-06, 4.813260429223516e-06
  ]

  for t in tqdm(range(T - 1), leave=False):
    # Update STN neurons
    v, n, h, r, Ca = v_stn[t], n_stn[t], h_stn[t], r_stn[t], Ca_stn[t]
    n_inf = x_inf(v, stn.tht_n, stn.sig_n)
    m_inf = x_inf(v, stn.tht_m, stn.sig_m)
    h_inf = x_inf(v, stn.tht_h, stn.sig_h)
    a_inf = x_inf(v, stn.tht_a, stn.sig_a)
    r_inf = x_inf(v, stn.tht_r, stn.sig_r)
    s_inf = x_inf(v, stn.tht_s, stn.sig_s)
    b_inf = x_inf(r, stn.tht_b, -stn.sig_b) - stn.b_const # [!]

    tau_n = tau_x(v, stn.tau_n_0, stn.tau_n_1, stn.tht_n_T, stn.sig_n_T)
    tau_h = tau_x(v, stn.tau_h_0, stn.tau_h_1, stn.tht_h_T, stn.sig_h_T)
    tau_r = tau_x(v, stn.tau_r_0, stn.tau_r_1, stn.tht_r_T, stn.sig_r_T)

    # Compute currents
    I_L_stn[t] = stn.g_L * (v - stn.v_L)
    I_K_stn[t] = stn.g_K * n**4 * (v - stn.v_K)
    I_Na_stn[t] = stn.g_Na * m_inf**3 * h * (v - stn.v_Na)
    I_T_stn[t] = stn.g_T * a_inf**3 * b_inf**2 * (v - stn.v_Ca)
    I_Ca_stn[t] = stn.g_Ca * s_inf**2 * (v - stn.v_Ca)
    I_AHP_stn[t] = stn.g_AHP * (v - stn.v_K) * Ca / (Ca + stn.k_1)
    I_G_S[t] = stn.g_G_S * (v - stn.v_G_S) * (c_G_S.T @ s_gpe[t])

    # Update state
    v_stn[t + 1] = v + dt * (-I_L_stn[t] - I_K_stn[t] - I_Na_stn[t] - I_T_stn[t] - I_Ca_stn[t] - I_AHP_stn[t] -
                             I_G_S[t] - I_ext_stn[t])
    n_stn[t + 1] = n + dt * stn.phi_n * (n_inf - n) / tau_n
    h_stn[t + 1] = h + dt * stn.phi_h * (h_inf - h) / tau_h
    r_stn[t + 1] = r + dt * stn.phi_r * (r_inf - r) / tau_r
    Ca_stn[t + 1] = Ca + dt * stn.eps * ((-I_Ca_stn[t] - I_T_stn[t]) - stn.k_Ca * Ca)

    # STN synapses
    H_inf = x_inf(v - stn.tht_g, stn.tht_g_H, stn.sig_g_H)
    s_stn[t + 1] = s_stn[t] + dt * (stn.alpha * H_inf * (1 - s_stn[t]) - stn.beta * s_stn[t])

    # Update GPe neurons
    v, n, h, r, Ca = v_gpe[t], n_gpe[t], h_gpe[t], r_gpe[t], Ca_gpe[t]
    n_inf = x_inf(v, gpe.tht_n, gpe.sig_n)
    m_inf = x_inf(v, gpe.tht_m, gpe.sig_m)
    h_inf = x_inf(v, gpe.tht_h, gpe.sig_h)
    a_inf = x_inf(v, gpe.tht_a, gpe.sig_a)
    r_inf = x_inf(v, gpe.tht_r, gpe.sig_r)
    s_inf = x_inf(v, gpe.tht_s, gpe.sig_s)

    tau_n = tau_x(v, gpe.tau_n_0, gpe.tau_n_1, gpe.tht_n_T, gpe.sig_n_T)
    tau_h = tau_x(v, gpe.tau_h_0, gpe.tau_h_1, gpe.tht_h_T, gpe.sig_h_T)

    I_L_gpe[t] = gpe.g_L * (v - gpe.v_L)
    I_K_gpe[t] = gpe.g_K * n**4 * (v - gpe.v_K)
    I_Na_gpe[t] = gpe.g_Na * m_inf**3 * h * (v - gpe.v_Na)
    I_T_gpe[t] = gpe.g_T * a_inf**3 * r * (v - gpe.v_Ca)
    I_Ca_gpe[t] = gpe.g_Ca * s_inf**2 * (v - gpe.v_Ca)
    I_AHP_gpe[t] = gpe.g_AHP * (v - gpe.v_K) * Ca / (Ca + gpe.k_1)
    I_G_G[t] = gpe.g_G_G * (v - gpe.v_G_G) * (c_G_G.T @ s_gpe[t])
    I_S_G[t] = gpe.g_S_G * (v - gpe.v_S_G) * (c_S_G.T @ s_stn[t])

    v_gpe[t + 1] = v + dt * (-I_L_gpe[t] - I_K_gpe[t] - I_Na_gpe[t] - I_T_gpe[t] - I_Ca_gpe[t] - I_AHP_gpe[t] -
                             I_ext_gpe[t] - I_G_G[t] - I_S_G[t] + I_app_gpe[t])
    n_gpe[t + 1] = n + dt * gpe.phi_n * (n_inf - n) / tau_n
    h_gpe[t + 1] = h + dt * gpe.phi_h * (h_inf - h) / tau_h
    r_gpe[t + 1] = r + dt * gpe.phi_r * (r_inf - r) / gpe.tau_r
    Ca_gpe[t + 1] = Ca + dt * gpe.eps * ((-I_Ca_gpe[t] - I_T_gpe[t]) - gpe.k_Ca * Ca)

    # GPe -> X synapses
    H_inf = x_inf(v - gpe.tht_g, gpe.tht_g_H, gpe.sig_g_H)
    s_gpe[t + 1] = s_gpe[t] + dt * (gpe.alpha * H_inf * (1 - s_gpe[t]) - gpe.beta * s_gpe[t])

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
  print(STN_Population(1000, N=8, tht_b=0).b_const)
  print(STN_Population(1000, N=8).b_const)
