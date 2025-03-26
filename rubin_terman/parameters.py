import numpy as np
from dataclasses import dataclass, field


@dataclass()
class STN_Parameters:
  # Conductances
  g_L: float = 2.25  # nS/um^2 # x
  g_K: float = 45.   # nS/um^2 # x
  g_Na: float = 37.5 # nS/um^2 # x
  g_T: float = 0.5   # nS/um^2 # x
  g_Ca: float = 0.5  # nS/um^2 # x
  g_AHP: float = 9.  # nS/um^2 # x
  g_G_S: float = 2.5 # nS/um^2 # x

  # Reversal potentials
  v_L: float = -60.    # mV # x
  v_K: float = -80.    # mV # x
  v_Na: float = 55.    # mV # x
  v_Ca: float = 140.   # mV # x
  v_G_S: float = -100. # -85. # mV [!]

  # Time constants
  tau_h_1: float = 500. # ms # x
  tau_n_1: float = 100. # ms # x
  tau_r_1: float = 17.5 # ms
  tau_h_0: float = 1.   # ms # x
  tau_n_0: float = 1.   # ms # x
  tau_r_0: float = 7.1  # 40. # ms [!]

  phi_h: float = 0.75 # x
  phi_n: float = 0.75 # x
  phi_r: float = 0.5  # 0.2 # [!]

  # Calcium parameters
  k_1: float = 15.     # x
  k_Ca: float = 22.5   # x
  eps: float = 3.75e-5 # ms^-1 x (== phi_h * 5e-5)

  # Threshold potentials
  tht_m: float = -30. # mV x
  tht_h: float = -39. # mV x
  tht_n: float = -32. # mV x
  tht_r: float = -67. # mV # x
  tht_a: float = -63. # mV # x
  tht_b: float = 0.25 # 0.4 [!]
  tht_s: float = -39. # mV x

  # Tau threshold potentials
  tht_h_T: float = -57. # mV # x
  tht_n_T: float = -80. # mV # x
  tht_r_T: float = 68.  # mV # x

  # Synaptic threshold potentials
  tht_g_H: float = -39. # mV
  tht_g: float = 30.    # mV # x

  # Synaptic rate constants
  alpha: float = 5. # ms^-1 # x
  beta: float = 1.  # ms^-1 # x

  # Sigmoid slopes
  sig_m: float = 15.   # x
  sig_h: float = -3.1  # x
  sig_n: float = 8.    # x
  sig_r: float = -2.0  # x
  sig_a: float = 7.8   # x
  sig_b: float = -0.07 # -0.1 [!]
  sig_s: float = 8.    # x

  # Tau sigmoid slopes
  sig_h_T: float = -3.  # x
  sig_n_T: float = -26. # x
  sig_r_T: float = -2.2 # x

  # Synaptic sigmoid slopes
  sig_g_H: float = 8.

  b_const: float = field(init=False) # Baked constant term of \b_infty

  def __post_init__(self):
    self.b_const = 1 / (1 + np.exp(-self.tht_b / self.sig_b))


@dataclass
class GPe_Parameters():
  # Conductances
  g_L: float = 0.1    # nS/um^2
  g_K: float = 30.0   # nS/um^2
  g_Na: float = 120.  # nS/um^2
  g_T: float = 0.5    # nS/um^2
  g_Ca: float = 0.15  # nS/um^2
  g_AHP: float = 30.  # nS/um^2
  g_S_G: float = 0.03 # nS/um^2
  g_G_G: float = 0.06 # nS/um^2

  # Reversal potentials
  v_L: float = -55.    # mV
  v_K: float = -80.    # mV
  v_Na: float = 55.    # mV
  v_Ca: float = 120.   # mV
  v_G_G: float = -100. # mV
  v_S_G: float = 0.    # mV

  # Time constants
  tau_h_1: float = 0.27 # ms
  tau_n_1: float = 0.27 # ms
  tau_h_0: float = 0.05 # ms
  tau_n_0: float = 0.05 # ms
  tau_r: float = 30.    # ms

  phi_h: float = 0.05
  phi_n: float = 0.05
  phi_r: float = 1.0

  # Calcium parameters
  k_1: float = 30
  k_Ca: float = 20
  eps: float = 1e-4 # ms^-1

  # Threshold potentials
  tht_m: float = -37 # mV
  tht_h: float = -58 # mV
  tht_n: float = -50 # mV
  tht_r: float = -70 # mV
  tht_a: float = -57 # mV
  tht_s: float = -35 # mV

  # Tau threshold potentials
  tht_h_T: float = -40 # mV
  tht_n_T: float = -40 # mV

  # Synaptic threshold potentials
  tht_g_H: float = -35 # mV [DODGY] in paper its -57 in code its -35
  tht_g: float = 20    # mV

  # Synaptic rate constants
  alpha: float = 2  # ms^-1
  beta: float = .08 # ms^-1

  # Sigmoid slopes
  sig_m: float = 10
  sig_h: float = -12
  sig_n: float = 14
  sig_r: float = -2
  sig_a: float = 2
  sig_s: float = 2

  # Tau sigmoid slopes
  sig_h_T: float = -12
  sig_n_T: float = -12

  # Synaptic sigmoid slope
  sig_g_H: float = 2


if __name__ == "__main__":
  print(STN_Parameters(tht_b=0).b_const)
  print(STN_Parameters().b_const)
