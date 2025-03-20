import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

a = np.array([2., 5.])
b = np.array([.08, 1.])
theta_g = np.array([20., 30.])
theta_gH = np.array([-57., -39.])
sigma_gH = np.array([2., 8.])
pre_type = ["GPe", "STN"]

y1 = np.linspace(-80, 80, 500)
y2 = np.linspace(-0, 1, 500)
X, Y = np.meshgrid(y1, y2)

for a, b, theta_g, theta_gH, sigma_gH, pre_type in zip(a, b, theta_g, theta_gH, sigma_gH, pre_type):
  H_inf = lambda v: 1 / (1 + np.exp(-(v - theta_gH) / sigma_gH))
  sj_inf = a * H_inf(y1 - theta_g) / (a * H_inf(y1 - theta_g) + b)

  v = a * H_inf(y1[None, :] - theta_g) * (1 - y2[:, None]) - b * y2[:, None]

  vmin, vmax = v.min(), v.max()
  norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
  pcm = plt.pcolormesh(X, Y, v, cmap="managua", norm=norm, shading="auto")
  ticks = np.concatenate((np.linspace(vmin, 0, 5), np.linspace(0, vmax, 5)[1:]))
  cbar = plt.colorbar(pcm, ticks=ticks, format=lambda x, _: f"{x:.2f}" if x < 0 else f" {x:.2f}")
  cbar.set_label(r"$\dot s_j$", rotation=0, fontsize=14, labelpad=10, va='center')
  plt.plot(y1, sj_inf, 'w:', lw=2)
  plt.title(f"% post channels open by presynaptic {pre_type}", fontsize=14, pad=20)
  plt.xlabel(r"$\rm{v_{pre}}$", fontsize=14)
  plt.ylabel("$s_j$", rotation=0, fontsize=14, labelpad=10, va='center')

  plt.savefig(f"syn_pre_{pre_type}.png")
  plt.show()
