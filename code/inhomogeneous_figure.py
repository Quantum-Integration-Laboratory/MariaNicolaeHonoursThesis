import numpy as np
import matplotlib as mpl
mpl.use('pgf')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 11.0
import matplotlib.pyplot as plt
from scipy import stats

lorentzian_fwhm = 1
lorentzian_range = 10
sigma = 10
z_range = 5

x_min = -sigma * z_range
x_max = sigma * z_range
x_gaussian = np.linspace(x_min, x_max, 1001)
y_gaussian =  np.exp(-1/2*x_gaussian**2/sigma**2)

x_min_lorentzian = -lorentzian_range * lorentzian_fwhm
x_max_lorentzian = lorentzian_range * lorentzian_fwhm
x_lorentzian = np.linspace(x_min_lorentzian, x_max_lorentzian, 101)
y_lorentzian = 1 / (1 + 4*x_lorentzian**2/lorentzian_fwhm**2)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
for p in np.linspace(0.005, 0.995, 20):
    z = stats.norm.ppf(p)
    dx = z * sigma
    ax.fill_between(x_lorentzian-dx, y_lorentzian, color='black', alpha=0.3, edgecolor='none')
ax.plot(x_gaussian, y_gaussian, color='black', ls='dashed')
ax.set_xticks([])
ax.set_xlabel('Frequency')
ax.set_xlim(x_min, x_max)
ax.set_yticks([])
ax.set_ylabel('Absorption')
ax.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig('latex-build/inhomogeneous-broadening.pgf', format='pgf')
