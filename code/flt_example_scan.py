import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

mpl.rcParams['font.size'] = 8.0

with open('data/4lt_lab_data.pkl', 'rb') as f:
    lab_data = pickle.load(f)

l_freq = lab_data['DR_power_scan3']['lfreq']
mu_freq = lab_data['DR_power_scan3']['mu_freq']
low_dBm, high_dBm = -20, -4
power_low = lab_data['DR_power_scan3']['signals'][low_dBm]
power_high = lab_data['DR_power_scan3']['signals'][high_dBm]
vmin = np.min(power_high)
vmax = np.max(power_high)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
fig.suptitle('Transduction Experiment Data')

ax2.set_title(f'Optical Pump {high_dBm} dBm')
mappable = ax2.pcolormesh(mu_freq, l_freq, power_high, vmin=vmin, vmax=vmax)
fig.colorbar(mappable=mappable, ax=ax2, label='Transduced Signal (dBm)')
ax2.set_xlabel('$\\omega_\\mu/2\\pi$ (MHz)')
ax2.set_ylabel('$\\omega_p/2\\pi$ Offset (MHz)')
ymin, _ = ax2.get_ylim()
ax2.set_ylim(ymin, 1800)

ax1.set_title(f'Optical Pump {low_dBm} dBm')
mappable = ax1.pcolormesh(mu_freq, l_freq, power_low, vmin=vmin, vmax=vmax)
fig.colorbar(mappable=mappable, ax=ax1, label='Transduced Signal (dBm)')
ax1.set_xlabel('$\\omega_\\mu/2\\pi$ (MHz)')
ax1.set_ylabel('$\\omega_p/2\\pi$ Offset (MHz)')
ymin, _ = ax1.get_ylim()
ax1.set_ylim(ymin, 1800)

fig.tight_layout()
fig.savefig('latex-build/4lt-example-scan.png', dpi=600)
