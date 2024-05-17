import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 8.0
import pandas

df = pandas.read_csv('data/4lt_power_calibration.csv')
P_dBm = df['Input Power (dBm)']
P_waveguide = df['Waveguide Power (uW)']
P_in = 1e3 * 10.0**(P_dBm/10.0)
[efficiency, b] = np.polyfit(P_in, P_waveguide, 1)

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(P_in, P_waveguide, label='Data', edgecolor='C0', facecolor='none', zorder=100)
    (x_min, x_max) = ax.get_xlim()
    x = np.array([x_min, x_max])
    y = efficiency*x+b
    ax.plot(x, y, c='black', linestyle='dashed', label=f'Fit ($\\eta={efficiency:0.3f}$)')
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel('Input Power ($\\mu$W)')
    ax.set_ylabel('Waveguide Power ($\\mu$W)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('latex-build/4lt-power-calibration.png', dpi=600)
