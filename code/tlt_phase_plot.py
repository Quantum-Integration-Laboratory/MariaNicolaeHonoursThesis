import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 8.0
import pandas

df = pandas.read_csv('data/tlt_phases.csv')
df['total_phase'] = (df['arg(Omega)'] + df['arg(beta_in)'] - df['arg(alpha_out)']) % (2*np.pi)

get_rho_key = lambda is_rho_ji: '$\\rho_{ji}$' if is_rho_ji else '$\\rho_{ij}$'
total_phase_data = {get_rho_key(True): dict(), get_rho_key(False): dict()}
for _, row in df.iterrows():
    row_dict = row.to_dict()
    rho_key = get_rho_key(row_dict['rho_ji'])
    microwave_dBm = row_dict['Microwave dBm']
    if microwave_dBm not in total_phase_data[rho_key]:
        total_phase_data[rho_key][microwave_dBm] = []
    total_phase_data[rho_key][microwave_dBm].append(row_dict['total_phase'])

t = np.linspace(0, 2*np.pi, 1000)
x = np.cos(t)
y = np.sin(t)

fig, axs = plt.subplots(2, len(total_phase_data[get_rho_key(True)]), figsize=(4.5, 3))
fig.suptitle('$\\arg(\\Omega) + \\arg(\\beta_\\text{in}) - \\arg(\\alpha_\\text{out})$')
for (rho_key, single_phase_data), axs_row in zip(total_phase_data.items(), axs):
    for (microwave_dBm, total_phase), ax in zip(single_phase_data.items(), axs_row):
        ax.set_title(f'{rho_key}, $P_\\mu = {microwave_dBm}$ dBm')
        total_phase = np.array(total_phase)
        zeros = np.zeros_like(total_phase)
        u = np.cos(total_phase)
        v = np.sin(total_phase)
        ax.plot(x, y, c='black')
        ax.scatter(0, 0, c='black', s=25, marker='o', zorder=100)
        ax.quiver(zeros, zeros, u, v, total_phase, cmap='hsv', clim=(0, 2*np.pi),
                  angles='xy', scale_units='xy', scale=1,
                  headlength=15, headaxislength=13.5, headwidth=9)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_axis_off()
fig.tight_layout()
fig.savefig('latex-build/3lt-phase.png', dpi=600)
