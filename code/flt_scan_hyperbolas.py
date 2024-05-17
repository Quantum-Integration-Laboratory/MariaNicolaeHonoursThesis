import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

mpl.rcParams['font.size'] = 8.0

with open('data/4lt_lab_data.pkl', 'rb') as f:
    lab_data = pickle.load(f)

l_freq = lab_data['DR_power_scan3']['lfreq']
mu_freq = lab_data['DR_power_scan3']['mu_freq']
power = lab_data['DR_power_scan3']['signals'][-4]

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.pcolormesh(mu_freq, l_freq, power)
ax.set_xlabel('$\\omega_\\mu/2\\pi$ (MHz)')
ax.set_ylabel('$\\omega_p/2\\pi$ Offset (MHz)')
xlim = ax.get_xlim()
ymin, _ = ax.get_ylim()

def plot_hyperbola(mu_min, mu_max, l_min, l_max, mu_centre, l_centre, vertex_dist, c='C0'):
    func = lambda mu: vertex_dist**2 / (8*(mu-mu_centre)) + l_centre
    if mu_min < mu_centre < mu_max:
        mu_intervals = [(mu_min, mu_centre), (mu_centre, mu_max)]
    else:
        mu_intervals = [(mu_min, mu_max)]
    if l_min < l_centre < l_max:
        l_intervals = [(l_min, l_centre), (l_centre, l_max)]
    else:
        l_intervals = [(l_min, l_max)]

    for mu_a, mu_b in mu_intervals:
        for l_a, l_b in l_intervals:
            mu_vals = np.linspace(mu_a, mu_b, 1000)
            l_vals = func(mu_vals)
            mask = (l_vals>=l_a) & (l_vals<=l_b)
            ax.plot(mu_vals[mask], l_vals[mask], c=c)

plot_hyperbola(3345, 3395, ymin, 1000, 3369.2, 652, 66, c='red')
plot_hyperbola(3367, 3373, 1250, 1450, 3370.1, 1350, 12.6, c='black')
ax.set_xlim(xlim)
ax.set_ylim(ymin, 1800)
fig.tight_layout()
fig.savefig('latex-build/4lt-scan-hyperbolas.png', dpi=600)
