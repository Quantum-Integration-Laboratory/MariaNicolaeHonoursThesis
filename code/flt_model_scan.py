import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import flt_model
import flt_experiment_parameters
mpl.rcParams['font.size'] = 8.0

hbar = 1.054571817e-34
omega_23 = 2*np.pi*304501.0e9 # rough

def signal_scan_parameter_dict(params):
    # size and resolution
    mu_freq_min = 3345
    mu_freq_max = 3395
    mu_freq_resolution = 101
    l_freq_min = 150
    l_freq_max = 1800
    l_freq_resolution = 101

    delta_mu, delta_B, signal_scan = flt_model.signal_scan(
        delta_mu_min=params['omega_34']-2*np.pi*mu_freq_max*1e6,
        delta_mu_max=params['omega_34']-2*np.pi*mu_freq_min*1e6,
        delta_mu_points=mu_freq_resolution,
        delta_B_min=2*np.pi*(1350-l_freq_max)*1e6,
        delta_B_max=2*np.pi*(1350-l_freq_min)*1e6,
        delta_B_points=l_freq_resolution,
        omega_12=params['omega_12'],
        omega_34=params['omega_34'],
        Omega_A=params['Omega_A'],
        Omega_B=params['Omega_B'],
        Omega_D=params['Omega_D'],
        Omega_E=params['Omega_E'],
        Omega_mu=params['Omega_mu'],
        tau_12=params['tau_12'],
        tau_34=params['tau_34'],
        gamma_13=params['gamma_13'],
        gamma_14=params['gamma_14'],
        gamma_23=params['gamma_23'],
        gamma_24=params['gamma_24'],
        gamma_2d=params['gamma_2d'],
        gamma_3d=params['gamma_3d'],
        gamma_4d=params['gamma_4d'],
        C_14=params['C_14'],
        C_24=params['C_24'],
        T=params['T'],
        Sigma=params['Sigma']
    )
    mu_freq = (params['omega_34']-delta_mu) / (2*np.pi*1e6)
    l_freq = 1350 - delta_B/(2*np.pi*1e6)
    omega_14 = params['omega_12'] + omega_23 + params['omega_34']
    signal_power = hbar*omega_14*params['N'] * signal_scan
    return mu_freq, l_freq, signal_power

# get experimental data
with open('data/4lt_lab_data.pkl', 'rb') as f:
    lab_data = pickle.load(f)

# noise model
rng = np.random.default_rng(seed=0)
random_data_dBm = np.ravel(lab_data['DR_power_scan3']['signals'][-40])
random_data_W = 1e-3 * 10.0**(random_data_dBm/10.0)
def random_noise(size=None):
    return rng.choice(random_data_W, size=size, replace=True)

def noisy_signal_scan_parameter_dict(params):
    mu_freq, l_freq, signal_power = signal_scan_parameter_dict(params)
    noised_signal_power = signal_power + random_noise(size=signal_power.shape)
    dBm = 10.0*np.log10(noised_signal_power/1e-3)
    return mu_freq, l_freq, dBm

# model scan parameter sets
pump_dBm = -4
params, mods = flt_experiment_parameters.get_parameters(pump_dBm)
mod_params = params.copy() | mods

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.5, 2))

lab_scan = lab_data['DR_power_scan3']['signals'][pump_dBm]
lab_scan_min = np.min(lab_scan)
lab_scan_max = np.max(lab_scan)
ax3.set_title('Experiment')
ax3.pcolormesh(
    lab_data['DR_power_scan3']['mu_freq'],
    lab_data['DR_power_scan3']['lfreq'],
    lab_scan)
ax3.set_xlabel('$\\omega_\\mu/2\\pi$ (MHz)')
ax3.set_ylabel('$\\omega_p/2\\pi$ Offset (MHz)')

mu_freq, l_freq, dBm = noisy_signal_scan_parameter_dict(params)
ax1.set_title('Model, Obeys Constraints')
ax1.pcolormesh(mu_freq, l_freq, dBm, vmin=lab_scan_min, vmax=lab_scan_max)
ax1.set_xlabel('$\\omega_\\mu/2\\pi$ (MHz)')
ax1.set_ylabel('$\\omega_p/2\\pi$ Offset (MHz)')

mu_freq, l_freq, dBm = noisy_signal_scan_parameter_dict(mod_params)
ax2.set_title('Model, Breaks Constraints')
ax2.pcolormesh(mu_freq, l_freq, dBm, vmin=lab_scan_min, vmax=lab_scan_max)
ax2.set_xlabel('$\\omega_\\mu/2\\pi$ (MHz)')
ax2.set_ylabel('$\\omega_p/2\\pi$ Offset (MHz)')

ax3.set_xlim(ax2.get_xlim())
ax3.set_ylim(ax2.get_ylim())

fig.tight_layout()
fig.savefig('latex-build/4lt-model-scan.png', dpi=600)
