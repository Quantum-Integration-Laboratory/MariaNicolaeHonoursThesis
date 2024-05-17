import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import flt_model
mpl.rcParams['font.size'] = 8.0

# atomic transition frequencies
omega_12 = 2*np.pi*0.698e9
omega_23 = 2*np.pi*304501.0e9
omega_34 = 2*np.pi*3.371e9
omega_13 = omega_12 + omega_23
omega_14 = omega_13 + omega_34
omega_24 = omega_23 + omega_34

# proportional dipole moments, calculated using spin Hamiltonian
dA = -0.99305509
dB = -0.1176503
dD = -0.1176503
dE = 0.99305509
dI = -1

# lifetimes
tau_12 = 1e-6
tau_34 = 10e-3

# optical transition decay rates
gamma_14 = 1.4e3
gamma_23 = 1.3e3
gamma_13 = gamma_23*np.abs(dB/dA)
gamma_24 = gamma_14*np.abs(dD/dE)

# waveguide coupling strengths
C_14 = np.sign(dE) * np.sqrt(gamma_14)
C_24 = np.sign(dD) * np.sqrt(gamma_24)

# dephasing rates
gamma_2d = 1e6
gamma_3d = 1e3
gamma_4d = 1e3

# experimental conditions
Omega_A = 2*np.pi*23e6
Omega_B = 0.38*Omega_A
Omega_D = 0
Omega_E = 0
Omega_mu = 2*np.pi*500e3
T = 1

def plot_atom_scan_intersections(ax, mu_freq_min, mu_freq_max, mu_freq_res,
                                 l_freq_min, l_freq_max, l_freq_res):
    _, _, inter_points, _, is_parallel = flt_model.find_curve_pixel_intersections(
        delta_mu_min=omega_34-2*np.pi*mu_freq_max*1e6,
        delta_mu_max=omega_34-2*np.pi*mu_freq_min*1e6,
        delta_mu_points=mu_freq_res,
        delta_B_min=2*np.pi*(1350-l_freq_max)*1e6,
        delta_B_max=2*np.pi*(1350-l_freq_min)*1e6,
        delta_B_points=l_freq_res,
        omega_12=omega_12,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu
    )

    delta_mu_points = np.array([point[0] for point in inter_points])
    delta_B_points = np.array([point[1] for point in inter_points])
    mu_freq_points = (omega_34-delta_mu_points) / (2*np.pi*1e6)
    l_freq_points = 1350 - delta_B_points/(2*np.pi*1e6)
    is_parallel = np.array(is_parallel)

    delta_mu, delta_B, atom_scan = flt_model.atom_scan(
        delta_mu_min=omega_34-2*np.pi*mu_freq_max*1e6,
        delta_mu_max=omega_34-2*np.pi*mu_freq_min*1e6,
        delta_mu_points=mu_freq_res,
        delta_B_min=2*np.pi*(1350-l_freq_max)*1e6,
        delta_B_max=2*np.pi*(1350-l_freq_min)*1e6,
        delta_B_points=l_freq_res,
        omega_12=omega_12,
        omega_34=omega_34,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu,
        tau_12=tau_12,
        tau_34=tau_34,
        gamma_13=gamma_13,
        gamma_14=gamma_14,
        gamma_23=gamma_23,
        gamma_24=gamma_24,
        gamma_2d=gamma_2d,
        gamma_3d=gamma_3d,
        gamma_4d=gamma_4d,
        C_14=C_14,
        C_24=C_24,
        T=T,
    )

    mu_freq = (omega_34-delta_mu) / (2*np.pi*1e6)
    l_freq = 1350 - delta_B/(2*np.pi*1e6)

    ax.pcolormesh(mu_freq, l_freq, np.log(atom_scan))
    ax.scatter(mu_freq_points[~is_parallel], l_freq_points[~is_parallel], s=15, edgecolor='black', facecolor='none')
    ax.scatter(mu_freq_points[is_parallel], l_freq_points[is_parallel], s=15, edgecolor='white', facecolor='none')
    ax.set_xlabel('$\\delta\'_\\mu$')
    ax.set_xticks([])
    ax.set_ylabel('$\\delta\'_p$')
    ax.set_yticks([])
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
plot_atom_scan_intersections(ax1, 3345, 3395, 31, 150, 1800, 31)
plot_atom_scan_intersections(ax2, 3345, 3345.001, 31, 657.57, 657.61, 31)
fig.tight_layout()
fig.savefig('latex-build/4lt-pixel-intersections.png', dpi=600)
