import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 8.0
import tlt_double_cavity

hbar = 1.05457e-34
kB = 1.380649e-23
eps0 = 8.8541878128e-12
mu0 = 1.25663706212e-6
c = 299792458

T = 0.1

g_o = 51.9
g_mu = 1.04

tau_2 = 11
tau_3 = 11e-3
gamma_2d = 1e6
gamma_3d = 1e6
gamma_mui = 2*np.pi*650e3
gamma_muc = 2*np.pi*1.5e6
gamma_oi = 2*np.pi*7.95e6
gamma_oc = 2*np.pi*1.7e6

omega_12 = 2*np.pi*5.186e9
omega_23 = 2*np.pi*195113.36e9 # 1536.504 nm

sigma_o = 2*np.pi*419e6
sigma_mu = 2*np.pi*5e6

d_13 = 1.63e-32
d_23 = 1.15e-32

def Omega_from_laser(P, diameter):
    area = np.pi*diameter**2 / 4
    E = np.sqrt(2*mu0*c*P / area)
    Omega = d_23*E / hbar
    return Omega

def in_from_power(P, omega):
    return np.sqrt(P / (hbar*omega))

def dBm_to_W(dBm):
    return 1e-3 * 10.0**(dBm/10.0)

def N_mu_func(N, delta_amu, T):
    exp = np.exp(hbar*(omega_12-delta_amu)/kB*T)
    return (exp-1)/(exp+1)

def compute_decay_rates(T, delta_amu):
    omega_cmu = omega_12 - delta_amu
    n_bath = 1 / (np.exp(hbar*(omega_cmu)/(kB*T))-1)
    gamma_12 = 1/tau_2 * 1/(n_bath+1)
    gamma_13 = 1/tau_3 * d_13**2/(d_13**2+d_23**2)
    gamma_23 = 1/tau_3 * d_23**2/(d_13**2+d_23**2)
    return n_bath, gamma_12, gamma_13, gamma_23

pump_powers = [1e-6, 100e-3]
microwave_dBms = [5, -75]
laser_diameter = 0.1e-3
alpha_in = 0
delta_mu = 0
delta_o = 0
delta_amu = 0
delta_ao = 0
N = 4.8e16
N_o = 2.2e15
N_mu = N_mu_func(N, delta_amu, T)
n_bath, gamma_12, gamma_13, gamma_23 = compute_decay_rates(T, delta_amu)

fig, axs = plt.subplots(2, 2, figsize=(6.5, 5))
for pump_power, microwave_dBm, ax_row in zip(pump_powers, microwave_dBms, axs):
    Omega = Omega_from_laser(pump_power, laser_diameter)
    beta_in = in_from_power(dBm_to_W(microwave_dBm), omega_12-delta_mu)
    (alpha, beta), _ = tlt_double_cavity.alpha_beta_steady_state(
        N_o=N_o,
        N_mu=N_mu,
        g_mu=g_mu,
        g_o=g_o,
        Omega=Omega,
        gamma_12=gamma_12,
        gamma_13=gamma_13,
        gamma_23=gamma_23,
        gamma_2d=gamma_2d,
        gamma_3d=gamma_3d,
        gamma_mui=gamma_mui,
        gamma_muc=gamma_muc,
        gamma_oi=gamma_oi,
        gamma_oc=gamma_oc,
        n_bath=n_bath,
        delta_mu=delta_mu,
        delta_o=delta_o,
        delta_amu=delta_amu,
        delta_ao=delta_ao,
        sigma_mu=sigma_mu,
        sigma_o=sigma_o,
        alpha_in=alpha_in,
        beta_in=beta_in
    )

    delta_amu_grid, delta_ao_grid = np.meshgrid(np.linspace(-100e6, 100e6, 101), np.linspace(-100e6, 100e6, 101))
    rho12 = np.zeros_like(delta_amu_grid, dtype=complex)
    rho13 = np.zeros_like(delta_amu_grid, dtype=complex)
    for i in range(rho13.shape[0]):
        for j in range(rho13.shape[1]):
            d_ao = delta_ao_grid[i,j]
            d_amu = delta_amu_grid[i,j]
            n_bath, gamma_12, gamma_13, gamma_23 = compute_decay_rates(T, d_amu)
            rho = tlt_double_cavity.rho_steady_state(
                g_mu=g_mu,
                g_o=g_o,
                Omega=Omega,
                gamma_12=gamma_12,
                gamma_13=gamma_13,
                gamma_23=gamma_23,
                gamma_2d=gamma_2d,
                gamma_3d=gamma_3d,
                n_bath=n_bath,
                delta_mu=delta_mu,
                delta_o=delta_o,
                delta_amu=d_amu,
                delta_ao=d_ao,
                alpha=alpha,
                beta=beta
            )
            rho12[i,j] = rho[0,1]
            rho13[i,j] = rho[0,2]

    ax1, ax2 = ax_row

    mappable = ax1.pcolormesh(delta_amu_grid/1e6, delta_ao_grid/1e6, np.abs(rho12))
    fig.colorbar(mappable=mappable, ax=ax1)
    ax1.set_aspect('equal')
    ax1.set_xlabel('$\\delta_{\\mu}-\\delta_{c\\mu}$ (MHz)')
    ax1.set_ylabel('$\\delta_{o}-\\delta_{co}$ (MHz)')
    ax1.set_title('$|\\rho_{12}|$')


    mappable = ax2.pcolormesh(delta_amu_grid/1e6, delta_ao_grid/1e6, np.abs(rho13))
    fig.colorbar(mappable=mappable, ax=ax2)
    ax2.set_aspect('equal')
    ax2.set_xlabel('$\\delta_{\\mu}-\\delta_{c\\mu}$ (MHz)')
    ax2.set_ylabel('$\\delta_{o}-\\delta_{co}$ (MHz)')
    ax2.set_title('$|\\rho_{13}|$')

fig.tight_layout()
fig.savefig('latex-build/3lt-replication.png', dpi=600)
