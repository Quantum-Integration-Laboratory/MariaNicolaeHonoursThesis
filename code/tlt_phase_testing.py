import numpy as np
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

abs_Omega = Omega_from_laser(1e-6, 0.1e-3)
delta_mu = 0
delta_o = 0
delta_amu = 0
delta_ao = 0
alpha_in = 0
N = 4.8e16
N_o = 2.2e15

rng = np.random.default_rng(seed=0)
n_random_phases = 60

header_row = ['rho_ji', 'abs(Omega)', 'arg(Omega)',
              'Microwave dBm', 'abs(beta_in)', 'arg(beta_in)',
              'abs(alpha)', 'arg(alpha)', 'abs(beta)', 'arg(beta)',
              'abs(alpha_out)', 'arg(alpha_out)', 'abs(beta_out)', 'arg(beta_out)']
print(','.join(str(cell) for cell in header_row))
for use_rho_ji in [False, True]:
    for microwave_dBm in [-200, -75, 5]:
        abs_beta_in = in_from_power(dBm_to_W(microwave_dBm), omega_12-delta_mu)
        Omega_phases = 2*np.pi*rng.random(n_random_phases)
        beta_phases = 2*np.pi*rng.random(n_random_phases)
        for beta_phase, Omega_phase in zip(beta_phases, Omega_phases):
            beta_in = abs_beta_in * np.exp(1j*beta_phase)
            Omega = abs_Omega * np.exp(1j*Omega_phase)
            N_mu = N_mu_func(N, delta_amu, T)
            n_bath, gamma_12, gamma_13, gamma_23 = compute_decay_rates(T, delta_amu)
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
                beta_in=beta_in,
                use_rho_ji=use_rho_ji
            )
            alpha_out = -alpha_in + np.sqrt(gamma_oc)*alpha
            beta_out = -beta_in + np.sqrt(gamma_muc)*beta
            row = [use_rho_ji, abs_Omega, Omega_phase,
                   microwave_dBm, abs_beta_in, beta_phase,
                   np.abs(alpha), np.angle(alpha), np.abs(beta), np.angle(beta),
                   np.abs(alpha_out), np.angle(alpha_out),
                   np.abs(beta_out), np.angle(beta_out)]
            print(','.join(str(cell) for cell in row))
