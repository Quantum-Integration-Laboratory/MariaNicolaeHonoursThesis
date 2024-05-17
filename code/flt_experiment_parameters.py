import numpy as np
import YbYVO_spin_hamiltonian as spin
import flt_power_calibration

efficiency = flt_power_calibration.efficiency
Omega_ref = 2*np.pi*6e6
P_ref = 1.7e-6 #2e-6

Bz = 2.09e-3

def Omega_A_from_dBm(pump_dBm):
    P_waveguide = efficiency * 1e-3*10.0**(pump_dBm/10.0)
    Omega = np.sqrt(P_waveguide) * Omega_ref/np.sqrt(P_ref)
    return Omega

def get_S_mtx(Bx, By, Bz):
    H_g = spin.hamiltonian_g(Bx, By, Bz)
    H_e = spin.hamiltonian_e(Bx, By, Bz)
    eigvecs = np.zeros((4, 8), dtype=complex)
    eigvals_g, eigvecs[:,:4] = np.linalg.eigh(H_g)
    eigvals_e, eigvecs[:,4:] = np.linalg.eigh(H_e)
    spin_S_mtx = np.array([spin.Sx, spin.Sy, spin.Sz])
    S_mtx = eigvecs.conj().T @ spin_S_mtx @ eigvecs
    return eigvals_g, eigvals_e, S_mtx

def get_parameters(pump_dBm):
    eigvals_g, eigvals_e, S_mtx = get_S_mtx(0.0, 0.0, Bz)
    _, _, S_mtx_0 = get_S_mtx(0.0, 0.0, 0.0)

    # Z axis only because others are zero for the four-level system
    dA = S_mtx[2,3,4]
    dB = S_mtx[2,2,4]
    dD = S_mtx[2,3,5]
    dE = S_mtx[2,2,5]
    dA_0 = S_mtx_0[2,3,4]
    dE_0 = S_mtx_0[2,2,5]

    # atomic transition frequencies
    params = dict()
    params['omega_12'] = eigvals_g[3] - eigvals_g[2]
    params['omega_34'] = omega_34 = eigvals_e[1] - eigvals_e[0]

    # inhomogeneous broadening covariance matrix,
    # basis (delta_13, delta_34)
    params['Sigma'] = np.array([
        [ 200e6**2, -2.028e12],
        [-2.028e12, 0.13e6**2]
    ])

    # number of atoms
    params['N'] = 1e14

    # lifetimes
    tau_12_0 = 54e-3
    params['tau_12'] = tau_12_0 * np.abs(S_mtx_0[2,2,3]/S_mtx[2,2,3])**2
    params['tau_34'] = 10e-3

    # optical transition decay rates
    gamma_14_0 = 1.4e3
    gamma_23_0 = 1.3e3
    params['gamma_14'] = gamma_14_0*np.abs(dE/dE_0)**2
    params['gamma_23'] = gamma_23_0*np.abs(dA/dA_0)**2
    params['gamma_13'] = params['gamma_23']*np.abs(dB/dA)**2
    params['gamma_24'] = params['gamma_14']*np.abs(dD/dE)**2

    # waveguide coupling strengths
    params['C_14'] = np.exp(1j*np.angle(dE)) * np.sqrt(params['gamma_14'])
    params['C_24'] = np.exp(1j*np.angle(dD)) * np.sqrt(params['gamma_24'])

    # dephasing rates
    g_g_tensor = np.diag([spin.g_perp_g, spin.g_perp_g, spin.g_para_g])
    g_e_tensor = np.diag([spin.g_perp_e, spin.g_perp_e, spin.g_para_e])
    gamma_2d_prop = np.linalg.norm(g_g_tensor@S_mtx[:,3,3] - g_g_tensor@S_mtx[:,2,2])**2
    gamma_3d_prop = np.linalg.norm(g_e_tensor@S_mtx[:,4,4] - g_g_tensor@S_mtx[:,2,2])**2
    gamma_4d_prop = np.linalg.norm(g_e_tensor@S_mtx[:,5,5] - g_g_tensor@S_mtx[:,2,2])**2

    params['gamma_2d'] = 1e4
    params['gamma_3d'] = params['gamma_2d'] * gamma_3d_prop/gamma_2d_prop
    params['gamma_4d'] = params['gamma_2d'] * gamma_4d_prop/gamma_2d_prop

    # experimental conditions
    params['Omega_A'] = Omega_A_from_dBm(pump_dBm)
    params['Omega_B'] = params['Omega_A']*dB/dA
    params['Omega_D'] = 0
    params['Omega_E'] = 0
    params['Omega_mu'] = 2*np.pi*1e6
    params['T'] = 1

    # modify parameters
    mods = dict()
    mods['N'] = 1e10
    tau_12_0 = 1e-6
    hybridisation_ratio = 0.38

    # recompute dependent parameters
    hybridisation_small = hybridisation_ratio**2 / (1+hybridisation_ratio**2)
    hybridisation_large = 1 / (1+hybridisation_ratio**2)
    mods['tau_12'] = tau_12_0 * np.abs(S_mtx_0[2,2,3]/S_mtx[2,2,3])**2
    mods['gamma_14'] = gamma_14_0*hybridisation_large
    mods['gamma_23'] = gamma_23_0*hybridisation_large
    mods['gamma_13'] = gamma_23_0*hybridisation_small
    mods['gamma_24'] = gamma_14_0*hybridisation_small
    mods['C_14'] = np.exp(1j*np.angle(dE)) * np.sqrt(mods['gamma_14'])
    mods['C_24'] = np.exp(1j*np.angle(dD)) * np.sqrt(mods['gamma_24'])
    mods['Omega_B'] = params['Omega_A'] * hybridisation_ratio*np.exp(1j*np.angle(dB/dA))

    return params, mods

if __name__ == '__main__':
    pump_dBm = -4
    params, mods = get_parameters(pump_dBm)
    _, _, S_mtx = get_S_mtx(0.0, 0.0, Bz)
    print(f'Parameters\n{params}\n')
    print(f'Parameter modifications\n{mods}\n')
    print(f'Sx\n{S_mtx[0,2:6,2:6]}\n')
    print(f'Sy\n{S_mtx[1,2:6,2:6]}\n')
    print(f'Sz\n{S_mtx[2,2:6,2:6]}\n')
