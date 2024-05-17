import numpy as np
import matplotlib.pyplot as plt

# GHz/T in hbar=1 units
mu_B = 2*np.pi*1.39962449361e10

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, 1j], [-1j, 0]], dtype=complex)
sigma_z = np.array([[-1, 0], [0, 1]], dtype=complex)

# tensor product order: electron spin, nuclear spin
Sx = np.kron(sigma_x/2, np.eye(2))
Sy = np.kron(sigma_y/2, np.eye(2))
Sz = np.kron(sigma_z/2, np.eye(2))
Ix = np.kron(np.eye(2), sigma_x/2)
Iy = np.kron(np.eye(2), sigma_y/2)
Iz = np.kron(np.eye(2), sigma_z/2)

# tensor product order: electronic, electron spin, nuclear spin
# sources are Ranon 1968 (Phys. Lett. A 28, 228)
# and Kindem et. al. 2018 (Phys. Rev. B 98, 024404)
g_perp_g = 0.85 # Ranon 1968
g_perp_e = 1.7 # Kindem et. al. 2018
g_para_g = -6.08 # Ranon 1968
g_para_e = 2.51 # Kindem et. al. 2018
A_perp_g = 2*np.pi*0.675e9 # Kindem et. al. 2018
A_perp_e = 2*np.pi*3.37e9 # Kindem et. al. 2018
A_para_g = 2*np.pi*-4.82e9 # Kindem et. al. 2018
A_para_e = 2*np.pi*4.86e9 # Kindem et. al. 2018

def hamiltonian(Bx, By, Bz, g_perp, g_para, A_perp, A_para):
    H_zeeman = mu_B * (g_perp*(Bx*Sx + By*Sy) + g_para*Bz*Sz)
    H_hyperfine = A_perp*(Ix@Sx + Iy@Sy) + A_para*Iz@Sz
    return H_zeeman + H_hyperfine

def hamiltonian_g(Bx, By, Bz, g_perp_g=g_perp_g, g_para_g=g_para_g,
                  A_perp_g=A_perp_g, A_para_g=A_para_g):
    return hamiltonian(Bx, By, Bz, g_perp_g, g_para_g, A_perp_g, A_para_g)

def hamiltonian_e(Bx, By, Bz, g_perp_e=g_perp_e, g_para_e=g_para_e,
                  A_perp_e=A_perp_e, A_para_e=A_para_e):
    return hamiltonian(Bx, By, Bz, g_perp_e, g_para_e, A_perp_e, A_para_e)
