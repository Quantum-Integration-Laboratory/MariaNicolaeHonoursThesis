import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 8.0
import YbYVO_spin_hamiltonian as spin

def get_S_mtx(Bz):
    H_g = spin.hamiltonian_g(0.0, 0.0, Bz)
    H_e = spin.hamiltonian_e(0.0, 0.0, Bz)
    eigvecs = np.zeros((4, 8), dtype=complex)
    _, eigvecs[:,:4] = np.linalg.eigh(H_g)
    _, eigvecs[:,4:] = np.linalg.eigh(H_e)
    spin_S_mtx = np.array([spin.Sx, spin.Sy, spin.Sz])
    S_mtx = eigvecs.conj().T @ spin_S_mtx @ eigvecs
    return S_mtx

def hybridisation_ratio(Bz):
    S = get_S_mtx(Bz)
    d13 = np.linalg.norm(S[:,2,4])
    d23 = np.linalg.norm(S[:,3,4])
    return d13 / d23

B = np.linspace(0, 0.01, 1000)
r = [hybridisation_ratio(Bz) for Bz in B]

fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
ax.set_title('Hybridisation Ratio vs Magnetic Field Strength')
ax.plot(B*1e3, r)
ax.axvline(2.09, c='black', linestyle='dashed', label='$2.09$ mT')
ax.axhline(0.38, c='black', linestyle='dotted', label='$d_{13}/d_{23}=0.38$')
ax.grid()
ax.set_xlabel('$B_z$ (mT)')
ax.set_xlim(B[0]*1e3, B[-1]*1e3)
ax.set_ylabel('$d_{13}/d_{23}$')
ax.set_ylim(min(r), max(r))
ax.legend()
fig.tight_layout()
fig.savefig('latex-build/4lt-hybridisation-ratio.png', dpi=600)
