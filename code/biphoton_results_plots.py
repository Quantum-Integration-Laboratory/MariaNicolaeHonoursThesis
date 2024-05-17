import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 8.0
import pickle

with open('data/biphoton_super_atom.pkl', 'rb') as f:
    data = pickle.load(f)

gamma_oc = 2*np.pi*1.7e6
gamma_muc = 2*np.pi*1.5e6

alpha = 0.5 # opacity of second run
lw = 1 # thickness of first two plots

titles = ['Small Detuning (Non-Adiabatic Regime)',
          'Large Detuning (Adiabatic Regime)']
data_fnames = [
    ['data_summary_dt_10_ps_2.pkl', 'data_summary_dt_10_ps_3.pkl'],
    ['data_summary_dt_50_ps_large_det_2.pkl', 'data_summary_dt_50_ps_large_det_3.pkl']
]
figure_fnames = ['biphoton-results-small.png', 'biphoton-results-large.png']
xlims = [(-1, 50), (-1, 50)]
ax1_ylims = [(1e8, 1e24), (1e15, 1e25)]
ax2_ylims = [(1e-9, 1), (1e-6, 1)]

iterator = zip(titles, data_fnames, figure_fnames, xlims, ax1_ylims, ax2_ylims)
for title, (solid_fname, dotted_fname), figure_fname, xlim, ax1_ylim, ax2_ylim in iterator:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.5, 5.7))
    fig.suptitle(title)

    datum_solid = data[solid_fname]
    t_solid = datum_solid['t']
    alpha_solid = datum_solid['alpha']
    beta_solid = datum_solid['beta']
    S_solid = datum_solid['S']
    datum_dotted = data[dotted_fname]
    t_dotted = datum_dotted['t']
    alpha_dotted = datum_dotted['alpha']
    beta_dotted = datum_dotted['beta']
    S_dotted = datum_dotted['S']

    ax1.set_title('Generation Rate')
    ax1.plot(t_solid*1e6, gamma_oc*np.abs(alpha_solid)**2, lw=lw, c='blue', label='$|\\alpha_\\text{out}|^2$')
    ax1.plot(t_solid*1e6, gamma_muc*np.abs(beta_solid)**2, lw=lw, c='red', label='$|\\beta_\\text{out}|^2$')
    ax1.plot(t_dotted*1e6, gamma_oc*np.abs(alpha_dotted)**2, lw=lw, c='blue', alpha=alpha)
    ax1.plot(t_dotted*1e6, gamma_muc*np.abs(beta_dotted)**2, lw=lw, c='red', alpha=alpha)
    ax1.set_xlabel('Time ($\\mu$s)')
    ax1.set_xlim(xlim)
    ax1.set_ylabel('Photon Rate (Hz)')
    ax1.set_ylim(ax1_ylim)
    ax1.set_yscale('log')
    ax1.grid()
    ax1.legend()

    ax2.set_title('Coherences')
    ax2.plot(t_solid*1e6, np.abs(S_solid[:,0,1]), lw=lw, c='red', label='$|\\langle\\rho_{12}\\rangle_\\ell|$')
    ax2.plot(t_solid*1e6, np.abs(S_solid[:,1,2]), lw=lw, c='blue', label='$|\\langle\\rho_{23}\\rangle_\\ell|$')
    ax2.plot(t_solid*1e6, np.abs(S_solid[:,0,2]), lw=lw, c='purple', label='$|\\langle\\rho_{13}\\rangle_\\ell|$')
    ax2.plot(t_dotted*1e6, np.abs(S_dotted[:,0,1]), lw=lw, c='red', alpha=alpha)
    ax2.plot(t_dotted*1e6, np.abs(S_dotted[:,1,2]), lw=lw, c='blue', alpha=alpha)
    ax2.plot(t_dotted*1e6, np.abs(S_dotted[:,0,2]), lw=lw, c='purple', alpha=alpha)
    ax2.set_xlabel('Time ($\\mu$s)')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ax2_ylim)
    ax2.set_yscale('log')
    ax2.grid()
    ax2.legend()

    ax3.set_title('Populations')
    ax3.plot(t_solid[2:]*1e6, np.real(S_solid[2:,0,0]), c='C0', label='$|1\\rangle$')
    ax3.plot(t_solid[2:]*1e6, np.real(S_solid[2:,1,1]), c='C1', label='$|2\\rangle$')
    ax3.plot(t_solid[2:]*1e6, np.real(S_solid[2:,2,2]), c='C2', label='$|3\\rangle$')
    ax3.plot(t_dotted[2:]*1e6, np.real(S_dotted[2:,0,0]), c='C0', alpha=alpha)
    ax3.plot(t_dotted[2:]*1e6, np.real(S_dotted[2:,1,1]), c='C1', alpha=alpha)
    ax3.plot(t_dotted[2:]*1e6, np.real(S_dotted[2:,2,2]), c='C2', alpha=alpha)
    ax3.set_xlabel('Time ($\\mu$s)')
    ax3.set_xlim(xlim)
    ax3.set_ylim(0, 1)
    ax3.grid()
    ax3.legend()

    fig.tight_layout()
    fig.savefig(f'latex-build/{figure_fname}', dpi=600)
