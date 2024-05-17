import numpy as np
from scipy import integrate, optimize, stats
import matplotlib.pyplot as plt
import sympy

def getL():
    # symbols
    s12 = sympy.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    s13 = sympy.Matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    s23 = sympy.Matrix([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    s21 = s12.H
    s31 = s13.H
    s32 = s23.H
    s11 = s12*s21
    s22 = s21*s12
    s33 = s31*s13
    
    delta_mu = sympy.symbols('delta_mu', real=True)
    delta_s = sympy.symbols('delta_s', real=True)
    gamma_mu = sympy.symbols('gamma_mu', real=True)
    gamma31, gamma32 = sympy.symbols('gamma3(1:3)', real=True)
    gamma2d, gamma3d = sympy.symbols('gamma(2:4)d', real=True)
    nbath = sympy.symbols('nbath', real=True)
    Omega_mu = sympy.symbols('Omega_mu', complex=True)
    Omega_o = sympy.symbols('Omega_o', complex=True)
    A = sympy.symbols('A', complex=True)

    def master_equation_rhs(rho):
        H = Omega_o*s32 + Omega_mu*s21 + A*s31
        H = H + H.H
        H = H + delta_mu*s22 + delta_s*s33
    
        L21 = gamma_mu/2 * (nbath+1) * (2*s12*rho*s21 - s22*rho - rho*s22)
        L12 = gamma_mu/2 * nbath * (2*s21*rho*s12 - s11*rho - rho*s11)
        L32 = gamma32/2 * (2*s23*rho*s32 - s33*rho - rho*s33)
        L31 = gamma31/2 * (2*s13*rho*s31 - s33*rho - rho*s33)
        L22 = gamma2d/2 * (2*s22*rho*s22 - s22*s22*rho - rho*s22*s22)
        L33 = gamma3d/2 * (2*s33*rho*s33 - s33*s33*rho - rho*s33*s33)
        loss = L21 + L12 + L32 + L31 + L22 + L33
        
        return -sympy.I*(H*rho - rho*H) + loss

    # obtain matrix representation of differential operator L
    L = sympy.zeros(9)
    for i in range(3):
        for j in range(3):
            rho = sympy.zeros(3)
            rho[i,j] = 1
            Lcol = master_equation_rhs(rho)
            col = 3*i+j
            for ip in range(3):
                for jp in range(3):
                    row = 3*ip+jp
                    L[row,col] = Lcol[ip,jp]

    # replace first row with row computing the trace of rho
    L[0,:] = sympy.Matrix([1, 0, 0, 0, 1, 0, 0, 0, 1]).T

    # lambdify
    args = (delta_mu, delta_s, gamma_mu, gamma31, gamma32, gamma2d, gamma3d, nbath, Omega_mu, Omega_o, A)
    Lfunc = sympy.lambdify(args, L, 'numpy')
    
    return L, Lfunc

L, Lfunc = getL()

# fundamental constants
hbar = 1.05457e-34
kB = 1.380649e-23
c = 299792458
eps0 = 8.854187817e-12
mu0 = 4*np.pi*1e-7
muB = 9.274009994e-24

def rho_steady_state_many_args(delta_mu, delta_s, gamma_mu, gamma31, gamma32, gamma2d, gamma3d, nbath, Omega_mu, Omega_o, A):
    Lmatrix = Lfunc(
        delta_mu=delta_mu,
        delta_s=delta_s,
        gamma_mu=gamma_mu,
        gamma31=gamma31,
        gamma32=gamma32,
        gamma2d=gamma2d,
        gamma3d=gamma3d,
        nbath=nbath,
        Omega_mu=Omega_mu,
        Omega_o=Omega_o,
        A=A
    )
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    x = np.linalg.solve(Lmatrix, b)
    rho = np.array([[x[0], x[1], x[2]],
                    [x[3], x[4], x[5]],
                    [x[6], x[7], x[8]]])
    return rho

def Omega_mu_from_Pin(Pin, omega_mu):
    V_mu_cavity = Vsample / fill_factor
    Pmu = 1e-3 * 10**(Pin/10)
    Q = omega_mu / (kappa_mi + 2*kappa_mc)
    S21 = 4 * kappa_mc**2 / (kappa_mi + 2*kappa_mc)**2
    energy_mu_cavity = 2*Pmu*Q*np.sqrt(S21) / omega_mu
    Bmu = np.sqrt(mu0*energy_mu_cavity / (2*V_mu_cavity))
    return -mu12*Bmu / hbar

def Omega_o_from_Pin(Pin, omega_o):
    Po = 1e-3 * 10**(Pin/10)
    pflux = Po / (hbar*omega_o)
    n_in = 4*pflux*kappa_oc / (kappa_oc+kappa_oi)**2
    Sspot = np.pi * Woc**2
    V_o_cavity = (Sspot*Loc + Sspot*Lsample*nYSO**3) / 2
    Eo = np.sqrt(n_in*hbar*omega_o / (2*eps0*V_o_cavity))
    return -d23*Eo / hbar

def rho_steady_state(Pin_mu, Pin_o, delta_mu, delta_o, a):
    omega_mu = omega_12 - delta_mu
    nbath = 1 / (np.exp(hbar*(omega_mu)/(kB*T))-1)
    rho = rho_steady_state_many_args(
        delta_mu = delta_mu,
        delta_s = delta_o - delta_mu,
        gamma_mu = 1/tau2 * 1/(nbath+1),
        gamma31 = 1/tau3 * d13**2 / (d13**2 + d23**2),
        gamma32 = 1/tau3 * d23**2 / (d13**2 + d23**2),
        gamma2d = 1e6,
        gamma3d = 1e6,
        nbath = nbath,
        Omega_mu = Omega_mu_from_Pin(Pin_mu, omega_12 - delta_mu),
        Omega_o = Omega_o_from_Pin(Pin_o, omega_23 - delta_o),
        A = g*a
    )
    return rho

def rho13_steady_state_ensemble(Pin_mu, Pin_o, mean_delta_mu, mean_delta_s, a):
    standard_norm = lambda z: np.exp(-z**2/2) / np.sqrt(2*np.pi)
    zrange = 3

    def ensemble_integrand(z_mu, z_s):
        delta_mu = mean_delta_mu + w12*z_mu
        delta_s = mean_delta_mu + w13*z_s
        envelope = standard_norm(z_mu)*standard_norm(z_s) / (w12*w13)
        rho13 = rho_steady_state(Pin_mu, Pin_o, delta_mu, delta_s, a)[0,2]
        jacobian = w12 * w13
        return envelope * rho13 * jacobian

    real_integrand = lambda z_mu, z_s: np.real(ensemble_integrand(z_mu, z_s))
    imag_integrand = lambda z_mu, z_s: np.imag(ensemble_integrand(z_mu, z_s))

    y_re, abserr_re = integrate.dblquad(real_integrand, -zrange, zrange, -zrange, zrange)
    y_im, abserr_im = integrate.dblquad(imag_integrand, -zrange, zrange, -zrange, zrange)
    y = y_re + 1j*y_im
    return y

def steady_a(Pin_mu, Pin_o, delta_oc, rescaling=1):
    def S13(a):
        return N*g*rho13_steady_state_ensemble(Pin_mu, Pin_o, 0, 0, a)
    
    def ffunc(a):
        return -1j*delta_oc*a - 1j*S13(a) - (kappa_oi+kappa_oc)*a/2
    
    def ffunc_R2toR2(a):
        [a_re, a_im] = a
        a = a_re + 1j*a_im
        fa = ffunc(a)
        fa *= rescaling
        return [np.real(fa), np.imag(fa)]
    
    result = optimize.root(ffunc_R2toR2, [0, 0])
    [a_re, a_im] = result.x
    return a_re + 1j*a_im, result

if __name__ == '__main__':
    # recreate Figure 4c from Fernandez-Gonzalvo et. al. 2019
    # (Phys. Rev. A 100, 033807)

    nYSO = 1.76 # refractive index of YSO
    T = 4.6 # experiment temperature
    N = 1.28e15 # erbium number density
    g = 51.9 # s13 to optical coupling

    omega_12 = 2*np.pi*5.186e9
    omega_23 = 2*np.pi*195113.36e9 # 1536.504 nm
    w12 = 2*np.pi*25e6
    w13 = 2*np.pi*170e6
    mu12 = 4.3803*muB
    d13 = 1.63e-32
    d23 = 1.15e-32
    tau2 = 1e-3
    tau3 = 11e-3

    kappa_mi = 2*np.pi*650e3
    kappa_mc = 2*np.pi*70e3
    kappa_oi = 2*np.pi*1.7e6
    kappa_oc = 2*np.pi*7.95e6

    Woc = 0.6e-3
    Loc = 49.5e-3
    fill_factor = 0.8 # fraction of microwave cavity filled by sample

    dsample = 5e-3
    Lsample = 12e-3
    Vsample = np.pi * dsample**2 * Lsample / 4

    Pin_mu = 0
    Pin_o = 10.4135

    a, _ = steady_a(-20, Pin_o, 0, rescaling=1e-6)

    delta_mu_v = 30e6 * np.linspace(-2*np.pi, 2*np.pi, 101) / (2*np.pi)
    delta_o_v = 50e6 * np.linspace(-2*np.pi, 2*np.pi, 101) / (2*np.pi)
    delta_mu, delta_o = np.meshgrid(delta_mu_v, delta_o_v)

    def steady_pop(Pin_mu, Pin_o, delta_mu, delta_o, a):
        rho = rho_steady_state(Pin_mu, Pin_o, delta_mu, delta_o, a)
        return np.real(np.diag(rho))

    sig = '(),(),(),(),()->(n)'
    pop = np.vectorize(steady_pop, signature=sig)(-20, Pin_o, delta_mu, delta_o, a)

    p11 = pop[:,:,0]
    p33 = pop[:,:,2]
    mhz = 1e6

    fig, ax = plt.subplots(1, 1)
    ax.pcolormesh(delta_o/mhz, delta_mu/mhz, p11-p33, vmin=0, vmax=0.05)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_title('rho11 - rho33 (Mine)')
    ax.set_xlabel('delta_o (rad MHz)')
    ax.set_ylabel('delta_mu (rad MHz)')
    plt.show()
