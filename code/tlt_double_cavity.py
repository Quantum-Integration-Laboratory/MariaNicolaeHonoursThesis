import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize, stats
import sympy
import pickle
import itertools

gauss_lobatto_n = 20

legendre_coeffs = (0,)*(gauss_lobatto_n-1) + (1,)
roots = np.polynomial.legendre.Legendre(legendre_coeffs).deriv().roots()
gauss_lobatto_points = np.concatenate([[-1], roots, [1]])
gauss_lobatto_points = (gauss_lobatto_points+1) / 2
    
k = np.arange(gauss_lobatto_n)
i, j = np.indices((gauss_lobatto_n, gauss_lobatto_n))
M = gauss_lobatto_points[j]**i
b = 1/(k+1)
gauss_lobatto_weights = np.linalg.solve(M, b)

# flattening and unflattening arrays
unravel = np.array([
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8]
])

ravel_i = np.zeros(9, dtype=int)
ravel_j = np.zeros(9, dtype=int)
for i in range(3):
    for j in range(3):
        k = unravel[i,j]
        ravel_i[k] = i
        ravel_j[k] = j
ravel = (ravel_i, ravel_j)

def get_symbolic():
    # parameters
    g_o = sympy.symbols('g_o', real=True)
    g_mu = sympy.symbols('g_mu', real=True)
    Omega = sympy.symbols('Omega')
    alpha = sympy.symbols('alpha')
    beta = sympy.symbols('beta')
    delta_mu = sympy.symbols('delta_mu', real=True)
    delta_o = sympy.symbols('delta_o', real=True)
    delta_amu = sympy.symbols('delta_amu', real=True)
    delta_ao = sympy.symbols('delta_ao', real=True)
    gamma_12 = sympy.symbols('gamma_12', real=True)
    gamma_13 = sympy.symbols('gamma_13', real=True)
    gamma_23 = sympy.symbols('gamma_23', real=True)
    gamma_2d = sympy.symbols('gamma_2d', real=True)
    gamma_3d = sympy.symbols('gamma_3d', real=True)
    n_bath = sympy.symbols('n_bath', real=True)
    symbols = {
        'g_o': g_o,
        'g_mu': g_mu,
        'Omega': Omega,
        'alpha': alpha,
        'beta': beta,
        'delta_mu': delta_mu,
        'delta_o': delta_o,
        'delta_amu': delta_amu,
        'delta_ao': delta_ao,
        'gamma_12': gamma_12,
        'gamma_13': gamma_13,
        'gamma_23': gamma_23,
        'gamma_2d': gamma_2d,
        'gamma_3d': gamma_3d,
        'n_bath': n_bath
    }

    # matrix symbols
    s12 = sympy.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    s13 = sympy.Matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    s23 = sympy.Matrix([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    s21 = s12.H
    s31 = s13.H
    s32 = s23.H
    s11 = s12*s21
    s22 = s21*s12
    s33 = s31*s13

    # Hamiltonian
    H_0 = (delta_amu-delta_mu)*s22 + (delta_ao-delta_o)*s33 + Omega*s32 + Omega.conjugate()*s23
    H_alpha = g_o*s31
    H_alpha_c = g_o*s13
    H_beta = g_mu*s21
    H_beta_c = g_mu*s12
    H = H_0 + alpha*H_alpha + alpha.conjugate()*H_alpha_c + beta*H_beta + beta.conjugate()*H_beta_c

    # loss superoperator
    def loss_superoperator(rho):
        L21 = gamma_12/2 * (n_bath+1) * (2*s12*rho*s21 - s22*rho - rho*s22)
        L12 = gamma_12/2 * n_bath * (2*s21*rho*s12 - s11*rho - rho*s11)
        L32 = gamma_23/2 * (2*s23*rho*s32 - s33*rho - rho*s33)
        L31 = gamma_13/2 * (2*s13*rho*s31 - s33*rho - rho*s33)
        L22 = gamma_2d/2 * (2*s22*rho*s22 - s22*rho - rho*s22)
        L33 = gamma_3d/2 * (2*s33*rho*s33 - s33*rho - rho*s33)
        loss = L21 + L12 + L32 + L31 + L22 + L33
        return loss

    # get liouvillan superoperator matrix
    def liouvillan_matrix(H, loss=None):
        def master_equation(rho):
            master_operator = -sympy.I * (H*rho-rho*H)
            if loss is not None:
                master_operator += loss(rho)
            return master_operator

        L = sympy.zeros(9)
        for column, (icol, jcol) in enumerate(zip(*ravel)):
            # basis matrix
            rho = sympy.zeros(3)
            rho[icol,jcol] = 1

            # get column of supermatrix
            L_column = master_equation(rho)
            for row, (irow, jrow) in enumerate(zip(*ravel)):
                L[row,column] = L_column[irow,jrow]

        return L

    L_0 = liouvillan_matrix(H_0, loss=loss_superoperator)
    L_alpha = liouvillan_matrix(H_alpha)
    L_alpha_c = liouvillan_matrix(H_alpha_c)
    L_beta = liouvillan_matrix(H_beta)
    L_beta_c = liouvillan_matrix(H_beta_c)
    L = L_0 + alpha*L_alpha + alpha.conjugate()*L_alpha_c + beta*L_beta + beta.conjugate()*L_beta_c

    return {
        'symbols': symbols,
        'H': H,
        'L': L,
        'L_0': L_0,
        'L_alpha': L_alpha,
        'L_alpha_c': L_alpha_c,
        'L_beta': L_beta,
        'L_beta_c': L_beta_c
    }

symbolic = get_symbolic()

symbolic_args_full = (
    symbolic['symbols']['g_o'],
    symbolic['symbols']['g_mu'],
    symbolic['symbols']['Omega'],
    symbolic['symbols']['gamma_12'],
    symbolic['symbols']['gamma_13'],
    symbolic['symbols']['gamma_23'],
    symbolic['symbols']['gamma_2d'],
    symbolic['symbols']['gamma_3d'],
    symbolic['symbols']['n_bath'],
    symbolic['symbols']['delta_mu'],
    symbolic['symbols']['delta_o'],
    symbolic['symbols']['delta_amu'],
    symbolic['symbols']['delta_ao'],
    symbolic['symbols']['alpha'],
    symbolic['symbols']['beta']
)

symbolic_args_0 = (
    symbolic['symbols']['g_o'],
    symbolic['symbols']['g_mu'],
    symbolic['symbols']['Omega'],
    symbolic['symbols']['gamma_12'],
    symbolic['symbols']['gamma_13'],
    symbolic['symbols']['gamma_23'],
    symbolic['symbols']['gamma_2d'],
    symbolic['symbols']['gamma_3d'],
    symbolic['symbols']['n_bath'],
    symbolic['symbols']['delta_mu'],
    symbolic['symbols']['delta_o'],
    symbolic['symbols']['delta_amu'],
    symbolic['symbols']['delta_ao']
)

symbolic_args_H = (
    symbolic['symbols']['g_o'],
    symbolic['symbols']['g_mu'],
    symbolic['symbols']['Omega'],
    symbolic['symbols']['delta_mu'],
    symbolic['symbols']['delta_o'],
    symbolic['symbols']['delta_amu'],
    symbolic['symbols']['delta_ao'],
    symbolic['symbols']['alpha'],
    symbolic['symbols']['beta']
)

H = sympy.lambdify(symbolic_args_H, symbolic['H'], 'numpy')
L = sympy.lambdify(symbolic_args_full, symbolic['L'], 'numpy')
L_0 = sympy.lambdify(symbolic_args_0, symbolic['L_0'], 'numpy')
L_alpha = sympy.lambdify(symbolic['symbols']['g_o'], symbolic['L_alpha'], 'numpy')
L_alpha_c = sympy.lambdify(symbolic['symbols']['g_o'], symbolic['L_alpha_c'], 'numpy')
L_beta = sympy.lambdify(symbolic['symbols']['g_mu'], symbolic['L_beta'], 'numpy')
L_beta_c = sympy.lambdify(symbolic['symbols']['g_mu'], symbolic['L_beta_c'], 'numpy')

def rho_steady_state(g_o, g_mu, Omega, gamma_12, gamma_13,
                     gamma_23, gamma_2d, gamma_3d, n_bath,
                     delta_mu, delta_o, delta_amu, delta_ao, alpha, beta):
    L_matrix = L(
        g_o=g_o,
        g_mu=g_mu,
        Omega=Omega,
        gamma_12=gamma_12,
        gamma_13=gamma_13,
        gamma_23=gamma_23,
        gamma_2d=gamma_2d,
        gamma_3d=gamma_3d,
        n_bath=n_bath,
        delta_mu=delta_mu,
        delta_o=delta_o,
        delta_amu=delta_amu,
        delta_ao=delta_ao,
        alpha=alpha,
        beta=beta
    )
    L_matrix[0,:] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
    b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    rho = np.linalg.solve(L_matrix, b)
    return rho[unravel]

def rho_steady_state_ensemble(g_o, g_mu, Omega, gamma_12, gamma_13,
                              gamma_23, gamma_2d, gamma_3d, n_bath,
                              delta_mu, delta_o, delta_amu, delta_ao,
                              sigma_mu, sigma_o, alpha, beta):
    # degenerate dressed state detuning
    def minor_det(M, i, j):
            return M[i,i]*M[j,j] - M[i,j]*M[j,i]
    
    def H_disc(delta_amu, delta_ao):
        H_matrix = H(
            g_mu=g_mu,
            g_o=g_o,
            Omega=Omega,
            alpha=alpha,
            beta=beta,
            delta_mu=delta_mu,
            delta_o=delta_o,
            delta_amu=delta_amu,
            delta_ao=delta_ao
        )
        a = -1
        b = np.trace(H_matrix)
        c = -(minor_det(H_matrix, 0, 1)
            + minor_det(H_matrix, 0, 2)
            + minor_det(H_matrix, 1, 2))
        d = np.linalg.det(H_matrix)
        Delta = 18*a*b*c*d - 4*b**3*d + b*b*c*c - 4*a*c**3 - 27*a*a*d*d
        return np.abs(Delta)
        
    def delta_amu_degenerate(delta_ao):
        # get close guess
        if delta_ao == delta_o:
            delta_amu0 = delta_mu
        elif np.abs(beta) < np.abs(Omega):
            delta_amu0 = -np.abs(Omega)**2/(delta_ao-delta_o) + delta_ao - delta_o + delta_mu
        else:
            delta_amu0 = np.abs(g_mu*beta)**2/(delta_ao-delta_o) + delta_mu
        
        f = lambda d_amu: H_disc(d_amu, delta_ao)
        result = optimize.minimize_scalar(f, bracket=(delta_amu0-sigma_mu, delta_amu0+sigma_mu))
        return result.x

    # gauss-lobatto integration
    def gauss_lobatto_nodes_weights(a, b):
        nodes = a + gauss_lobatto_points*(b-a)
        weights = gauss_lobatto_weights*(b-a)
        return nodes, weights

    def composite_gauss_lobatto_nodes_weights(bounds):
        bounds = np.sort(bounds)
        nodes = np.array([], dtype=float)
        weights = np.array([], dtype=float)
        for a, b in zip(bounds[:-1], bounds[1:]):
            new_nodes, new_weights = gauss_lobatto_nodes_weights(a, b)
            nodes = np.concatenate([nodes, new_nodes])
            weights = np.concatenate([weights, new_weights])
        return nodes, weights

    # rho helper
    def rho_fewer_args(d_ao, d_amu):
        return rho_steady_state(
            g_o=g_o,
            g_mu=g_mu,
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

    # generate outer integral nodes and weights
    bounds = [
        delta_ao,
        delta_ao - sigma_o,
        delta_ao + sigma_o,
        delta_ao - 3*sigma_o,
        delta_ao + 3*sigma_o,
        delta_ao - 10*sigma_o,
        delta_ao + 10*sigma_o,
        delta_o,
        delta_o - gamma_3d,
        delta_o + gamma_3d,
        delta_o - 5*gamma_3d,
        delta_o + 5*gamma_3d
    ]
    d_ao_nodes, d_ao_weights = composite_gauss_lobatto_nodes_weights(bounds)

    # outer integral envelope function
    z_ao_nodes = (d_ao_nodes-delta_ao) / sigma_o
    G_ao_nodes = stats.norm.pdf(z_ao_nodes) / sigma_o

    # perform integrals
    rho_integral = np.zeros((3, 3), dtype=complex)
    for d_ao, d_ao_w, G_ao in zip(d_ao_nodes, d_ao_weights, G_ao_nodes):
        # generate inner integral nodes and weights
        bounds = [
            delta_amu,
            delta_amu - sigma_mu,
            delta_amu + sigma_mu,
            delta_amu - 3*sigma_mu,
            delta_amu + 3*sigma_mu,
            delta_amu - 10*sigma_mu,
            delta_amu + 10*sigma_mu,
            delta_mu,
            delta_mu - gamma_2d,
            delta_mu + gamma_2d,
            delta_mu - 5*gamma_2d,
            delta_mu + 5*gamma_2d,
            delta_amu_degenerate(d_ao)
        ]
        d_amu_nodes, d_amu_weights = composite_gauss_lobatto_nodes_weights(bounds)

        # inner integral envelope function
        z_amu_nodes = (d_amu_nodes-delta_amu) / sigma_mu
        G_amu_nodes = stats.norm.pdf(z_amu_nodes) / sigma_mu

        # perform inner integral
        for d_amu, d_amu_w, G_amu in zip(d_amu_nodes, d_amu_weights, G_amu_nodes):
            rho = rho_fewer_args(d_ao, d_amu)
            rho_integral += G_ao * G_amu * rho * d_ao_w * d_amu_w

    return rho_integral

def alpha_beta_langevin_differential(N_o, N_mu, g_o, g_mu, Omega, gamma_12, gamma_13,
                                     gamma_23, gamma_2d, gamma_3d, gamma_muc, gamma_mui,
                                     gamma_oc, gamma_oi, n_bath, delta_mu, delta_o, delta_amu,
                                     delta_ao, sigma_mu, sigma_o, alpha, beta, alpha_in, beta_in,
                                     use_rho_ji=True):
    rho_steady = rho_steady_state_ensemble(
        g_o=g_o,
        g_mu=g_mu,
        Omega=Omega,
        gamma_12=gamma_12,
        gamma_13=gamma_13,
        gamma_23=gamma_23,
        gamma_2d=gamma_2d,
        gamma_3d=gamma_3d,
        n_bath=n_bath,
        delta_mu=delta_mu,
        delta_o=delta_o,
        delta_amu=delta_amu,
        delta_ao=delta_ao,
        sigma_mu=sigma_mu,
        sigma_o=sigma_o,
        alpha=alpha,
        beta=beta
    )
    S12 = N_mu * g_mu * (rho_steady[1,0] if use_rho_ji else rho_steady[0,1])
    S13 = N_o * g_o * (rho_steady[2,0] if use_rho_ji else rho_steady[0,2])
    alpha_diff = 1j*delta_o*alpha - 1j*S13 - (gamma_oc+gamma_oi)*alpha/2 + np.sqrt(gamma_oc)*alpha_in
    beta_diff = 1j*delta_mu*beta - 1j*S12 - (gamma_muc+gamma_mui)*beta/2 + np.sqrt(gamma_muc)*beta_in
    return alpha_diff, beta_diff

def alpha_beta_steady_state(N_o, N_mu, g_o, g_mu, Omega, gamma_12, gamma_13,
                            gamma_23, gamma_2d, gamma_3d, gamma_muc, gamma_mui,
                            gamma_oc, gamma_oi, n_bath, delta_mu, delta_o, delta_amu,
                            delta_ao, sigma_mu, sigma_o, alpha_in, beta_in, use_rho_ji=True):
    def root_function(alpha_beta_vec):
        alpha_r = alpha_beta_vec[0]
        alpha_i = alpha_beta_vec[1]
        beta_r = alpha_beta_vec[2]
        beta_i = alpha_beta_vec[3]
        alpha = alpha_r + 1j*alpha_i
        beta = beta_r + 1j*beta_i
        alpha_res, beta_res = alpha_beta_langevin_differential(
            N_o=N_o,
            N_mu=N_mu,
            g_o=g_o,
            g_mu=g_mu,
            Omega=Omega,
            gamma_12=gamma_12,
            gamma_13=gamma_13,
            gamma_23=gamma_23,
            gamma_2d=gamma_2d,
            gamma_3d=gamma_3d,
            gamma_muc=gamma_muc,
            gamma_mui=gamma_mui,
            gamma_oc=gamma_oc,
            gamma_oi=gamma_oi,
            n_bath=n_bath,
            delta_mu=delta_mu,
            delta_o=delta_o,
            delta_amu=delta_amu,
            delta_ao=delta_ao,
            sigma_mu=sigma_mu,
            sigma_o=sigma_o,
            alpha=alpha,
            beta=beta,
            alpha_in=alpha_in,
            beta_in=beta_in,
            use_rho_ji=use_rho_ji
        )
        alpha_beta_res_vec = np.zeros(4, dtype=float)
        alpha_beta_res_vec[0] = np.real(alpha_res)
        alpha_beta_res_vec[1] = np.imag(alpha_res)
        alpha_beta_res_vec[2] = np.real(beta_res)
        alpha_beta_res_vec[3] = np.imag(beta_res)
        return alpha_beta_res_vec

    # initial guess for root-finding; root for zero atomic interaction
    alpha_0 = alpha_in*np.sqrt(gamma_oc) / ((gamma_oi+gamma_oc)/2 - 1j*delta_o)
    beta_0 = beta_in*np.sqrt(gamma_muc) / ((gamma_mui+gamma_muc)/2 - 1j*delta_mu)
    
    # perform root finding
    x0 = np.zeros(4, dtype=float)
    x0[0] = np.real(alpha_0)
    x0[1] = np.imag(alpha_0)
    x0[2] = np.real(beta_0)
    x0[3] = np.imag(beta_0)
    result = optimize.root(root_function, x0=x0, tol=1e-12)

    # restore result to complex numbers
    alpha_r = result.x[0]
    alpha_i = result.x[1]
    beta_r = result.x[2]
    beta_i = result.x[3]
    alpha = alpha_r + 1j*alpha_i
    beta = beta_r + 1j*beta_i
    
    return (alpha, beta), result
