import numpy as np
from scipy import signal
import sympy
import pickle
import os

# fundamental constants

hbar = 1.054571817e-34
kB = 1.380649e-23

# flattening and unflattening of density matrices

unflatten = np.array([
    [ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11],
    [12, 13, 14, 15]
])

flatten_i = np.zeros(16, dtype=int)
flatten_j = np.zeros(16, dtype=int)
for i in range(4):
    for j in range(4):
        k = unflatten[i,j]
        flatten_i[k] = i
        flatten_j[k] = j
flatten = (flatten_i, flatten_j)

# computer algebra for Liouvillan matrix

s11 = sympy.Matrix([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
s12 = sympy.Matrix([[0,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
s13 = sympy.Matrix([[0,0,1,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
s14 = sympy.Matrix([[0,0,0,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
s21 = s12.T
s31 = s13.T
s41 = s14.T
s22 = s21*s12
s23 = s21*s13
s24 = s21*s14
s32 = s23.T
s42 = s24.T
s33 = s31*s13
s34 = s31*s14
s43 = s34.T
s44 = s43*s34

omega12 = sympy.symbols('omega_12', real=True)
delta_B = sympy.symbols('delta_B', real=True)
delta_mu = sympy.symbols('delta_mu', real=True)
Omega_mu = sympy.symbols('Omega_mu')
Omega_A = sympy.symbols('Omega_A')
Omega_B = sympy.symbols('Omega_B')
Omega_D = sympy.symbols('Omega_D')
Omega_E = sympy.symbols('Omega_E')

gamma12, gamma13, gamma14 = sympy.symbols('gamma_1(2:5)', real=True)
gamma23, gamma24 = sympy.symbols('gamma_2(3:5)', real=True)
gamma34 = sympy.symbols('gamma_34', real=True)
gamma2d, gamma3d, gamma4d = sympy.symbols('gamma_(2:5)d', real=True)
nbath_12 = sympy.symbols('n_12', real=True)
nbath_34 = sympy.symbols('n_34', real=True)

H = Omega_A*s32 + Omega_B*s31 + Omega_D*s42 + Omega_E*s41 + Omega_mu*s43
H += H.H
H += omega12*s22 + delta_B*s33 + (delta_B+delta_mu)*s44

def loss_superoperator(rho):
    L12 = gamma12*(nbath_12+1)/2 * (2*s12*rho*s21 - s22*rho - rho*s22)
    L21 = gamma12*nbath_12/2 * (2*s21*rho*s12 - s11*rho - rho*s11)
    L13 = gamma13/2 * (2*s13*rho*s31 - s33*rho - rho*s33)
    L14 = gamma14/2 * (2*s14*rho*s41 - s44*rho - rho*s44)
    L23 = gamma23/2 * (2*s23*rho*s32 - s33*rho - rho*s33)
    L24 = gamma24/2 * (2*s24*rho*s42 - s44*rho - rho*s44)
    L34 = gamma34*(nbath_34+1)/2 * (2*s34*rho*s43 - s44*rho - rho*s44)
    L43 = gamma34*nbath_34/2 * (2*s43*rho*s34 - s33*rho - rho*s33)
    L2d = gamma2d/2 * (2*s22*rho*s22 - s22*rho - rho*s22)
    L3d = gamma3d/2 * (2*s33*rho*s33 - s33*rho - rho*s33)
    L4d = gamma4d/2 * (2*s44*rho*s44 - s44*rho - rho*s44)
    return L12 + L21 + L13 + L14 + L23 + L24 + L34 + L43 + L2d + L3d + L4d

def liouvillan_superoperator(rho):
    return -sympy.I*(H*rho-rho*H) + loss_superoperator(rho)

# construct Liouvillan matrix
L = sympy.zeros(16)
for column, (icol, jcol) in enumerate(zip(*flatten)):
    # basis matrix
    rho = sympy.zeros(4)
    rho[icol,jcol] = 1

    # get column of supermatrix
    L_column = liouvillan_superoperator(rho)
    for row, (irow, jrow) in enumerate(zip(*flatten)):
        L[row,column] = L_column[irow,jrow]

# matrix to convert complex Hermitian to real nonsymmetric
C = sympy.zeros(16)
for k in range(4):
    ik = unflatten[k,k]
    C[ik,ik] = 1
for j in range(3):
    for k in range(j+1, 4):
        ij = unflatten[j,k]
        ik = unflatten[k,j]
        C[ij,ij] = sympy.Rational(1, 2)
        C[ij,ik] = sympy.Rational(1, 2)
        C[ik,ij] = -sympy.I/2
        C[ik,ik] = sympy.I/2

# Liouvillan matrix converted to real
C_inv = C.inv()
Lreal = sympy.re(C*L*C.inv())

# lambdified functions
def short_lambdify(params, expr):
    return sympy.lambdify(params, expr, 'numpy', cse=True, docstring_limit=0)

Hsymbols = (omega12, delta_B, delta_mu, Omega_mu,
            Omega_A, Omega_B, Omega_D, Omega_E)
Lsymbols = Hsymbols + (gamma12, gamma13, gamma14, gamma23, gamma24, gamma34,
                        gamma2d, gamma3d, gamma4d, nbath_12, nbath_34)
Hfunc = short_lambdify(Hsymbols, H)
Lrealfunc = short_lambdify(Lsymbols, Lreal)

# gross hack to get numerical matrices from sympy
x = sympy.symbols('x')
complex_to_real = short_lambdify(x, C)(None)
real_to_complex = short_lambdify(x, C_inv)(None)

# discriminant symbolic and lambdified functions for numerical methods

def expr_to_poly_coeffs(expr, x):
    poly = sympy.poly(expr, x)
    n = poly.degree(x)
    coeffs = sympy.zeros(n+1, 1)
    for (k,), coeff in zip(poly.monoms(), poly.coeffs()):
        coeffs[k] = coeff
    return coeffs

def diff_poly_coeffs(coeffs):
    n = coeffs.shape[0]-1
    diff_coeffs = sympy.zeros(n, 1)
    for k in range(1, n+1):
        diff_coeffs[k-1,0] = sympy.Integer(k)*coeffs[k,0]
    return diff_coeffs

# restore the symbolic expressions from a cache, if it exists
fname = 'result-cache/Delta_expr.pkl'
if os.path.isfile(fname):
    with open(fname, 'rb') as f:
        Delta_expr = pickle.load(f)
        Delta = Delta_expr['Delta']
        Delta_coeffs_delta_B = Delta_expr['Delta_coeffs_delta_B']
        Delta_coeffs_delta_mu = Delta_expr['Delta_coeffs_delta_mu']
        dmu_Delta = Delta_expr['dmu_Delta']
        dmu_Delta_coeffs_delta_B = Delta_expr['dmu_Delta_coeffs_delta_B']
        dmu_Delta_coeffs_delta_mu = Delta_expr['dmu_Delta_coeffs_delta_mu']
        dB_Delta = Delta_expr['dB_Delta']
        dB_Delta_coeffs_delta_B = Delta_expr['dB_Delta_coeffs_delta_B']
        dB_Delta_coeffs_delta_mu = Delta_expr['dB_Delta_coeffs_delta_mu']
        d2mu_Delta = Delta_expr['d2mu_Delta']
        d2B_Delta = Delta_expr['d2B_Delta']
else:
    # compute the discriminant Delta of the characteristic polynomial
    # of the Hamiltonian; this slightly convoluted method of using
    # a generic quartic is faster than the obvious way
    print('Computing the symbolic polynomial coefficients of the discriminant')
    print('This takes about 20 minutes on my machine, so get ready to wait')
    print(f'The result will be saved as {fname} to be re-used in future builds')
    a = [a0, a1, a2, a3, a4] = sympy.symbols('a_(0:5)', real=True)
    x = sympy.symbols('x', real=True)
    poly = sympy.poly(a4*x**4 + a3*x**3 + a2*x**2 + a1*x + a0, x)
    Delta = poly.discriminant()
    charpoly = H.charpoly()
    for (k,), coeff in zip(charpoly.monoms(), charpoly.coeffs()):
        Delta = Delta.subs(a[k], coeff)

    # compute various polynomial coefficients and derivatives of Delta
    Delta_coeffs_delta_B = expr_to_poly_coeffs(Delta, delta_B)
    Delta_coeffs_delta_mu = expr_to_poly_coeffs(Delta, delta_mu)
    dmu_Delta = sympy.diff(Delta, delta_mu)
    dmu_Delta_coeffs_delta_B = sympy.diff(Delta_coeffs_delta_B, delta_mu)
    dmu_Delta_coeffs_delta_mu = diff_poly_coeffs(Delta_coeffs_delta_mu)
    dB_Delta = sympy.diff(Delta, delta_B)
    dB_Delta_coeffs_delta_B = diff_poly_coeffs(Delta_coeffs_delta_B)
    dB_Delta_coeffs_delta_mu = sympy.diff(Delta_coeffs_delta_mu, delta_B)
    d2mu_Delta = sympy.diff(Delta, delta_mu, 2)
    d2B_Delta = sympy.diff(Delta, delta_B, 2)
    Delta_expr = {
        'Delta': Delta,
        'Delta_coeffs_delta_B': Delta_coeffs_delta_B,
        'Delta_coeffs_delta_mu': Delta_coeffs_delta_mu,
        'dmu_Delta': dmu_Delta,
        'dmu_Delta_coeffs_delta_B': dmu_Delta_coeffs_delta_B,
        'dmu_Delta_coeffs_delta_mu': dmu_Delta_coeffs_delta_mu,
        'dB_Delta': dB_Delta,
        'dB_Delta_coeffs_delta_B': dB_Delta_coeffs_delta_B,
        'dB_Delta_coeffs_delta_mu': dB_Delta_coeffs_delta_mu,
        'd2mu_Delta': d2mu_Delta,
        'd2B_Delta': d2B_Delta
    }
    with open(fname, 'wb') as f:
        pickle.dump(Delta_expr, f)

H_symbols_common = (omega12, Omega_A, Omega_B, Omega_D, Omega_E, Omega_mu)
H_symbols_no_dB = H_symbols_common + (delta_mu,)
H_symbols_no_dmu = H_symbols_common + (delta_B,)
H_symbols_all = Hsymbols

# lambdify these expressions; this can't be pickled
Delta_func = short_lambdify(H_symbols_all, Delta)
Delta_coeffs_delta_B_func = short_lambdify(H_symbols_no_dB,
                                           Delta_coeffs_delta_B)
Delta_coeffs_delta_mu_func = sympy.lambdify(H_symbols_no_dmu,
                                            Delta_coeffs_delta_mu)
dmu_Delta_func = sympy.lambdify(H_symbols_all, dmu_Delta)
dmu_Delta_coeffs_delta_B_func = sympy.lambdify(H_symbols_no_dB,
                                               dmu_Delta_coeffs_delta_B)
dmu_Delta_coeffs_delta_mu_func = sympy.lambdify(H_symbols_no_dmu,
                                                dmu_Delta_coeffs_delta_mu)
dB_Delta_func = sympy.lambdify(H_symbols_all, dB_Delta)
dB_Delta_coeffs_delta_B_func = sympy.lambdify(H_symbols_no_dB,
                                              dB_Delta_coeffs_delta_B)
dB_Delta_coeffs_delta_mu_func = sympy.lambdify(H_symbols_no_dmu,
                                               dB_Delta_coeffs_delta_mu)
d2mu_Delta_func = sympy.lambdify(H_symbols_all, d2mu_Delta)
d2B_Delta_func = sympy.lambdify(H_symbols_all, d2B_Delta)

# helper function for multivariate normal distribution; scipy's has a bad API

def multivariate_normal_pdf(Sigma, *args):
    d = len(args)
    x = np.array(args)
    coef = 1 / np.sqrt((2*np.pi)**d * np.linalg.det(Sigma))

    # permutation sending the first dimension to the second-last
    axes = list(range(x.ndim))
    axes[0] = -2
    axes[-2] = 0
    
    xDivSigma = np.linalg.solve(Sigma, np.transpose(x, axes=axes))
    xDivSigma = np.transpose(xDivSigma, axes=axes)
    
    return coef * np.exp(-np.einsum('i...,i...', x, xDivSigma)/2)

# computes a grid of atom output signal, without multisampling

def rho_steady_state(omega_12, omega_34, delta_B, delta_mu, Omega_A,
                     Omega_B, Omega_D, Omega_E, Omega_mu, tau_12,
                     tau_34, gamma_13, gamma_14, gamma_23, gamma_24,
                     gamma_2d, gamma_3d, gamma_4d, T):
    n_12 = 1 / np.expm1(hbar*omega_12/(kB*T))
    n_34 = 1 / np.expm1(hbar*omega_34/(kB*T))
    L = Lrealfunc(
        omega_12=omega_12,
        delta_B=delta_B,
        delta_mu=delta_mu,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu,
        gamma_12=1/(tau_12*(n_12+1)),
        gamma_13=gamma_13,
        gamma_14=gamma_14,
        gamma_23=gamma_23,
        gamma_24=gamma_24,
        gamma_34=1/(tau_34*(n_34+1)),
        gamma_2d=gamma_2d,
        gamma_3d=gamma_3d,
        gamma_4d=gamma_4d,
        n_12=n_12,
        n_34=n_34
    )
    L[0,:] = np.identity(4)[flatten]

    b = np.zeros(16)
    b[0] = 1

    rho_real = np.linalg.solve(L, b)
    rho = real_to_complex @ rho_real
    return rho[unflatten]

def atom_signal(omega_12, omega_34, delta_B, delta_mu, Omega_A,
                Omega_B, Omega_D, Omega_E, Omega_mu, tau_12, tau_34,
                gamma_13, gamma_14, gamma_23, gamma_24, C_14,
                C_24, gamma_2d, gamma_3d, gamma_4d, T):
    rho = rho_steady_state(
        omega_12=omega_12,
        omega_34=omega_34,
        delta_B=delta_B,
        delta_mu=delta_mu,
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
        T=T
    )
    
    signal_D = C_24 * rho[3,1]
    signal_E = C_14 * rho[3,0]
    photon_rate_out = np.abs(signal_D + signal_E)**2
    return photon_rate_out

def atom_scan(delta_mu_min, delta_mu_max, delta_mu_points,
              delta_B_min, delta_B_max, delta_B_points,
              omega_12, omega_34, Omega_A, Omega_B, Omega_D, Omega_E,
              Omega_mu, tau_12, tau_34, gamma_13, gamma_14, C_14, C_24,
              gamma_23, gamma_24, gamma_2d, gamma_3d, gamma_4d, T):
    delta_mu = np.linspace(delta_mu_min, delta_mu_max, delta_mu_points)
    delta_B = np.linspace(delta_B_min, delta_B_max, delta_B_points)
    delta_mu, delta_B = np.meshgrid(delta_mu, delta_B, indexing='ij')

    scan = np.zeros_like(delta_mu)
    for i in range(scan.shape[0]):
        for j in range(scan.shape[1]):
            scan[i,j] = atom_signal(
                omega_12=omega_12,
                omega_34=omega_34,
                delta_B=delta_B[i,j],
                delta_mu=delta_mu[i,j],
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
                T=T
            )

    return delta_mu, delta_B, scan

# multisampling functions

def poly_real_roots(poly_coeffs):
    poly = np.polynomial.Polynomial(poly_coeffs)
    roots = poly.roots()
    real_roots = np.array([np.real(x) for x in roots if np.imag(x)==0])
    return real_roots

def poly_positive_minima(poly_func, diff_coeffs):
    critical_x = poly_real_roots(diff_coeffs)
    critical_y = poly_func(critical_x)

    # base cases
    if len(critical_x) == 0:
        return []
    if len(critical_x) <= 1:
        return critical_x

    minima = []
    
    # check if leftmost critical point is minimum
    if critical_y[0] < critical_y[1]:
        minima.append(critical_x[0])

    # check for mimima between critical points
    for i in range(len(critical_x)-2):
        y1 = critical_y[i]
        x2 = critical_x[i+1]
        y2 = critical_y[i+1]
        y3 = critical_y[i+2]
        if y1 > y2 and y3 > y2:
            minima.append(x2)

    # check if rightmost critical point is mimimum
    if critical_y[-1] < critical_y[-2]:
        minima.append(critical_x[-1])

    return minima

def H_get_minimal_delta_B(omega_12, delta_mu, Omega_A,
                          Omega_B, Omega_D, Omega_E, Omega_mu):
    # minima along delta_B
    dB_Delta_coeffs = dB_Delta_coeffs_delta_B_func(
        omega_12=omega_12,
        delta_mu=delta_mu,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu
    )[:,0]
    Delta_short_func = lambda delta_B: Delta_func(
        omega_12=omega_12,
        delta_mu=delta_mu,
        delta_B=delta_B,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu
    )
    minima = list(poly_positive_minima(Delta_short_func, dB_Delta_coeffs))
    is_parallel = [True]*len(minima)

    # minima along delta_mu
    dmu_Delta_coeffs = dmu_Delta_coeffs_delta_B_func(
        omega_12=omega_12,
        delta_mu=delta_mu,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu
    )[:,0]
    roots = poly_real_roots(dmu_Delta_coeffs)
    for root in roots:
        curvature = d2mu_Delta_func(
            omega_12=omega_12,
            delta_mu=delta_mu,
            delta_B=root,
            Omega_A=Omega_A,
            Omega_B=Omega_B,
            Omega_D=Omega_D,
            Omega_E=Omega_E,
            Omega_mu=Omega_mu
        )
        if curvature >= 0:
            minima.append(root)
            is_parallel.append(False)

    return minima, is_parallel

def H_get_minimal_delta_mu(omega_12, delta_B, Omega_A,
                           Omega_B, Omega_D, Omega_E, Omega_mu):
    # minima along delta_mu
    dmu_Delta_coeffs = dmu_Delta_coeffs_delta_mu_func(
        omega_12=omega_12,
        delta_B=delta_B,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu
    )[:,0]
    Delta_short_func = lambda delta_mu: Delta_func(
        omega_12=omega_12,
        delta_mu=delta_mu,
        delta_B=delta_B,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu
    )
    minima = list(poly_positive_minima(Delta_short_func, dmu_Delta_coeffs))
    is_parallel = [True]*len(minima)

    # minima along delta_mu
    dB_Delta_coeffs = dB_Delta_coeffs_delta_mu_func(
        omega_12=omega_12,
        delta_B=delta_B,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu
    )[:,0]
    roots = poly_real_roots(dB_Delta_coeffs)
    for root in roots:
        curvature = d2B_Delta_func(
            omega_12=omega_12,
            delta_mu=root,
            delta_B=delta_B,
            Omega_A=Omega_A,
            Omega_B=Omega_B,
            Omega_D=Omega_D,
            Omega_E=Omega_E,
            Omega_mu=Omega_mu
        )
        if curvature >= 0:
            minima.append(root)
            is_parallel.append(False)

    return minima, is_parallel

def find_curve_pixel_intersections(delta_mu_min, delta_mu_max, delta_mu_points,
                                   delta_B_min, delta_B_max, delta_B_points,
                                   omega_12, Omega_A, Omega_B,
                                   Omega_D, Omega_E, Omega_mu):
    def linspace_edges(start, stop, count):
        dx = (stop-start) / (count-1)
        edges = np.linspace(start-dx/2, stop+dx/2, count+1)
        return edges

    delta_mu_edges = linspace_edges(delta_mu_min, delta_mu_max, delta_mu_points)
    delta_B_edges = linspace_edges(delta_B_min, delta_B_max, delta_B_points)
    delta_mu_edges_min, delta_mu_edges_max = delta_mu_edges[0], delta_mu_edges[-1]
    delta_B_edges_min, delta_B_edges_max = delta_B_edges[0], delta_B_edges[-1]
    
    intersection_points = []
    intersecting_pixels = dict()
    is_parallel = []
    
    for i, dmu in enumerate(delta_mu_edges):
        dBs, paras = H_get_minimal_delta_B(
            omega_12=omega_12,
            delta_mu=dmu,
            Omega_A=Omega_A,
            Omega_B=Omega_B,
            Omega_D=Omega_D,
            Omega_E=Omega_E,
            Omega_mu=Omega_mu
        )
        for dB, para in zip(dBs, paras):
            if not delta_B_edges_min <= dB <= delta_B_edges_max:
                continue
            point = (dmu, dB)
            intersection_points.append((dmu, dB))
            is_parallel.append(para)
            
            j = int(delta_B_points * (dB-delta_B_edges_min) / (delta_B_edges_max-delta_B_edges_min))
            if i > 0:
                if (i-1,j) not in intersecting_pixels:
                    intersecting_pixels[i-1,j] = []
                intersecting_pixels[i-1,j].append(point)
            if i < delta_mu_points:
                if (i,j) not in intersecting_pixels:
                    intersecting_pixels[i,j] = []
                intersecting_pixels[i,j].append(point)
    
    for j, dB in enumerate(delta_B_edges):
        dmus, paras = H_get_minimal_delta_mu(
            omega_12=omega_12,
            delta_B=dB,
            Omega_A=Omega_A,
            Omega_B=Omega_B,
            Omega_D=Omega_D,
            Omega_E=Omega_E,
            Omega_mu=Omega_mu
        )
        for dmu, para in zip(dmus, paras):
            if not delta_mu_edges_min <= dmu <= delta_mu_edges_max:
                continue
            point = (dmu, dB)
            intersection_points.append((dmu, dB))
            is_parallel.append(para)
            
            i = int(delta_mu_points * (dmu-delta_mu_edges_min) / (delta_mu_edges_max-delta_mu_edges_min))
            if j > 0:
                if (i,j-1) not in intersecting_pixels:
                    intersecting_pixels[i,j-1] = []
                intersecting_pixels[i,j-1].append(point)
            if j < delta_B_points:
                if (i,j) not in intersecting_pixels:
                    intersecting_pixels[i,j] = []
                intersecting_pixels[i,j].append(point)

    return delta_mu_edges, delta_B_edges, intersection_points, intersecting_pixels, is_parallel

# precompute Gauss-Lobatto quadrature nodes and weights
gauss_lobatto_nodes_weights = dict()
for n in range(2, 100):
    legendre_coeffs = (0,)*(n-1) + (1,)
    legendre_poly = np.polynomial.Legendre(legendre_coeffs)
    nodes = legendre_poly.deriv().roots()
    nodes = np.concatenate([[-1.0], nodes, [1.0]])
    weights = 2 / (n*(n-1)*legendre_poly(nodes)**2)

    # convert from [-1, 1] to [0, 1]
    nodes = (nodes+1)/2
    weights = weights/2
    
    gauss_lobatto_nodes_weights[n] = (nodes, weights)

def composite_gauss_lobatto_nodes_weights(n, a, b, intermediates=[]):
    if n not in gauss_lobatto_nodes_weights:
        raise ValueError
    if a >= b:
        raise ValueError
    
    intermediates = [x for x in intermediates if a < x < b]
    points = sorted([a, b] + intermediates)
    num_points = len(points)
    base_nodes, base_weights = gauss_lobatto_nodes_weights[n]
    if num_points == 2:
        L = points[1] - points[0]
        nodes = points[0] + base_nodes*L
        weights = base_weights*L
        return nodes, weights

    count = (num_points-1)*(n-1) + 1
    nodes = np.zeros(count)
    weights = np.zeros(count)

    # first and last node
    nodes[0] = points[0]
    weights[0] = base_weights[0] * (points[1]-points[0])
    nodes[1] = points[-1]
    weights[1] = base_weights[-1] * (points[-1]-points[-2])
    filled = 2

    # nodes at intermediate points
    for i in range(1,num_points-1):
        nodes[filled] = points[i]
        weights[filled] = base_weights[-1]*(points[i]-points[i-1])
        weights[filled] += base_weights[0]*(points[i+1]-points[i])
        filled += 1

    # nodes between points
    if n == 2:
        return nodes, weights
    for i in range(num_points-1):
        L = points[i+1]-points[i]
        nodes[filled:filled+n-2] = points[i] + base_nodes[1:-1]*L
        weights[filled:filled+n-2] = base_weights[1:-1]*L
        filled += n-2

    return nodes, weights

def atom_signal_integrated(delta_mu_min, delta_mu_max, delta_B_min, delta_B_max,
                           edge_points, omega_12, omega_34, Omega_A, Omega_B, Omega_D,
                           Omega_E, Omega_mu, tau_12, tau_34, gamma_13, gamma_14,
                           gamma_23, gamma_24, gamma_2d, gamma_3d, gamma_4d, C_14, C_24, T):
    outer_order = 5
    inner_order = 10

    delta_B_linewidth = gamma_3d
    delta_mu_linewidth = gamma_3d + gamma_4d

    area = (delta_mu_max-delta_mu_min) * (delta_B_max-delta_B_min)
    
    def lerp(a, b, t):
        return a + t*(b-a)
    def ilerp(a, b, x):
        return (x-a)/(b-a)

    def normalise_delta_mu(delta_mu):
        return ilerp(delta_mu_min, delta_mu_max, delta_mu)
    def unnormalise_delta_mu(t):
        return lerp(delta_mu_min, delta_mu_max, t)
    def normalise_delta_B(delta_B):
        return ilerp(delta_B_min, delta_B_max, delta_B)
    def unnormalise_delta_B(t):
        return lerp(delta_B_min, delta_B_max, t)

    def bounds_range(array):
        if len(array) == 0:
            return [], 0
        if len(array) == 1:
            return list(array), 0

        a = np.min(array)
        b = np.max(array)
        return [a, b], b-a

    def atom_signal_short(delta_mu, delta_B):
        return atom_signal(
            omega_12=omega_12,
            omega_34=omega_34,
            delta_mu=delta_mu,
            delta_B=delta_B,
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

    edge_mu_points = [x[0] for x in edge_points]
    edge_B_points = [x[1] for x in edge_points]
    
    # get direction for outer integral
    if len(edge_points) == 0:
        outer = 'delta_mu' # arbitrary
    elif len(edge_points) == 1:
        edge_mu = edge_mu_points[0]
        edge_B = edge_B_points[0]
        if normalise_delta_mu(edge_mu) > normalise_delta_B(edge_B):
            outer = 'delta_mu'
        else:
            outer = 'delta_B'
    else:
        edge_mu_min = np.min(edge_mu_points)
        edge_mu_max = np.max(edge_mu_points)
        edge_B_min = np.min(edge_B_points)
        edge_B_max = np.max(edge_B_points)
        width_mu = (edge_mu_max-edge_mu_min) / (delta_mu_max-delta_mu_min)
        width_B = (edge_B_max-edge_B_min) / (delta_B_max-delta_B_min)
        if width_mu > width_B:
            outer = 'delta_mu'
        else:
            outer = 'delta_B'

    # orientation-neutral definitions
    def atom_signal_short_neutral(delta_outer, delta_inner):
        if outer == 'delta_mu':
            delta_mu = delta_outer
            delta_B = delta_inner
        else:
            delta_mu = delta_inner
            delta_B = delta_outer
        return atom_signal_short(delta_mu, delta_B)
        
    def H_get_minimal_delta_inner(delta_outer):
        if outer == 'delta_mu':
            return H_get_minimal_delta_B(
                omega_12=omega_12,
                delta_mu=delta_outer,
                Omega_A=Omega_A,
                Omega_B=Omega_B,
                Omega_D=Omega_D,
                Omega_E=Omega_E,
                Omega_mu=Omega_mu
            )
        else:
            return H_get_minimal_delta_mu(
                omega_12=omega_12,
                delta_B=delta_outer,
                Omega_A=Omega_A,
                Omega_B=Omega_B,
                Omega_D=Omega_D,
                Omega_E=Omega_E,
                Omega_mu=Omega_mu
            )
    
    if outer == 'delta_mu':
        delta_outer_min = delta_mu_min
        delta_outer_max = delta_mu_max
        delta_inner_min = delta_B_min
        delta_inner_max = delta_B_max
        delta_inner_linewidth = delta_B_linewidth
        edge_outer_points = edge_mu_points
        edge_inner_points = edge_B_points
    else:
        delta_outer_min = delta_B_min
        delta_outer_max = delta_B_max
        delta_inner_min = delta_mu_min
        delta_inner_max = delta_mu_max
        delta_inner_linewidth = delta_mu_linewidth
        edge_outer_points = edge_B_points
        edge_inner_points = edge_mu_points

    edge_outer_bounds, _ = bounds_range(edge_outer_points)

    # set up diagnostic logging
    log = {
        'delta_mu_min': delta_mu_min,
        'delta_mu_max': delta_mu_max,
        'delta_B_min': delta_B_min,
        'delta_B_max': delta_B_max,
        'edge_points': edge_points,
        'omega_12': omega_12,
        'omega_34': omega_34,
        'Omega_A': Omega_A,
        'Omega_B': Omega_B,
        'Omega_D': Omega_D,
        'Omega_E': Omega_E,
        'Omega_mu': Omega_mu,
        'tau_12': tau_12,
        'tau_34': tau_34,
        'gamma_13': gamma_13,
        'gamma_14': gamma_14,
        'gamma_23': gamma_23,
        'gamma_24': gamma_24,
        'gamma_2d': gamma_2d,
        'gamma_3d': gamma_3d,
        'gamma_4d': gamma_4d,
        'C_14': C_14,
        'C_24': C_24,
        'T': T,
        'outer': outer
    }
    
    # perform integral
    integral = 0
    outer_nodes, outer_weights = composite_gauss_lobatto_nodes_weights(
        outer_order, delta_outer_min, delta_outer_max, intermediates=edge_outer_bounds)
    log['outer_integral'] = {
        'intermediates': edge_outer_bounds,
        'nodes': outer_nodes,
        'weights': outer_weights,
        'inner_integrals': []
    }
    for outer_node, outer_weight in zip(outer_nodes, outer_weights):
        inner_intersections, _ = H_get_minimal_delta_inner(outer_node)
        inner_intermediates = [z*delta_inner_linewidth+x
                               for x in inner_intersections
                               for z in (-10, -3, -1, 0, 1, 3, 10)]
        log_inner_integral = {
            'intersections': inner_intersections,
            'intermediates': inner_intermediates
        }
        inner_nodes, inner_weights = composite_gauss_lobatto_nodes_weights(
            inner_order, delta_inner_min, delta_inner_max, intermediates=inner_intermediates)
        log_inner_integral['nodes'] = inner_nodes
        log_inner_integral['weights'] = inner_weights
        log_inner_integral['values'] = []
        for inner_node, inner_weight in zip(inner_nodes, inner_weights):
            weight = outer_weight * inner_weight
            sample = atom_signal_short_neutral(outer_node, inner_node) / area
            log_inner_integral['values'].append(sample)
            integral += weight * sample
        log['outer_integral']['inner_integrals'].append(log_inner_integral)

    log['integral'] = integral
    return integral, log

def atom_scan_integrated(delta_mu_min, delta_mu_max, delta_mu_points,
                         delta_B_min, delta_B_max, delta_B_points,
                         omega_12, omega_34, Omega_A, Omega_B, Omega_D, Omega_E,
                         Omega_mu, tau_12, tau_34, gamma_13, gamma_14, gamma_23,
                         gamma_24, gamma_2d, gamma_3d, gamma_4d, C_14, C_24, T):
    
    (delta_mu_edges,
     delta_B_edges,
     intersection_points,
     intersecting_pixels, _) = find_curve_pixel_intersections(
        delta_mu_min=delta_mu_min,
        delta_mu_max=delta_mu_max,
        delta_mu_points=delta_mu_points,
        delta_B_min=delta_B_min,
        delta_B_max=delta_B_max,
        delta_B_points=delta_B_points,
        omega_12=omega_12,
        Omega_A=Omega_A,
        Omega_B=Omega_B,
        Omega_D=Omega_D,
        Omega_E=Omega_E,
        Omega_mu=Omega_mu
    )
    
    scan = np.full((delta_mu_points, delta_B_points), np.nan, dtype=float)
    logs = dict()
    for (i,j), edge_points in intersecting_pixels.items():
        scan[i,j], logs[i,j] = atom_signal_integrated(
            delta_mu_min=delta_mu_edges[i],
            delta_mu_max=delta_mu_edges[i+1],
            delta_B_min=delta_B_edges[j],
            delta_B_max=delta_B_edges[j+1],
            edge_points=edge_points,
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
            T=T
        )

    delta_mu = np.linspace(delta_mu_min, delta_mu_max, delta_mu_points)
    delta_B = np.linspace(delta_B_min, delta_B_max, delta_B_points)
    delta_mu, delta_B = np.meshgrid(delta_mu, delta_B, indexing='ij')
    for i in range(scan.shape[0]):
        for j in range(scan.shape[1]):
            if not np.isnan(scan[i,j]):
                continue
            scan[i,j] = atom_signal(
                omega_12=omega_12,
                omega_34=omega_34,
                delta_B=delta_B[i,j],
                delta_mu=delta_mu[i,j],
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
                T=T
            )

    return (delta_mu, delta_B, scan), logs

def signal_scan(delta_mu_min, delta_mu_max, delta_mu_points,
                delta_B_min, delta_B_max, delta_B_points,
                omega_12, omega_34, Omega_A, Omega_B, Omega_D, Omega_E,
                Omega_mu, tau_12, tau_34, gamma_13, gamma_14,
                gamma_23, gamma_24, gamma_2d, gamma_3d, gamma_4d, T,
                C_14, C_24, Sigma, z_max=5, integrated=True):
    delta_mu_res = (delta_mu_max-delta_mu_min) / (delta_mu_points-1)
    delta_B_res = (delta_B_max-delta_B_min) / (delta_B_points-1)

    # produce Gaussian filter kernel
    sigma_13 = np.sqrt(Sigma[0,0])
    sigma_34 = np.sqrt(Sigma[1,1])
    sigma_13_px = sigma_13 / delta_B_res
    sigma_34_px = sigma_34 / delta_mu_res
    radius_13_px = int(sigma_13_px*z_max)
    radius_34_px = int(sigma_34_px*z_max)
    delta_13_points = 2*radius_13_px + 1
    delta_34_points = 2*radius_34_px + 1
    delta_13 = np.linspace(-sigma_13*z_max, sigma_13*z_max, delta_13_points)
    delta_34 = np.linspace(-sigma_34*z_max, sigma_34*z_max, delta_34_points)
    delta_34, delta_13 = np.meshgrid(delta_34, delta_13, indexing='ij')
    kernel = multivariate_normal_pdf(Sigma, delta_13, delta_34)
    kernel /= np.sum(kernel)

    # collect atomic emission data
    deltap_B_min = delta_B_min - delta_B_res*radius_13_px
    deltap_B_max = delta_B_max + delta_B_res*radius_13_px
    deltap_B_points = delta_B_points + delta_13_points - 1
    deltap_mu_min = delta_mu_min - delta_mu_res*radius_34_px
    deltap_mu_max = delta_mu_max + delta_mu_res*radius_34_px
    deltap_mu_points = delta_mu_points + delta_34_points - 1
    
    if integrated:
        (_, _, atom_scan), _ = atom_scan_integrated(
            delta_mu_min=deltap_mu_min,
            delta_mu_max=deltap_mu_max,
            delta_mu_points=deltap_mu_points,
            delta_B_min=deltap_B_min,
            delta_B_max=deltap_B_max,
            delta_B_points=deltap_B_points,
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
            T=T
        )
    else:
        _, _, atom_scan = atom_scan(
            delta_mu_min=deltap_mu_min,
            delta_mu_max=deltap_mu_max,
            delta_mu_points=deltap_mu_points,
            delta_B_min=deltap_B_min,
            delta_B_max=deltap_B_max,
            delta_B_points=deltap_B_points,
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
            T=T
        )

    # convolve
    ensemble_signal = signal.fftconvolve(atom_scan, kernel, mode='valid')

    # return results
    delta_mu = np.linspace(delta_mu_min, delta_mu_max, delta_mu_points)
    delta_B = np.linspace(delta_B_min, delta_B_max, delta_B_points)
    delta_mu, delta_B = np.meshgrid(delta_mu, delta_B, indexing='ij')
    return delta_mu, delta_B, ensemble_signal
