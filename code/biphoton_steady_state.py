import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
import sympy

hbar = 1.054571817e-34
kB = 1.380649e-23

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

def composite_gauss_lobatto_nodes_weights(n, points):
    if n not in gauss_lobatto_nodes_weights:
        raise ValueError

    num_points = len(points)
    if num_points < 2:
        raise ValueError
    
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

# unit matrices
s12 = sympy.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
s13 = sympy.Matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
s21 = s12.H
s31 = s13.H
s11 = s12*s21
s22 = s21*s12
s23 = s21*s13
s32 = s23.H
s33 = s31*s13

# Hamiltonians for Lambda-system and V-system

Omega_p = sympy.symbols('Omega_p')
Omega_o = sympy.symbols('Omega_o')
Omega_mu = sympy.symbols('Omega_mu')
delta_o = sympy.symbols('delta_o', real=True)
delta_mu = sympy.symbols('delta_mu', real=True)

HLambda = s21*Omega_mu + s32*Omega_o + s31*Omega_p
HLambda += HLambda.H
HLambda += s22*delta_mu + s33*(delta_mu+delta_o)

HV = s21*Omega_o + s32*Omega_mu + s31*Omega_p
HV += HV.H
HV += s22*delta_o + s33*(delta_mu+delta_o)

systems = ('Lambda', 'V')
H = {'Lambda': HLambda, 'V': HV}

# decomposition of Hamiltonians into linear components
Omega_or, Omega_oi = sympy.symbols('Omega_or Omega_oi', real=True)
Omega_mur, Omega_mui = sympy.symbols('Omega_\\mu\\ r Omega_\\mu\\ i', real=True)
subs = [
    (Omega_o, Omega_or+sympy.I*Omega_oi),
    (Omega_mu, Omega_mur+sympy.I*Omega_mui)
]

H_subs = dict()
for sys in systems:
    H_subs[sys] = H[sys].subs(subs)

H0 = dict()
Hor = dict()
Hoi = dict()
Hmur = dict()
Hmui = dict()
for sys in systems:
    H0[sys] = H[sys].subs([(Omega_o, 0), (Omega_mu, 0)])
    Hor[sys] = sympy.diff(H_subs[sys], Omega_or)
    Hoi[sys] = sympy.diff(H_subs[sys], Omega_oi)
    Hmur[sys] = sympy.diff(H_subs[sys], Omega_mur)
    Hmui[sys] = sympy.diff(H_subs[sys], Omega_mui)

# discriminant of characteristic polynomial of Hamiltonian

Delta_poly_coeffs = dict()
for sys in systems:
    Delta = H[sys].charpoly().discriminant()
    Delta_poly = sympy.poly(Delta, delta_mu, delta_o)
    n_delta_mu = Delta_poly.degree(delta_mu)
    n_delta_o = Delta_poly.degree(delta_o)
    Delta_poly_coeffs[sys] = sympy.zeros(n_delta_mu+1, n_delta_o+1)
    for (i, j), coeff in zip(Delta_poly.monoms(), Delta_poly.coeffs()):
        Delta_poly_coeffs[sys][i,j] = coeff

# liouvillan superoperators with loss

gamma_12, gamma_13 = sympy.symbols('gamma_1(2:4)', real=True, negative=False)
gamma_23 = sympy.symbols('gamma_23', real=True, negative=False)
gamma_2d, gamma_3d = sympy.symbols('gamma_(2:4)d', real=True, negative=False)
n_b = sympy.symbols('n_b', real=True, negative=False)

def loss_operator_common(rho):
    L13 = gamma_13/2 * (2*s13*rho*s31 - rho*s33 - s33*rho)
    L2d = gamma_2d/2 * (2*s22*rho*s22 - rho*s22 - s22*rho)
    L3d = gamma_3d/2 * (2*s33*rho*s33 - rho*s33 - s33*rho)
    return L13 + L2d + L3d

def loss_operator_Lambda(rho):
    L12 = gamma_12*(n_b+1)/2 * (2*s12*rho*s21 - rho*s22 - s22*rho)
    L21 = gamma_12*n_b/2 * (2*s21*rho*s12 - rho*s11 - s11*rho)
    L23 = gamma_23/2 * (2*s23*rho*s32 - rho*s33 - s33*rho)
    return L12 + L21 + L23 + loss_operator_common(rho)

def loss_operator_V(rho):
    L12 = gamma_12/2 * (2*s12*rho*s21 - rho*s22 - s22*rho)
    L23 = gamma_23*(n_b+1)/2 * (2*s23*rho*s32 - rho*s33 - s33*rho)
    L32 = gamma_23*n_b/2 * (2*s32*rho*s23 - rho*s22 - s22*rho)
    return L12 + L23 + L32 + loss_operator_common(rho)

loss_operator = {'Lambda': loss_operator_Lambda, 'V': loss_operator_V}

def liouvillan_superoperator(H, rho, loss=None):
    Lrho = -sympy.I*(H*rho - rho*H)
    if loss is not None:
        Lrho += loss(rho)
    return Lrho

# liouvillan matrices
def flattening_indices(order):
    n = order.shape[0]
    unflatten = order
    
    flatten_i = np.zeros(n**2, dtype=int)
    flatten_j = np.zeros(n**2, dtype=int)
    for i in range(n):
        for j in range(n):
            k = order[i,j]
            flatten_i[k] = i
            flatten_j[k] = j
    flatten = (flatten_i, flatten_j)

    return unflatten, flatten

def liouvillan_matrix(H, order, loss=None):
    n = order.shape[0]
    mtx = sympy.zeros(n**2)
    for icol in range(n):
        for jcol in range(n):
            col = order[icol,jcol]
            rho = sympy.zeros(n)
            rho[icol,jcol] = 1
            Lrho = liouvillan_superoperator(H, rho, loss=loss)
            
            for irow in range(n):
                for jrow in range(n):
                    row = order[irow,jrow]
                    mtx[row,col] = Lrho[irow,jrow]
    return mtx

order = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])
unflatten, flatten = flattening_indices(order)

L0 = dict()
Lor = dict()
Loi = dict()
Lmur = dict()
Lmui = dict()
L = dict()
for sys in systems:
    L0[sys] = liouvillan_matrix(H0[sys], order, loss=loss_operator[sys])
    Lor[sys] = liouvillan_matrix(Hor[sys], order)
    Loi[sys] = liouvillan_matrix(Hoi[sys], order)
    Lmur[sys] = liouvillan_matrix(Hmur[sys], order)
    Lmui[sys] = liouvillan_matrix(Hmui[sys], order)
    L[sys] = liouvillan_matrix(H[sys], order, loss=loss_operator[sys])

# real matrices

def hermitian_complex_to_real(order):
    n = order.shape[0]
    C = sympy.zeros(n**2)

    # diagonals are already real, so keep them as-is
    for k in range(n):
        ik = order[k,k]
        C[ik,ik] = 1

    # transform off-diagonal pairs from (z, z*) to (Re z, Im z)
    for j in range(n-1):
        for k in range(j+1, n):
            i_upper = order[j,k]
            i_lower = order[k,j]
            C[i_upper,i_upper] = sympy.Rational(1, 2)
            C[i_upper,i_lower] = sympy.Rational(1, 2)
            C[i_lower,i_upper] = -sympy.I/2
            C[i_lower,i_lower] = sympy.I/2

    return C

CtoR = hermitian_complex_to_real(order)
RtoC = CtoR.inv()

L0_real = dict()
Lor_real = dict()
Loi_real = dict()
Lmur_real = dict()
Lmui_real = dict()
L_real = dict()
for sys in systems:
    L0_real[sys] = sympy.re(CtoR * L0[sys] * RtoC)
    Lor_real[sys] = sympy.re(CtoR * Lor[sys] * RtoC)
    Loi_real[sys] = sympy.re(CtoR * Loi[sys] * RtoC)
    Lmur_real[sys] = sympy.re(CtoR * Lmur[sys] * RtoC)
    Lmui_real[sys] = sympy.re(CtoR * Lmui[sys] * RtoC)
    L_real[sys] = sympy.re(CtoR * L[sys] * RtoC)

# numerical matrices

def lambdify_wrapper(args, expr):
    return sympy.lambdify(args, expr, 'numpy', cse=True, docstring_limit=0)

def numerify_zero_args_expr(expr):
    # evil hack
    x = sympy.symbols('x')
    return lambdify_wrapper(x, expr)(None)

CtoR_num = numerify_zero_args_expr(CtoR)
RtoC_num = numerify_zero_args_expr(RtoC)

args_d = (Omega_p, Omega_mu, Omega_o)
args_h0 = (delta_mu, delta_o, Omega_p)
args_in = (Omega_mu, Omega_o)
args_decay = (gamma_12, gamma_13, gamma_23, gamma_2d, gamma_3d, n_b)
 
args_h = args_h0 + args_in
args_l0 = args_decay + args_h0
args_l = args_l0 + args_in

H0_func = dict()
Hor_num = dict()
Hoi_num = dict()
Hmur_num = dict()
Hmui_num = dict()
H_func = dict()
for sys in systems:
    H0_func[sys] = lambdify_wrapper(args_h0, H0[sys])
    Hor_num[sys] = numerify_zero_args_expr(Hor[sys])
    Hoi_num[sys] = numerify_zero_args_expr(Hoi[sys])
    Hmur_num[sys] = numerify_zero_args_expr(Hmur[sys])
    Hmui_num[sys] = numerify_zero_args_expr(Hmui[sys])
    H_func[sys] = lambdify_wrapper(args_h, H[sys])

Delta_poly_coeffs_func = dict()
for sys in systems:
    Delta_poly_coeffs_func[sys] = lambdify_wrapper(args_d, Delta_poly_coeffs[sys])

L0_func = dict()
Lor_num = dict()
Loi_num = dict()
Lmur_num = dict()
Lmui_num = dict()
L_func = dict()
for sys in systems:
    L0_func[sys] = lambdify_wrapper(args_l0, L0[sys])
    Lor_num[sys] = numerify_zero_args_expr(Lor[sys])
    Loi_num[sys] = numerify_zero_args_expr(Loi[sys])
    Lmur_num[sys] = numerify_zero_args_expr(Lmur[sys])
    Lmui_num[sys] = numerify_zero_args_expr(Lmui[sys])
    L_func[sys] = lambdify_wrapper(args_l, L[sys])

L0_real_func = dict()
Lor_real_num = dict()
Loi_real_num = dict()
Lmur_real_num = dict()
Lmui_real_num = dict()
L_real_func = dict()
for sys in systems:
    L0_real_func[sys] = lambdify_wrapper(args_l0, L0_real[sys])
    Lor_real_num[sys] = numerify_zero_args_expr(Lor_real[sys])
    Loi_real_num[sys] = numerify_zero_args_expr(Loi_real[sys])
    Lmur_real_num[sys] = numerify_zero_args_expr(Lmur_real[sys])
    Lmui_real_num[sys] = numerify_zero_args_expr(Lmui_real[sys])
    L_real_func[sys] = lambdify_wrapper(args_l, L_real[sys])

def poly_bivariate_coeffs_diff_left(coeffs):
    nx = coeffs.shape[0]-1
    ny = coeffs.shape[1]-1
    i, j = np.indices((nx, ny+1))
    return (i+1) * coeffs[1:,:]

def poly_bivariate_coeffs_diff_right(coeffs):
    nx = coeffs.shape[0]-1
    ny = coeffs.shape[1]-1
    i, j = np.indices((nx+1, ny))
    return (j+1) * coeffs[:,1:]

def poly_coeffs_diff(coeffs):
    n = len(coeffs)-1
    k = np.arange(1, n+1)
    return k * coeffs[1:]

def poly_bivariate_coeffs_evaluate_left(coeffs, x):
    n = coeffs.shape[0]-1
    k = np.arange(n+1)
    return (x**k) @ coeffs

def poly_bivariate_coeffs_evaluate_right(coeffs, y):
    n = coeffs.shape[1]-1
    k = np.arange(n+1)
    return coeffs @ (y**k)

def poly_bivariate_coeffs_evaluate(coeffs, x, y):
    i, j = np.indices(coeffs.shape)
    return np.sum(coeffs * x**i * y**j)

def poly_coeffs_evaluate(coeffs, x):
    n = len(coeffs)-1
    k = np.arange(n+1)
    return np.sum(coeffs * x**k)

def poly_coeffs_roots(coeffs):
    roots = np.polynomial.polynomial.polyroots(coeffs)
    roots = np.unique(roots)
    is_real = (np.imag(roots)==0)
    return np.real(roots[is_real])

def rho_steady_state(gamma_12, gamma_13, gamma_23, gamma_2d, gamma_3d, n_b,
                     delta_mu, delta_o, Omega_p, Omega_mu, Omega_o, sys):
    L_mtx = L_real_func[sys](
        gamma_12=gamma_12,
        gamma_13=gamma_13,
        gamma_23=gamma_23,
        gamma_2d=gamma_2d,
        gamma_3d=gamma_3d,
        n_b=n_b,
        delta_mu=delta_mu,
        delta_o=delta_o,
        Omega_p=Omega_p,
        Omega_mu=Omega_mu,
        Omega_o=Omega_o
    )
    L_mtx[0,:] = np.identity(3)[flatten]
    b = np.zeros(9)
    b[0] = 1
    rho_real = np.linalg.solve(L_mtx, b)
    rho = RtoC_num @ rho_real
    rho = rho[unflatten]
    return rho

def rho_linear_steady_state_components(gamma_12, gamma_13, gamma_23, gamma_2d, gamma_3d,
                                       n_b, delta_mu, delta_o, Omega_p, sys):
    L0_mtx = L0_real_func[sys](
        gamma_12=gamma_12,
        gamma_13=gamma_13,
        gamma_23=gamma_23,
        gamma_2d=gamma_2d,
        gamma_3d=gamma_3d,
        n_b=n_b,
        delta_mu=delta_mu,
        delta_o=delta_o,
        Omega_p=Omega_p
    )
    L0_mtx[0,:] = np.identity(3)[flatten]
    
    b = np.zeros(9)
    b[0] = 1
    rho_0_real = np.linalg.solve(L0_mtx, b)
    rho_0 = RtoC_num @ rho_0_real

    b = -Lor_real_num[sys] @ rho_0_real
    b[0] = 0
    rho_or_real = np.linalg.solve(L0_mtx, b)
    rho_or = RtoC_num @ rho_or_real

    b = -Loi_real_num[sys] @ rho_0_real
    b[0] = 0
    rho_oi_real = np.linalg.solve(L0_mtx, b)
    rho_oi = RtoC_num @ rho_oi_real

    b = -Lmur_real_num[sys] @ rho_0_real
    b[0] = 0
    rho_mur_real = np.linalg.solve(L0_mtx, b)
    rho_mur = RtoC_num @ rho_mur_real

    b = -Lmui_real_num[sys] @ rho_0_real
    b[0] = 0
    rho_mui_real = np.linalg.solve(L0_mtx, b)
    rho_mui = RtoC_num @ rho_mui_real

    rho_0 = rho_0[unflatten]
    rho_or = rho_or[unflatten]
    rho_oi = rho_oi[unflatten]
    rho_mur = rho_mur[unflatten]
    rho_mui = rho_mui[unflatten]
    return rho_0, rho_or, rho_oi, rho_mur, rho_mui

def rho_steady_state_ensemble(gamma_12, gamma_13, gamma_23, gamma_2d, gamma_3d,
                              n_b, delta_mu, delta_o, Omega_p, Omega_mu,
                              Omega_o, sigma_mu, sigma_o, sys, logging=False):
    gauss_lobatto_order = 20
    gamma_oh = gamma_2d + (gamma_3d if sys=='Lambda' else 0)
    gamma_muh = gamma_2d + (gamma_3d if sys=='V' else 0)

    # distribution envelope
    G_mu = stats.norm(loc=delta_mu, scale=sigma_mu).pdf
    G_o = stats.norm(loc=delta_o, scale=sigma_o).pdf
    G = lambda dp_mu, dp_o: G_mu(dp_mu) * G_o(dp_o)

    # set up for curve finding
    Delta_coeffs_2 = Delta_poly_coeffs_func[sys](
        Omega_p=Omega_p,
        Omega_mu=Omega_mu,
        Omega_o=Omega_o
    )
    dmu_Delta_coeffs_2 = poly_bivariate_coeffs_diff_left(Delta_coeffs_2)
    do_Delta_coeffs_2 = poly_bivariate_coeffs_diff_right(Delta_coeffs_2)

    # get integral points
    
    node_deltap_o = np.array([], dtype=float)
    node_deltap_mu = np.array([], dtype=float)
    node_weight = np.array([], dtype=float)

    # set up outer integral
    deltap_o_intervals = [delta_o + z*sigma_o
                          for z in (-10, -3, -1, 0, 1, 3, 10)]
    deltap_o_intervals += [z*gamma_oh for z in (-5, -1, 0, 1, 5)]
    deltap_o_intervals = sorted(deltap_o_intervals)
    deltap_o_nodes, deltap_o_weights = composite_gauss_lobatto_nodes_weights(
        gauss_lobatto_order, deltap_o_intervals)

    # set up inner integral
    for dp_o, w_o in zip(deltap_o_nodes, deltap_o_weights):
        dmu_Delta_coeffs = poly_bivariate_coeffs_evaluate_right(dmu_Delta_coeffs_2, dp_o)
        do_Delta_coeffs = poly_bivariate_coeffs_evaluate_right(do_Delta_coeffs_2, dp_o)
        dmu_critical = np.concatenate([
            poly_coeffs_roots(dmu_Delta_coeffs),
            poly_coeffs_roots(do_Delta_coeffs)
        ])

        deltap_mu_intervals = [delta_mu + z*sigma_mu
                               for z in (-10, -3, -1, 1, 3, 10)]
        deltap_mu_intervals += [z*gamma_muh for z in (-5, -1, 0, 1, 5)]
        deltap_mu_intervals += list(dmu_critical)
        
        deltap_mu_intervals = sorted(deltap_mu_intervals)
        deltap_mu_nodes, deltap_mu_weights = composite_gauss_lobatto_nodes_weights(
            gauss_lobatto_order, deltap_mu_intervals)

        node_deltap_o = np.concatenate([node_deltap_o, np.full_like(deltap_mu_nodes, dp_o)])
        node_deltap_mu = np.concatenate([node_deltap_mu, deltap_mu_nodes])
        node_weight = np.concatenate([node_weight, w_o*deltap_mu_weights])

    node_G = G(node_deltap_mu, node_deltap_o)

    # perform integral
    
    integral = np.zeros((3,3), dtype=complex)
    for dp_o, dp_mu, w, g in zip(node_deltap_o, node_deltap_mu, node_weight, node_G):
        integral += w*g * rho_steady_state(
            gamma_12=gamma_12,
            gamma_13=gamma_13,
            gamma_23=gamma_23,
            gamma_2d=gamma_2d,
            gamma_3d=gamma_3d,
            n_b=n_b,
            delta_mu=dp_mu,
            delta_o=dp_o,
            Omega_p=Omega_p,
            Omega_mu=Omega_mu,
            Omega_o=Omega_o,
            sys=sys
        )

    if logging:
        return (node_deltap_mu, node_deltap_o, node_weight, node_G), integral
    else:
        return integral

def Omega_cavity(g, alpha):
    return g * np.exp(1j*np.angle(alpha)) * np.sqrt(np.abs(alpha)**2 + 1)

def cavity_langevin_diff(gamma_12, gamma_13, gamma_23, gamma_2d, gamma_3d,
                         n_b, gamma_oi, gamma_oc, gamma_mui, gamma_muc,
                         N_o, N_mu, g_o, g_mu, Omega_p, alpha, beta,
                         alpha_in, beta_in, delta_mu, delta_o, delta_co,
                         delta_cmu, sigma_mu, sigma_o, sys, return_S=False):
    S = rho_steady_state_ensemble(
        gamma_12=gamma_12,
        gamma_13=gamma_13,
        gamma_23=gamma_23,
        gamma_2d=gamma_2d,
        gamma_3d=gamma_3d,
        n_b=n_b,
        delta_mu=delta_mu,
        delta_o=delta_o,
        Omega_p=Omega_p,
        Omega_o=Omega_cavity(g_o, alpha),
        Omega_mu=Omega_cavity(g_mu, beta),
        sigma_o=sigma_o,
        sigma_mu=sigma_mu,
        sys=sys
    )
    S_alpha = N_o*np.conj(g_o) * (S[1,0] if sys=='V' else S[2,1])
    S_beta = N_mu*np.conj(g_mu) * (S[2,1] if sys=='V' else S[1,0])

    d_alpha_S = -1j*S_alpha
    d_beta_S = -1j*S_beta
    d_alpha_not_S = -1j*delta_co*alpha - (gamma_oi+gamma_oc)*alpha/2 + np.sqrt(gamma_oc)*alpha_in
    d_beta_not_S = -1j*delta_cmu*beta - (gamma_mui+gamma_muc)*beta/2 + np.sqrt(gamma_muc)*beta_in
    d_alpha = d_alpha_S + d_alpha_not_S
    d_beta = d_beta_S + d_beta_not_S

    if return_S:
        return d_alpha_S, d_beta_S, d_alpha_not_S, d_beta_not_S
    else:
        return d_alpha, d_beta

def cavity_steady_state(gamma_12, gamma_13, gamma_23, gamma_2d, gamma_3d,
                        n_b, gamma_oi, gamma_oc, gamma_mui, gamma_muc,
                        N_o, N_mu, g_o, g_mu, Omega_p, alpha_in, beta_in,
                        delta_mu, delta_o, delta_co, delta_cmu, sigma_mu,
                        sigma_o, sys, alpha_0=1, beta_0=1):
    def pack_vector(alpha, beta):
        vec = np.zeros(4)
        vec[0] = np.real(alpha)
        vec[1] = np.imag(alpha)
        vec[2] = np.real(beta)
        vec[3] = np.imag(beta)
        return vec

    def unpack_vector(vec):
        alpha_r = vec[0]
        alpha_i = vec[1]
        beta_r = vec[2]
        beta_i = vec[3]
        alpha = alpha_r + 1j*alpha_i
        beta = beta_r + 1j*beta_i
        return alpha, beta
    
    def diff_func(vec):
        alpha, beta = unpack_vector(vec)
        d_alpha, d_beta = cavity_langevin_diff(
            gamma_12=gamma_12,
            gamma_13=gamma_13,
            gamma_23=gamma_23,
            gamma_2d=gamma_2d,
            gamma_3d=gamma_3d,
            n_b=n_b,
            gamma_oi=gamma_oi,
            gamma_oc=gamma_oc,
            gamma_mui=gamma_mui,
            gamma_muc=gamma_muc,
            N_o=N_o,
            N_mu=N_mu,
            g_o=g_o,
            g_mu=g_mu,
            Omega_p=Omega_p,
            alpha=alpha,
            beta=beta,
            alpha_in=alpha_in,
            beta_in=beta_in,
            delta_mu=delta_mu,
            delta_o=delta_o,
            delta_co=delta_co,
            delta_cmu=delta_cmu,
            sigma_mu=sigma_mu,
            sigma_o=sigma_o,
            sys=sys
        )
        
        d_vec = pack_vector(d_alpha, d_beta)
        return d_vec

    v0 = pack_vector(alpha_0, beta_0)
    result = optimize.root(diff_func, v0)#, method='broyden1', options={'ftol':1e-12})
    return result

def planck_excitation(T, omega):
    return 1 / np.expm1(hbar*omega/(kB*T))

if __name__ == '__main__':
    sys = 'Lambda'
    omega_12 = 2*np.pi*5.186e9
    d13 = 1.63e-32
    d23 = 1.15e-32
    tau_12 = 11
    tau_3 = 0.011
    gamma_2d = 1e6
    gamma_3d = 1e6
    sigma_o = 2*np.pi*419e6
    sigma_mu = 2*np.pi*5e6
    N = 1e16
    gamma_oi = 2*np.pi*7.95e6
    gamma_oc = 2*np.pi*1.7e6
    gamma_mui = 2*np.pi*650e3
    gamma_muc = 2*np.pi*1.5e6
    g_o = 51.9
    g_mu = 1.04

    T = 4.6
    n_b = planck_excitation(T, omega_12)
    tau_13 = tau_3 * d13**2 / (d13**2 + d23**2)
    tau_23 = tau_3 * d23**2 / (d13**2 + d23**2)
    gamma_12 = 1 / (tau_12*(n_b+1))
    gamma_13 = 1 / tau_13
    gamma_23 = 1 / tau_23
    N_o = N
    N_mu = N

    Omega_p = 35000.0

    delta_o_small = -100e3
    delta_mu_small = 1e6
    delta_o_large = -6.5*sigma_o
    delta_mu_large = 8*sigma_mu

    iterator = [(delta_o_small, delta_mu_small, 'Small detuning'),
                (delta_o_large, delta_mu_large, 'Large detuning')]
    for delta_o, delta_mu, regime in iterator:
        print(f'\n{regime} regime')
        result = cavity_steady_state(
            gamma_12=gamma_12,
            gamma_13=gamma_13,
            gamma_23=gamma_23,
            gamma_2d=gamma_2d,
            gamma_3d=gamma_3d,
            n_b=n_b,
            gamma_oi=gamma_oi,
            gamma_oc=gamma_oc,
            gamma_mui=gamma_mui,
            gamma_muc=gamma_muc,
            N_o=N_o,
            N_mu=N_mu,
            g_o=g_o,
            g_mu=g_mu,
            Omega_p=Omega_p,
            alpha_in=0.0,
            beta_in=0.0,
            delta_mu=delta_mu,
            delta_o=delta_o,
            delta_co=0.0,
            delta_cmu=0.0,
            sigma_mu=sigma_mu,
            sigma_o=sigma_o,
            sys=sys,
            alpha_0=1,
            beta_0=1
        )
        print(result)
