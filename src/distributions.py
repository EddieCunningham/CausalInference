import jax.numpy as np
from jax import jit, vmap
from jax import random
from scipy.stats import multivariate_normal, invwishart, matrix_normal
from jax.scipy.special import digamma, gammaln

################################################################################################
# Normal: y ~ N(x|mu, sigma)

def easy_normal( d ):
    mu = np.zeros( d )
    sigma = np.eye( d )
    return normal_std_to_nat( mu, sigma )

@jit
def normal_std_to_nat( mu, sigma ):
    sigma_inv = np.linalg.inv( sigma )
    h = np.dot( sigma_inv, mu )
    J = -0.5*sigma_inv
    return J, h

@jit
def normal_nat_to_std( J, h ):
    sigma = -0.5*np.linalg.inv( J )
    mu = np.dot( sigma, h )
    return mu, sigma

@jit
def normal_expected_stats( normal ):
    mu, sigma = normal_nat_to_std( *normal )
    t1 = np.outer( mu, mu ) + sigma
    t2 = mu
    return t1, t2

def normal_sample( rng, normal, return_stat=False ):
    mu, sigma = normal_nat_to_std( *normal )
    sigma_chol = np.linalg.cholesky( sigma )
    z = random.normal( key=rng, shape=( mu.shape[0], ) )
    x = mu + np.dot( sigma_chol, z )
    if( return_stat ):
        return np.outer( x, x ), x
    return x

@jit
def normal_log_normalizer( normal ):
    J, h = normal
    mu, sigma = normal_nat_to_std( J, h )
    sigma_inv = -2*J
    k = mu.shape[0]
    return 0.5*( np.dot( mu, h ) + np.linalg.slogdet( sigma )[1] + k*np.log( 2*np.pi ) )

@jit
def normal_log_likelihood( normal, y ):
    J, h = normal
    mu, sigma = normal_nat_to_std( J, h )
    sigma_inv = -2*J
    k = mu.shape[0]
    ll = np.einsum( 'i,ij,j->', mu-y, sigma_inv, mu-y )
    ll += np.linalg.slogdet( sigma )[1] + k*np.log( 2*np.pi )
    return -0.5*ll

################################################################################################
# Regression: y ~ N(y|Ax + u, sigma)

@jit
def easy_regression( d1, d2 ):
    A = np.random.random( ( d1, d2 ) )
    sigma = np.eye( d1 )
    u = np.zeros( d1 )
    return regression_std_to_nat( A, sigma, u )

@jit
def regression_std_to_nat( A, sigma, u ):
    sigma_inv = np.linalg.inv( sigma )
    J22 = -0.5*sigma_inv
    J12 = A.T @ sigma_inv
    J11 = -0.5 * A.T @ J12.T
    h2 = np.dot( sigma_inv, u )
    h1 = -np.dot( A.T, h2 )
    return J11, J12, J22, h1, h2

@jit
def regression_nat_to_std( J11, J12, J22, h1, h2 ):
    sigma = -0.5*np.linalg.inv( J22 )
    A = ( J12 @ sigma ).T
    u = np.dot( sigma, h2 )
    return A, sigma, u

@jit
def regression_stat_dot_matrix( regression, function_of_y=True ):
    J11, J12, J22, h1, h2 = regression
    if( function_of_y ):
        A, B, C = J22, J12, h2
    else:
        A, B, C = J11, J12.T, h1
    return A, B, C

@jit
def regression_extract_blocks( mat, d ):
    # Top left
    idx = np.arange( d )
    J11 = mat[idx[:, None], idx]

    # Top right
    idx1 = np.arange( d )
    idx2 = np.arange( d, mat.shape[0] )
    J12 = mat[idx1[:, None], idx2]

    # Bottom right
    idx = np.arange( d, mat.shape[0] )
    J22 = mat[idx[:, None], idx]

    return J11, J12, J22

@jit
def regression_expected_stats( regression ):
    J11, J12, J22, h1, h2 = regression

    J = np.vstack( [ np.hstack( [ J11, 0.5*J12 ] ),
                     np.hstack( [ 0.5*J12.T, J22 ] ) ] )
    h = np.hstack( [ h1, h2 ] )
    E1, E2 = normal_expected_stats( ( J, h ) )
    E_xx, E_yx, E_yy = regression_extract_blocks( E1, d=h1.size )
    E_x, E_y = E2[:h1.size], E2[h1.size:]

    return E_xx, E_yx, E_xx, E_y, E_x

@jit
def regression_marginalize( regression, integrate_y=True, get_log_normalizer=False ):
    _J11, _J12, _J22, _h1, _h2 = regression

    if( integrate_y == False ):
        J11, J12, J22, h1, h2 = _J22, 0.5*_J12.T, _J11, _h2, _h1
    else:
        J11, J12, J22, h1, h2 = _J11, 0.5*_J12, _J22, _h1, _h2

    J22_inv = np.linalg.inv( J22 )
    J = J11 - J12.T @ J22_inv @ J12
    h = h1 - np.dot( J12.T @ J22_inv, h2 )

    return J, h

@jit
def regression_sample( rng, regression, normal_x, return_stat=False ):
    x = normal_sample( rng, normal_x )
    A, sigma, u = regression_nat_to_std( *regression )
    mu = np.dot( A, x ) + u

    sigma_chol = np.linalg.cholesky( sigma )
    k = A.shape[0]
    z = random.normal( key=rng, shape=( k, ) )
    y = mu + np.dot( sigma_chol, z )

    if( return_stat ):
        return np.outer( y, y ), np.outer( x, y ), np.outer( x, x )
    return y, x

@jit
def regression_conditioned_sample( rng, regression, x ):
    A, sigma, u = regression_nat_to_std( *regression )
    mu = np.dot( A, x ) + u

    sigma_chol = np.linalg.cholesky( sigma )
    k = A.shape[0]
    z = random.normal( key=rng, shape=( k, ) )
    y = mu + np.dot( sigma_chol, z )
    return y

@jit
def regression_likelihood_potential( regression, x ):
    J11, J12, J22, h1, h2 = regression
    J = J22
    h = h2 + np.dot( J12.T, x )
    return J, h

@jit
def regression_posterior_potential( regression, y ):
    J11, J12, J22, h1, h2 = regression
    J = J11
    h = h1 + np.dot( J12, y )
    return J, h

################################################################################################

@jit
def multigammaln( a, p ):
    ans = np.sum( [ gammaln( a + ( 1 - j - 1 ) / 2 ) for j in np.arange( p ) ] )
    return ans + np.log( np.pi ) * p * ( p - 1 ) / 4

@jit
def multigammaln_derivative( a, p ):
    return np.sum( digamma( a + ( 1 - np.arange( 1, p + 1 ) ) / 2 ) )

@jit
def easy_niw( d ):
    mu = np.zeros( d )
    kappa = 1.0
    psi = np.eye( d )
    nu = d
    return niw_std_to_nat( mu, kappa, psi, nu )

@jit
def niw_std_to_nat( mu, kappa, psi, nu ):
    n1 = kappa*np.outer( mu, mu ) + psi
    n2 = kappa*mu
    n3 = kappa
    n4 = nu + psi.shape[0] + 2
    return n1, n2, n3, n4

@jit
def niw_nat_to_std( n1, n2, n3, n4 ):
    kappa = n3
    mu = 1 / kappa * n2
    psi = n1 - kappa * np.outer( mu, mu )
    p = mu.shape[0]
    nu = n4 - p - 2
    return mu, kappa, psi, nu

@jit
def niw_log_normalizer( niw ):
    mu, kappa, psi, nu = niw_nat_to_std( *niw )
    p = psi.shape[0]

    log_z = -0.5 * nu * np.linalg.slogdet( psi )[1]
    log_z += multigammaln( nu / 2, p )
    log_z += nu * p / 2 * np.log( 2 )
    log_z += -p / 2 * np.log( kappa )
    return log_z

@jit
def niw_expected_stats( niw ):
    n1, n2, n3, n4 = niw
    p = n2.shape[0]

    k = -0.5*( n4 - p - 2 )
    P = n1 - np.outer( n2, n2 ) / n3
    P_inv = np.linalg.inv( P )

    t1 = k * P_inv
    t2 = -2 * k / n3 * np.dot( P_inv, n2 )
    t3 = k * np.einsum( 'i,ij,j->', n2, P_inv, n2 ) / n3**2 - p / ( 2 * n3 )
    t4 = -0.5 * np.linalg.slogdet( P )[1] + 0.5 * multigammaln_derivative( -k, p ) + p / 2 * np.log( 2 )
    return t1, t2, t3, t4

@jit
def niw_sample( rng, niw, return_stat=False ):
    mu_0, kappa, psi, nu = niw_nat_to_std( *niw )
    sigma = invwishart.rvs( scale=psi, df=int( nu ) )
    sigma_chol = np.linalg.cholesky( sigma / kappa )
    k = mu_0.shape[0]
    z = random.normal( key=rng, shape=( k, ) )
    mu = mu_0 + np.dot( sigma_chol, z )

    if( return_stat ):
        sigma_inv = np.linalg.inv( sigma )
        x1 = -0.5*sigma_inv
        x2 = np.dot( sigma_inv, mu )
        x3 = -0.5*np.einsum( 'i,ij,j->', mu, sigma_inv, mu )
        x4 = -0.5*np.linalg.slogdet( sigma )[1]
        return x1, x2, x3, x4

    return mu, sigma

@jit
def niw_kl_divergence( q, p ):
    stats = niw_expected_stats( q )
    kl_div = np.sum( [ np.sum( ( q[i] - p[i] ) * t ) for i, t in enumerate( stats ) ] )
    kl_div -= ( niw_log_normalizer( q ) - niw_log_normalizer( p ) )
    return kl_div

################################################################################################

@jit
def easy_mniw( d1, d2 ):
    M = np.zeros( ( d1, d2 ) )
    V = np.eye( d2 )
    psi = np.eye( d1 )
    nu = d1 + d2
    return mniw_std_to_nat( M, V, psi, nu )

@jit
def mniw_std_to_nat( M, V, psi, nu ):
    n, p = M.shape
    V_inv = np.linalg.inv( V )
    n1 = V_inv
    n2 = V_inv @ M.T
    n3 = M @ V_inv @ M.T + psi
    n4 = nu + n + p + 1
    return n1, n2, n3, n4

@jit
def mniw_nat_to_std( n1, n2, n3, n4 ):
    p, n = n2.shape
    V = np.linalg.inv( n1 )
    M = n2.T @ V
    psi = n3 - M @ n1 @ M.T
    nu = n4 - 1 - n - p
    return M, V, psi, nu

@jit
def mniw_log_normalizer( mniw ):
    M, V, psi, nu = mniw_nat_to_std( *mniw )
    n = M.shape[0]

    log_z = -nu / 2 * np.linalg.slogdet( psi )[1]
    log_z += multigammaln( -0.5*nu, n )
    log_z += nu * n / 2 * np.log( 2 )
    log_z += n / 2 * np.linalg.slogdet( V )[1]
    return log_z

@jit
def mniw_expected_stats( mniw ):
    n1, n2, n3, n4 = mniw
    p, n = n2.shape

    k = ( n4 - 1 - n - p ) / 2
    n1_inv = np.linalg.inv( n1 )
    P = n3 - n2.T @ n1_inv @ n2
    P_inv = np.linalg.inv( P )

    t1 = -k * ( n1_inv @ n2 @ P_inv @ n2.T @ n1_inv ).T - n / 2 * n1_inv.T
    t2 = 2 * k * n1_inv @ n2 @ P_inv
    t3 = -k * P_inv
    t4 = -0.5 * np.linalg.slogdet( P )[1] + 0.5 * multigammaln_derivative( k, n ) + n / 2 * np.log( 2 )
    return t1, t2, t3, t4

@jit
def mniw_sample( mniw, return_stat=False ):
    M, V, psi, nu = mniw_nat_to_std( *mniw )

    sigma = invwishart.rvs( scale=psi, df=int( nu ) )
    A = matrix_normal.rvs( mean=M, rowcov=sigma, colcov=V )

    if( return_stat == True ):
        sigma_inv = np.linalg.inv( sigma )
        x1 = -0.5 * A.T @ sigma_inv @ A
        x2 = A.T @ sigma_inv
        x3 = -0.5 * sigma_inv
        x4 = -0.5 * np.linalg.slogdet( sigma )[1]
        return x1, x2, x3, x4

    return A, sigma

@jit
def mniw_kl_divergence( q, p ):
    stats = mniw_expected_stats( q )
    kl_div = np.sum( [ np.sum( ( q[i] - p[i] ) * t ) for i, t in enumerate( stats ) ] )
    kl_div -= ( mniw_log_normalizer( q ) - mniw_log_normalizer( p ) )
    return kl_div