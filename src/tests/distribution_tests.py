import jax.numpy as np
from jax import jit, vmap
from jax import random
from jax.util import partial
from scipy.stats import multivariate_normal, invwishart, matrix_normal
from jax.scipy.special import digamma, gammaln
from host.src.distributions import *
from host.debug import *

def test_nat_to_std( nat_to_std, std_to_nat, dist ):
    std = nat_to_std( *dist )
    nat = std_to_nat( *std )
    std2 = nat_to_std( *nat )
    for s1, s2 in zip( std, std2 ):
        assert np.allclose( s1, s2 )
    for n1, n2 in zip( nat, dist ):
        assert np.allclose( n1, n2 )
    print( 'Passed the nat to std test!' )

def test_expected_stats( expected_stats, sample, dist, N=1000 ):

    true_stats = expected_stats( dist )

    emperical_t = [ np.zeros_like( t ) for t in true_stats ]

    key = random.PRNGKey( 0 )

    for _ in range( N ):
        _, key = random.split( key )
        xs = sample( key, dist, return_stat=True )
        for i, x in enumerate( xs ):
            emperical_t[i] += x

    for i in range( len( emperical_t ) ):
        emperical_t[i] /= N

    for t, est_t in zip( true_stats, emperical_t ):
        assert np.allclose( t, est_t, atol=2e-1 )

    print( 'Passed the expected stats test!' )

################################################################################################

def test_distributions():
    d1 = 2
    d2 = 3

    normal = easy_normal( d1 )
    test_nat_to_std( normal_nat_to_std, normal_std_to_nat, normal )
    test_expected_stats( normal_expected_stats, normal_sample, normal )

    regression = easy_regression( d1, d2 )
    test_nat_to_std( regression_nat_to_std, regression_std_to_nat, regression )
    # test_expected_stats( regression_expected_stats, regression_sample, regression )

    niw = easy_niw( d1 )
    test_nat_to_std( niw_nat_to_std, niw_std_to_nat, niw )
    test_expected_stats( niw_expected_stats, niw_sample, niw )

    mniw = easy_mniw( d1, d2 )
    test_nat_to_std( mniw_nat_to_std, mniw_std_to_nat, mniw )
    test_expected_stats( mniw_expected_stats, mniw_sample, mniw )

if( __name__ == '__main__' ):
    test_distributions()