import numpy as np
from scipy.special import logsumexp
from opt_einsum import contract_path
from collections import deque
import itertools
import yaml
from timeit import default_timer as timer

__all__ = [ 'log_einsum', 'log_einsum_path' ]

################################################################################################################

def real_product( x, y ):
    return x * y

def real_integrate( x, axis ):
    return np.sum( x, axis=axis )

def log_product( x, y ):
    return x + y

def log_integrate( x, axis ):
    return logsumexp( x, axis=axis )

################################################################################################################

class DeviceProfiler():

    total_complexity = 0

    def __init__( self, device_name ):

        if( device_name is None ):
            device_name = 'macbook.yaml'

        self.device_name = device_name
        self.device_path = './host/device_profiles/%s'%( device_name )
        if( self.device_path.endswith( '.yaml' ) == False ):
            self.device_path = self.device_path + '.yaml'

        if( os.path.exists( self.device_path ) ):
            with open( self.device_path, 'w' ) as yaml_file:
                yaml_data = yaml.load( yaml_file )
                self.m = yaml_data['coefficients']
        else:
            self.m = None

    def reset( self ):
        DeviceProfiler.total_complexity = 0

    def log_product( self, x, y ):
        DeviceProfiler.total_complexity += 0
        return log_product( x, y )

    def log_integrate( self, x, axis ):
        assert x.ndim == 2
        for i, ax in enumerate( axis ):
            DeviceProfiler.total_complexity += self.polyval2d( x.shape[0] - i, x.shape[ax] )
        return log_integrate( x, axis )

    def polyfit2d( self, x, y, z, order=3 ):
        ncols = ( order + 1 )**2
        G = np.zeros( ( x.size, ncols ) )
        ij = itertools.product( range( order+1 ), range( order+1 ) )
        for k, ( i,j ) in enumerate( ij ):
            G[:,k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq( G, z )
        return m

    def polyval2d( self, x, y ):
        order = int( np.sqrt( len( self.m ) ) ) - 1
        ij = itertools.product( range( order+1 ), range( order+1 ) )
        z = np.zeros_like( x )
        for a, ( i,j ) in zip( self.m, ij ):
            z += a * x**i * y**j
        return z

    def profile( self ):
        # Taken from here https://stackoverflow.com/a/7997925

        # Addition is basically instant, so only look at logsumexp
        non_integrate_dims = np.linspace( 10, 10000, 10 )
        integrate_dims     = np.linspace( 10, 10000, 10 )

        # See how long it takes to logsumexp different sized arrays
        integration_times = np.zeros( ( 10 * 10, 3 ) )
        for i, ( non_int_dim, int_dim ) in enumerate( itertools.product( non_integrate_dims, integrate_dims ) ):
            x = np.random.random( ( non_int_dim, int_dim ) )

            times = np.zeros( 3 )
            for i in range( 3 ):
                start = timer()
                log_integrate( x, axis=1 )
                times[i] = timer() - start

            integration_times[i] = np.array( [ non_int_dim, int_dim, np.mean( times ) ] )

        # Do a polynomial fit
        m = self.polyfit2d( integration_times[:, 0], integration_times[:, 1], integration_times[:, 2] )

        # Save off the coefficients
        yaml_data = { 'device_name': self.device_name, 'coefficients': m }
        with open( self.device_path, 'w' ) as yaml_file:
            yaml.dump( yaml_data, yaml_file )

        import matplotlib.pyplot as plt
        plt.switch_backend( 'agg' )

        x = integration_times[:, 0]
        y = integration_times[:, 1]
        z = integration_times[:, 2]

        # Evaluate it on a grid...
        nx, ny = 20, 20
        xx, yy = np.meshgrid( np.linspace(x.min(), x.max(), nx ),
                              np.linspace(y.min(), y.max(), ny ) )
        zz = polyval2d( xx, yy, m )

        # Plot
        plt.imshow( zz, extent=( x.min(), y.max(), x.max(), y.min() ) )
        plt.scatter( x, y, c=z )

        plt.savefig( './host/integration_plot.png' )


################################################################################################################

def log_einsum_path( contract, *args ):
    """ Return the contraction list for args that have the given shapes
        Also returns the shape of the output

        Args:
            contract         - The contraction to perform
            shapes           - The shapes of the tensors to contract

        Returns:
            The shape of the result and the contraction order
    """
    # contract_path requires a numpy array
    shapes = [ arg.shape for arg in args ]
    # args = [ np.empty( shape ) for shape in shapes ]
    _, contraction_list = contract_path( contract, *args, einsum_call=True, optimize='auto' )

    # Compute the output shape
    unique_letters = ''.join( sorted( set( contract ) ) ).replace( ',', '' )
    input_str, result = contract.split( '->' )
    letter_sizes = [ 0 for _ in unique_letters ]
    for operand, shape in zip( input_str.split( ',' ), shapes ):
        for letter, size in zip( operand, shape ):
            letter_sizes[unique_letters.index( letter )] = size

    output_shape = tuple( [ letter_sizes[unique_letters.index( letter ) ] for letter in result ] )
    return np.empty( output_shape ), contraction_list

################################################################################################################

def log_einsum( contract, *args, contraction_list=None, _test=False, _profile=False, device_name=None ):
    """ Taken from here https://github.com/dgasmith/opt_einsum/blob/master/opt_einsum/contract.py
        but instead of ( multiply, sum ), will do ( sum, logsumexp ).  So this is just einsum in log space.

        Args:
            contract         - The contraction to perform
            args             - The tensors to contract
            contraction_list - Pre-computed the contraction order
            _test            - If we want to test this function

        Returns:
            The result of the contraction
    """
    if( _profile == True ):
        # If we want to profile the code, this will count how expensive
        # it is to evaluate a computation
        device_profiler = DeviceProfiler( device_name )
        def product( x, y ):
            return device_profiler.log_product( x, y )

        def integrate( x, axis ):
            return device_profiler.log_integrate( x, axis )
    else:
        # Make it easy to test against a correct einsum implementation
        if( _test == True ):
            def product( x, y ):
                return real_product( x, y )

            def integrate( x, axis ):
                return real_integrate( x, axis )
        else:
            def product( x, y ):
                return log_product( x, y )

            def integrate( x, axis ):
                return log_integrate( x, axis )

    # If we haven't passed in the contraction list, find it
    if( contraction_list is None ):
        _, contraction_list = contract_path( contract, *args, einsum_call=True, optimize='auto' )

    operands = list( args )

    # Find the unique letters in the contract and allocate a list for the final transpose
    unique_letters = ''.join( sorted( set( contract ) ) ).replace( ',', '' )
    n_unique_letters = len( unique_letters )
    transpose_back = [ 0 for _ in unique_letters ]

    # Start contraction loop
    for num, ( inds, idx_rm, einsum_str, remaining, _ ) in enumerate( contraction_list ):

        # Retrieve the current operands and get split the contract
        tmp_operands = [ operands.pop(x) for x in inds ]
        input_str, results_index = einsum_str.split( '->' )

        # Check if we should multiply and then contract
        if( len( inds ) > 1 ):
            left_operand, right_operand = tmp_operands
            input_left, input_right = input_str.split( ',' )

            # Want to transpose the operands to be in alphabetical order so that multiplying them is easy
            not_in_left  = ''.join( [ letter for letter in unique_letters if letter not in input_left ] )
            not_in_right = ''.join( [ letter for letter in unique_letters if letter not in input_right ] )
            left_shape  = input_left + not_in_left
            right_shape = input_right + not_in_right

            # Align operands on the correct axes in order to do the sum
            transpose_left  = tuple( [ left_shape.index( letter ) for letter in unique_letters ] )
            transpose_right = tuple( [ right_shape.index( letter ) for letter in unique_letters ] )

            # Extend the axes of the operands and transpose them
            shape_left  = list( left_operand.shape )  + [ 1 for _ in range( len( left_operand.shape ), n_unique_letters ) ]
            shape_right = list( right_operand.shape ) + [ 1 for _ in range( len( right_operand.shape ), n_unique_letters ) ]
            reshaped_left  = left_operand.reshape( tuple( shape_left ) ).transpose( transpose_left )
            reshaped_right = right_operand.reshape( tuple( shape_right ) ).transpose( transpose_right )

            # Sum up the terms
            summed = product( reshaped_left, reshaped_right )

            # Transpose the output back and put the removal indices on the last axes
            not_in_result = ''.join( [ letter for letter in unique_letters if letter not in results_index ] )
            full_results_index = results_index + not_in_result
            for i, letter in enumerate( full_results_index ):
                transpose_back[i] = unique_letters.index( letter )

            swapped_summed = summed.transpose( tuple( transpose_back ) )

            # Integrate out terms if needed
            if( len( idx_rm ) > 0 ):
                remove_idx = tuple( list( range( len( results_index ), n_unique_letters ) ) )
                new_view = integrate( swapped_summed, axis=remove_idx )
            else:
                # Don't squeeze the first dim!  This messes things up if we have a batch size of 1!
                trailing_ones = tuple( [ i for i, s in enumerate( swapped_summed.shape ) if s == 1 and i > 0 ] )
                new_view = swapped_summed.squeeze( axis=trailing_ones )

        else:

            # Then we just need to do an integration step
            remove_idx = tuple( [ input_str.index( letter ) for letter in idx_rm ] )
            new_view = integrate( tmp_operands[0], axis=remove_idx )

        # Append new items and dereference what we can
        operands.append( new_view )
        del tmp_operands, new_view

    if( _test == True ):
        check = np.einsum( contract, *args )
        assert np.allclose( check, operands[0] )

    return operands[0]

################################################################################################################

def test_einsum():
    I = np.random.rand( 1, 10, 10, 10, 10 )
    C = np.random.rand( 1, 10, 10 )
    operands, contraction_list = contract_path( 'tea,tfb,tabcd,tgc,thd->tefgh', C, C, I, C, C, einsum_call=True )

    ans = log_einsum('tea,tfb,tabcd,tgc,thd->tefgh', C, C, I, C, C, _test=True)
    print( ( np.sin( ans )**2 ).sum() - ( np.sin(np.einsum('tea,tfb,tabcd,tgc,thd->tefgh', C, C, I, C, C ) )**2 ).sum() )

################################################################################################################

if( __name__ == '__main__' ):
    test_einsum()