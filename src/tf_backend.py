import numpy as np
from scipy.special import logsumexp
from opt_einsum import contract_path
from collections import deque
import itertools
import yaml
from timeit import default_timer as timer
import tensorflow as tf

__all__ = [ 'log_einsum_tf' ]

################################################################################################################

def real_product( x, y ):
    return x * y

def real_integrate( x, axis ):
    return tf.reduce_sum( x, axis=axis )

def log_product( x, y ):
    return x + y

def log_integrate( x, axis ):
    return tf.reduce_logsumexp( x, axis=axis )

################################################################################################################

def log_einsum_tf( contract, *args, contraction_list=None, _test=False ):
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

            reshaped_left_no_transpose = tf.reshape( left_operand, tuple( shape_left ) )
            reshaped_right_no_transpose = tf.reshape( right_operand, tuple( shape_right ) )

            reshaped_left  = tf.transpose( reshaped_left_no_transpose, transpose_left )
            reshaped_right = tf.transpose( reshaped_right_no_transpose, transpose_right )

            # Sum up the terms
            summed = product( reshaped_left, reshaped_right )

            # Transpose the output back and put the removal indices on the last axes
            not_in_result = ''.join( [ letter for letter in unique_letters if letter not in results_index ] )
            full_results_index = results_index + not_in_result
            for i, letter in enumerate( full_results_index ):
                transpose_back[i] = unique_letters.index( letter )

            swapped_summed = tf.transpose( summed, tuple( transpose_back ) )

            # Integrate out terms if needed
            if( len( idx_rm ) > 0 ):
                remove_idx = tuple( list( range( len( results_index ), n_unique_letters ) ) )
                new_view = integrate( swapped_summed, axis=remove_idx )
            else:
                # Don't squeeze the first dim!  This messes things up if we have a batch size of 1!
                trailing_ones = tuple( [ i for i, s in enumerate( swapped_summed.shape ) if s == 1 and i > 0 ] )
                new_view = tf.squeeze( swapped_summed, axis=trailing_ones )

        else:

            # Then we just need to do an integration step
            remove_idx = tuple( [ input_str.index( letter ) for letter in idx_rm ] )
            new_view = integrate( tmp_operands[0], axis=remove_idx )

        # Append new items and dereference what we can
        operands.append( new_view )
        del tmp_operands, new_view

    if( _test == True ):
        check = tf.einsum( contract, *args )
        diff = operands[0]**2 - check**2
        assert np.isclose( tf.reduce_sum( diff ).numpy(), 0.0, atol=1e-6 )

    return operands[0]

################################################################################################################

def test_einsum():
    I = tf.constant( np.random.rand( 1, 10, 10, 10, 10 ) )
    C = tf.constant( np.random.rand( 1, 10, 10 ) )

    ans = log_einsum_tf('tea,tfb,tabcd,tgc,thd->tefgh', C, C, I, C, C, _test=True)
    print( 'Passed!' )

################################################################################################################

if( __name__ == '__main__' ):
    from host.debug import *
    tf.enable_eager_execution()
    test_einsum()