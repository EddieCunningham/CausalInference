import numpy as np
from scipy.special import logsumexp
from opt_einsum import contract_path
from collections import deque
import itertools
import yaml
from timeit import default_timer as timer

__all__ = [ 'log_einsum_np', 'log_einsum_path' ]

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

def log_einsum_np( contract, *args, contraction_list=None, _test=False ):
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
    unique_letters = ''.join( sorted( set( contract ) ) ).replace( ',', '' ).replace( '-', '' ).replace( '>', '' )
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
                if( len( trailing_ones ) == 0 ):
                    new_view = swapped_summed
                else:
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

    ans = log_einsum_np('tea,tfb,tabcd,tgc,thd->tefgh', C, C, I, C, C, _test=True)
    print( ( np.sin( ans )**2 ).sum() - ( np.sin(np.einsum('tea,tfb,tabcd,tgc,thd->tefgh', C, C, I, C, C ) )**2 ).sum() )

################################################################################################################

if( __name__ == '__main__' ):
    test_einsum()