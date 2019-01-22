import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.pair cimport pair
from libcpp.deque cimport deque
from libc.limits cimport INT_MAX
from libcpp cimport bool

__all__ = [ 'message_passing' ]

"""
The functions in this file allow efficient message passing over undirected
polyforests.

See definitions.pxd for an overview of the data structures
"""

cdef int INVALID_INDEX = INT_MAX

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef int get_edge( int polyforest_index ) nogil except *:
    return polyforest_index // 2

@cython.boundscheck( False )
@cython.wraparound( False )
cdef int get_relative_index( int polyforest_index ) nogil except *:
    return polyforest_index + 1 if polyforest_index % 2 == 0 else polyforest_index - 1

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void polyforest_message_passing( int[:]       polyforest,
                                      vector[int]& messages,
                                      int[:]       root_indices,
                                      vector[int]& batch_sizes,
                                      int          n_nodes=-1,
                                      bool         loopy=False,
                                      int          loopy_stop=-5 ) nogil except *:
    """ Message passing over polyforests.  This will ideally be a clique tree.
        Intuitively, each node within an edge is the link to other edges.

        Args:
            polyforest    : This contains the adjacency matrix for the hypergraph.

        Returns:
            messages      : The correct order that the edges are to be processed
            batch_sizes   : The number of elements per batch.  Elements within a batch
                            can be processed in parallel.
    """

    # The final messages will have exactly as many elements as the polyforest.
    cdef int[:] message_counts
    cdef int last_message_index = 0
    cdef int message_index = 0

    cdef int recorded_index
    cdef int i, j
    cdef int tree_index
    cdef int progress
    cdef int n_edges
    cdef int node_index
    cdef int relative_index
    cdef int related_node
    cdef int edge_start
    cdef int family_index
    cdef int family_edge
    cdef int current_node, current_edge
    cdef int other_edge
    cdef int root_index

    cdef bool loopy_force = False

    cdef int polyforest_length = polyforest.shape[0]
    cdef int TMP_STORAGE = NODE_INDEX

    cdef int[:]      linked_sparse
    cdef int[:, :]   node_meta

    cdef bool loopy_valid
    cdef bool mpp_valid

    ############################################################
    # If the user didn't pass in the number of nodes, search for it
    if( n_nodes == -1 ):
        n_nodes = 0
        for i in range( polyforest_length ):
            current_node = polyforest[i]
            if( current_node > n_nodes ):
                n_nodes = current_node
        n_nodes += 1

    with gil:
        message_counts = np.empty_like( polyforest, dtype=np.int32 )
        linked_sparse = np.full( polyforest_length, INVALID_INDEX, dtype=np.int32 )
        node_meta     = np.full( ( 2, n_nodes ), INVALID_INDEX, dtype=np.int32 )

    ############################################################
    # Create the linked list that points from edge to edge
    # Go backwards through the polyforest and store the next index for every index
    for i in range( polyforest_length ):
        j = polyforest_length - i - 1
        current_node = polyforest[j]
        if( node_meta[TMP_STORAGE, current_node] != INVALID_INDEX ):
            linked_sparse[j] = node_meta[TMP_STORAGE, current_node]
        node_meta[TMP_STORAGE, current_node] = j

    ############################################################
    # Fill in node_meta
    for i in range( polyforest_length ):
        current_node = polyforest[i]

        # Increment the number of edges for the node and check if
        # this edge is the first edge
        if( node_meta[N_EDGES, current_node] == INVALID_INDEX ):
            node_meta[N_EDGES, current_node] = 0
        node_meta[N_EDGES, current_node] += 1

        if( node_meta[NODE_INDEX, current_node] == INVALID_INDEX ):
            node_meta[NODE_INDEX, current_node] = i

    ############################################################
    # Initialize the semaphore
    for i in range( polyforest_length ):
        current_edge = get_edge( i )
        current_node = polyforest[i]

        # We have to wait on all other edges
        message_counts[i] = node_meta[N_EDGES, current_node] - 1

    ############################################################
    # Find the leaves (the ones that unique nodes)
    for current_node in range( n_nodes ):

        # Nodes with unique nodes are base case nodes
        if( node_meta[N_EDGES, current_node] == 1 ):

            # Record the base case nodes
            messages[message_index] = node_meta[NODE_INDEX, current_node]
            message_index += 1

    # If there are no leaves (only cycles), then just use the first node.
    # Otherwise, record the root indices
    if( message_index == 0 ):
        for i in range( polyforest_length ):
            current_node = polyforest[i]
            if( current_node == 0 ):
                messages[message_index] = i
                message_index += 1
    else:
        for i in range( root_indices.shape[0] ):
            root_index = root_indices[i]
            messages[message_index] = root_index
            message_index += 1

    # Record the first batch size (base case)
    batch_sizes.push_back( message_index )

    ############################################################
    # Complete the message passing algorithm
    while( True ):
        progress = message_index

        ############################################################
        # Update the semaphores related to the last completed iteration
        for i in range( last_message_index, progress ):
            recorded_index = messages[i]
            current_edge = get_edge( recorded_index )
            related_node = polyforest[get_relative_index( recorded_index )]

            # For this node's relative, decrement a count for all of its edges
            relative_index = node_meta[NODE_INDEX, related_node]
            while( relative_index != INVALID_INDEX ):
                other_edge = get_edge( relative_index )

                if( other_edge != current_edge ):

                    message_counts[relative_index] -= 1

                    # If we are using lpb, enter this if we haven't traversed this node too many times.
                    # Otherwise, only enter this if we have received messages from all other edges.
                    loopy_valid = ( loopy_force == True ) and ( message_counts[relative_index] >= loopy_stop )
                    mpp_valid   = message_counts[relative_index] == 0

                    if( loopy_valid or mpp_valid ):

                        # Record this message and increment the index
                        if( message_index < messages.size() ):
                            messages[message_index] = relative_index
                        else:
                            messages.push_back( relative_index )
                        message_index += 1

                relative_index = linked_sparse[relative_index]

        ############################################################
        # Check if we're done
        if( progress == message_index ):
            # Check if we should resort to loopy propagation belief
            if( message_index != polyforest_length and loopy_force == False and loopy == True ):
                loopy_force = True
                continue
            else:
                break

        # Record the last back size
        batch_sizes.push_back( message_index - progress )

        # Update the last_message_index
        last_message_index = progress

###################################################################################

def message_passing( polyforest,
                     root_indices,
                     n_nodes=-1,
                     loopy=False,
                     loopy_iters=5 ):
    """ Wrapper around message_passing_cython that will take care of types

        Args:
            polyforest    : This contains the adjacency matrix for the hypergraph.
            root_indices  : The indices in polyforest that should be computed first
            n_nodes       : Optional.  The number of nodes in polyforest
            loopy         : Whether or not to run loopy propagation belief
            loopy_iters   : When to stop loopy propagation belief

        Returns:
            messages      : The correct order that the edges are to be processed
            batch_sizes   : The number of elements per batch.  Elements within a batch
                            can be processed in parallel.
    """
    return message_passing_cython( polyforest.astype( np.int32 ),
                                   root_indices.astype( np.int32 ),
                                   <int>n_nodes,
                                   <bool>loopy,
                                   <int>loopy_iters )

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef message_passing_cython( int[:] polyforest,
                              int[:] root_indices,
                              int    n_nodes=-1,
                              bool   loopy=False,
                              int    loopy_iters=5 ):
    """ Run message passing over the polyforest.  The algorithm starts at both
        the leaves of polyforest, and at root_indices.

        Args:
            polyforest    : This contains the adjacency matrix for the hypergraph.
            root_indices  : The indices in polyforest that should be computed first
            n_nodes       : Optional.  The number of nodes in polyforest
            loopy         : Whether or not to run loopy propagation belief
            loopy_iters   : When to stop loopy propagation belief

        Returns:
            messages      : The correct order that the edges are to be processed
            batch_sizes   : The number of elements per batch.  Elements within a batch
                            can be processed in parallel.
    """
    cdef vector[int] messages
    cdef vector[int] batch_sizes

    ############################################################
    # Allocate the memory here.  We will only need this much
    # memory if we don't do loopy propagation
    messages.resize( polyforest.shape[0] )

    ############################################################
    # Run the message passing algorithm
    polyforest_message_passing( polyforest,
                                messages,
                                root_indices,
                                batch_sizes,
                                n_nodes=n_nodes,
                                loopy=loopy,
                                loopy_stop=-loopy_iters )

    return np.asarray( messages ), batch_sizes