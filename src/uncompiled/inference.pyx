import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool as bool_t
from libcpp.unordered_set cimport unordered_set
from compiled.definitions cimport *
from compiled.traversal cimport *

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef inferenceInstructions( int[:, :] edge_parents,
                             int[:, :] edge_children,
                             int[:, :] node_meta,
                             int[:, :] edge_meta,
                             int[:]    graph_meta,
                             int[:]    brute_force ):
    """
    This file will return the instructions needed to perform inference
    over Bayesian networks.  It will return the list of computations needed
    to actually perform inference.

    The algorithm is for polytrees, however if you brute force over
    a feedback vertex set, it will work on multiply-connected
    graphs.

    If a directed cycle is count and the brute_force set is insufficient, the
    algorithm will default to loopy propagation belief (not implemented yet)

    In order to use the juntion tree algorithm, the user must hand this
    file the graph decomposition.

    At each computation, at each step the user needs to know the following:
        - Which nodes to integrate
        - Which nodes to evaluate a potential over
        - The type of message being computed
        - Which messages are needed

    Args:
        edge_parents  : Info about edges' parent nodes
        edge_children : Info about edges' child nodes
        node_meta     : Meta data about nodes
        edge_meta     : Meta data about the edges
        graph_meta    : Meta data about the graph
        brute_force   : Will not integrate these out

    Returns:
        u_output_order
            - target node
            - potential over which nodes
            - who to integrate
            - u for which nodes
            - v for which nodes
        v_output_order
            - target node and edge
            - potential over which nodes
            - who to integrate
            - u for which nodes
            - v for which nodes
    """
    cdef int[:]                 u_count
    cdef int[:]                 v_count
    cdef int[:]                 u_output_order
    cdef int[:, :]              v_output_order
    cdef vector[pair[int, int]] batch_sizes
    cdef unordered_set[int]     brute_force_set

    cdef int[:] edge_parents_buffer  = np.empty( graph_meta[MAX_PARENTS], dtype=np.int32 )
    cdef int[:] edge_children_buffer = np.empty( graph_meta[MAX_CHILDREN], dtype=np.int32 )
    cdef int[:] child_edges_buffer   = np.empty( graph_meta[MAX_CHILD_EDGES], dtype=np.int32 )

    # We want a u for every node
    u_count        = np.zeros( node_meta.shape[1], dtype=np.int32 )
    u_output_order = np.zeros( node_meta.shape[1], dtype=np.int32 )

    # We want a v for every [parent, child edge] combo
    v_count        = np.zeros( edge_parents.shape[1], dtype=np.int32 )
    v_output_order = np.zeros( ( edge_parents.shape[1], 2 ), dtype=np.int32 )

    # Populate the brute force set
    with nogil:
        for i in range( brute_force.shape[0] ):
            brute_force_set.insert( brute_force[i] )

    # Find the order that we need to visit the nodes
    batch_sizes = fastMessagePassing( edge_parents,
                                      edge_children,
                                      node_meta,
                                      edge_meta,
                                      graph_meta,
                                      brute_force_set,
                                      u_output_order,
                                      v_output_order,
                                      u_count,
                                      v_count,
                                      edge_parents_buffer,
                                      edge_children_buffer,
                                      child_edges_buffer )

    # Add in the extra information for inference
    pass

    return np.asarray( u_output_order ), np.asarray( v_output_order ), batch_sizes