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
from compiled.traversal import *
from cython.operator cimport dereference as deref, preincrement as inc

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef cutMessagePassing( int[:, :] edge_parents,
                         int[:, :] edge_children,
                         int[:, :] node_meta,
                         int[:, :] edge_meta,
                         int[:]    graph_meta,
                         int[:]    brute_force ):
    """
    Perform message passing while brute forcing nodes in brute_force

    Args:
        edge_parents  : Info about edges' parent nodes
        edge_children : Info about edges' child nodes
        node_meta     : Meta data about nodes
        edge_meta     : Meta data about the edges
        graph_meta    : Meta data about the graph
        brute_force   : Will not integrate these out

    Returns:
        None
    """
    cdef int edge_offset
    cdef int last_edge
    cdef int cut_edge
    cdef int edge
    cdef int node
    cdef int current_edge
    cdef int i
    cdef int j = 0
    cdef int k = 0
    cdef int mapped_index
    cdef int[:] node_mapping         = np.zeros( node_meta.shape[1], dtype=np.int32 )
    cdef int[:] reverse_node_mapping = np.zeros( node_meta.shape[1], dtype=np.int32 )
    cdef int[:] edge_mapping         = np.zeros( edge_meta.shape[1], dtype=np.int32 )
    cdef int[:] reverse_edge_mapping = np.zeros( edge_meta.shape[1], dtype=np.int32 )
    cdef int[:, :] cut_edge_parents  = np.zeros( ( 2, edge_parents.shape[1] ), dtype=np.int32 )
    cdef int[:, :] cut_edge_children = np.zeros( ( 2, edge_children.shape[1] ), dtype=np.int32 )
    cdef unordered_set[int] brute_force_set
    cdef unordered_set[int] edge_set
    cdef unordered_set[int].iterator edge_set_it

    with nogil:

        # Fill the brute force set
        for i in range( brute_force.shape[0] ):
            brute_force_set.insert( brute_force[i] )

        for node in range( node_meta.shape[1] ):
            if( brute_force_set.find( node ) != brute_force_set.end() ):
                node_mapping[node] = -1
            else:
                node_mapping[node] = j
                j += 1

        # Fill in the edge set
        for edge in range( edge_meta.shape[1] ):
            edge_set.insert( edge )

        # Fill cut_edge_parents
        j = 0
        for i in range( edge_parents.shape[1] ):
            edge = edge_parents[EDGE, i]
            node = edge_parents[NODE, i]

            if( brute_force_set.find( node ) == brute_force_set.end() ):
                cut_edge_parents[EDGE, j] = edge
                cut_edge_parents[NODE, j] = node_mapping[node]
                j += 1

                # Remove this edge
                edge_set_it = edge_set.find( edge )
                if( edge_set_it != edge_set.end() ):
                    edge_set.erase( edge_set_it )

        # Fill cut_edge_children
        k = 0
        for i in range( edge_children.shape[1] ):
            edge = edge_children[EDGE, i]
            node = edge_children[NODE, i]

            if( brute_force_set.find( node ) == brute_force_set.end() ):
                cut_edge_children[EDGE, k] = edge
                cut_edge_children[NODE, k] = node_mapping[node]
                k += 1

                # Remove this edge
                edge_set_it = edge_set.find( edge )
                if( edge_set_it != edge_set.end() ):
                    edge_set.erase( edge_set_it )

        # Find the new indices for the edges
        current_edge = 0
        for edge in range( edge_meta.shape[1] ):
            if( edge_set.find( edge ) == edge_set.end() ):
                edge_mapping[edge] = current_edge
                current_edge += 1
            else:
                edge_mapping[edge] = -1

        # Re-index the edges to account for the completely missing edges
        for i in range( cut_edge_parents.shape[1] ):
            edge = cut_edge_parents[EDGE, i]
            cut_edge_parents[EDGE, i] = edge_mapping[ edge ]

        # Do the exact same thing, but with the edge_children
        for i in range( cut_edge_children.shape[1] ):
            edge = cut_edge_children[EDGE, i]
            cut_edge_children[EDGE, i] = edge_mapping[ edge ]

    # Trim the views
    cut_edge_parents  = cut_edge_parents[:, :j]
    cut_edge_children = cut_edge_children[:, :k]

    processed_results = preprocessSparseGraphForTraversal( cut_edge_parents, cut_edge_children )
    edge_parents, edge_children, node_meta, edge_meta, graph_meta = processed_results

    with nogil:
        # Find the reversed edge mapping
        for i in range( edge_mapping.shape[0] ):
            mapped_index = edge_mapping[i]

            if( mapped_index != -1 ):
                reverse_edge_mapping[ mapped_index ] = i

        # Find the reversed node mapping
        for i in range( node_mapping.shape[0] ):
            mapped_index = node_mapping[i]

            if( mapped_index != -1 ):
                reverse_node_mapping[ mapped_index ] = i

    return processed_results, reverse_edge_mapping, reverse_node_mapping

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef cutsetMessagePassing( int[:, :] edge_parents,
                            int[:, :] edge_children,
                            int[:, :] node_meta,
                            int[:, :] edge_meta,
                            int[:]    graph_meta,
                            int[:]    brute_force ):
    """
    Return the message passing order for graph that does not traverse nodes
    in brute force

    Args:
        edge_parents  : Info about edges' parent nodes
        edge_children : Info about edges' child nodes
        node_meta     : Meta data about nodes
        edge_meta     : Meta data about the edges
        graph_meta    : Meta data about the graph
        brute_force   : Will not integrate these out

    Returns:
        u_output_order : The order nodes are visited for computing U
        v_output_order : The order (edge,node) are visited in edge_parents for computing V
        batch_sizes    : The number of elements per batch.  Elements within a batch
                         can be processed in parallel.  First element is the elements
                         to take from U and second is the elements to take from V
    """
    cdef int[:]                 u_count
    cdef int[:]                 v_count
    cdef int[:]                 u_output_order
    cdef int[:, :]              v_output_order
    cdef vector[pair[int, int]] batch_sizes
    cdef unordered_set[int]     brute_force_set
    cdef int[:] edge_parents_buffer
    cdef int[:] edge_children_buffer
    cdef int[:] child_edges_buffer
    cdef int i
    cdef int[:] reverse_node_mapping = np.zeros( node_meta.shape[1], dtype=np.int32 )
    cdef int[:] reverse_edge_mapping = np.zeros( edge_meta.shape[1], dtype=np.int32 )

    # We want to do message passing on the graph cut by brute_force
    cut_results =  cutMessagePassing( edge_parents,
                                      edge_children,
                                      node_meta,
                                      edge_meta,
                                      graph_meta,
                                      brute_force )

    ( cut_edge_parents, cut_edge_children, cut_node_meta, cut_edge_meta, cut_graph_meta ), reverse_edge_mapping, reverse_node_mapping = cut_results

    edge_parents_buffer  = np.empty( cut_graph_meta[MAX_PARENTS], dtype=np.int32 )
    edge_children_buffer = np.empty( cut_graph_meta[MAX_CHILDREN], dtype=np.int32 )
    child_edges_buffer   = np.empty( cut_graph_meta[MAX_CHILD_EDGES], dtype=np.int32 )

    # We want a u for every node
    u_count        = np.zeros( cut_node_meta.shape[1], dtype=np.int32 )
    u_output_order = np.zeros( cut_node_meta.shape[1], dtype=np.int32 )

    # We want a v for every [parent, child edge] combo
    v_count        = np.zeros( cut_edge_parents.shape[1], dtype=np.int32 )
    v_output_order = np.zeros( ( cut_edge_parents.shape[1], 2 ), dtype=np.int32 )

    # Find the order that we need to visit the nodes
    batch_sizes = fastMessagePassing( cut_edge_parents,
                                      cut_edge_children,
                                      cut_node_meta,
                                      cut_edge_meta,
                                      cut_graph_meta,
                                      u_output_order,
                                      v_output_order,
                                      u_count,
                                      v_count,
                                      edge_parents_buffer,
                                      edge_children_buffer,
                                      child_edges_buffer )

    with nogil:
        # Retrieve the original indices
        for i in range( u_output_order.shape[0] ):
            u_output_order[i] = reverse_node_mapping[u_output_order[i]]

        for i in range( v_output_order.shape[0] ):
            v_output_order[i, EDGE] = reverse_edge_mapping[v_output_order[i, EDGE]]
            v_output_order[i, NODE] = reverse_node_mapping[v_output_order[i, NODE]]

    return np.asarray( u_output_order ), np.asarray( v_output_order ), batch_sizes

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef batchedInferenceInstructions( int[:, :]             edge_parents,
                                    int[:, :]             edge_children,
                                    int[:, :]             node_meta,
                                    int[:, :]             edge_meta,
                                    int[:]                graph_meta,
                                    int[:]                brute_force,
                                    int[:]                u_order,
                                    int[:, :]             v_order,
                                    vector[pair[int,int]] batch_sizes ):
    """
    Will compress the output order and determine which computations can
    be combined into a batch.

    Will group by computation type and number of nodes in potential

    Args:
        edge_parents   : Info about edges' parent nodes
        edge_children  : Info about edges' child nodes
        node_meta      : Meta data about nodes
        edge_meta      : Meta data about the edges
        graph_meta     : Meta data about the graph
        brute_force    : Will not integrate these out
        u_output_order : The order nodes are visited for computing U
        v_output_order : The order (edge,node) are visited in edge_parents for computing V
        batch_sizes    : The number of elements per batch.  Elements within a batch
                         can be processed in parallel.  First element is the elements
                         to take from U and second is the elements to take from V

    Returns:
        vector[vector[WorkTicket]] that what computations to do
    """
    cdef int i
    cdef int u_index = 0
    cdef int v_index = 0
    cdef vector[pair[int,int]] new_batch_sizes

    with nogil:

        for i in range( batch_sizes.size() ):

            # Compute U
            for j in range( u_index, u_index + batch_sizes.at( i ).first ):

                # U for every non-fbs parent

                # V for every non-fbs parent at every child edge thats not this node's parent edge

                # Transition to this node

                # Transition to every sibling

                # V for every sibling at every child edge

                # Integrate over every parent and sibling that isn't in the fbs

            # Compute V
            for j in range( v_index, v_index + batch_sizes.at( i ).second ):

                # U for every non-fbs mate

                # V for every non-fbs mate at every child edge thats not e

                # Transition to every child

                # V for every child at every child edge

                # Integrate over every mate and child that isn't in the fbs

            u_index += batch_sizes.at( i ).first
            v_index += batch_sizes.at( i ).second

            # Make sure that we don't integrate out cutset nodes

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef getGraphWorkTickets( graphs ):
    sparses = [ graph.toSparse() for graph in graphs ]
    edge_parents, edge_children = Graph.combineSparse( sparses )
    big_sparse = preprocessSparseGraphForTraversal( edge_parents.astype( np.int32 ), edge_children.astype( np.int32 ) )
    edge_parents, edge_children, node_meta, edge_meta, graph_meta = big_sparse
    cutset = Graph.combineCutSets( graphs )

    u_order, v_order, batch_sizes = cutsetMessagePassing( edge_parents,
                                                          edge_children,
                                                          node_meta,
                                                          edge_meta,
                                                          graph_meta,
                                                          cutset )

    work_tickets = batchedInferenceInstructions( edge_parents,
                                                 edge_children,
                                                 node_meta,
                                                 edge_meta,
                                                 graph_meta,
                                                 cutset,
                                                 u_order,
                                                 v_order,
                                                 batch_sizes )

    return work_tickets