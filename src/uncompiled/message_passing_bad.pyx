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

"""
The functions in this file allow efficient message passing over undirected
polytrees.

See definitions.pxd for an overview of the data structures
"""

cdef int INVALID_INDEX = INT_MAX

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef preprocessPolytree( int[:, :] polytree,
                          int       n_nodes=-1,
                          bool      polytree_check=False ):
    """ Pre-process the polytree tree so that we can traverse it quickly

        Args:
            polytree       : This contains the adjacency matrix for the hypergraph.
                             MUST BE SORTED BY EDGE!!!!
            n_nodes        : The number of nodes in polytree.  If not passed in, will
                             be found at runtime
            polytree_check : Whether or not to check to order of the edges in polytree

        Returns:
            linked_sparse  : Linked list that points to the next index that a node at
                             index i in polytree is located at
            node_meta      : Meta data for each node
            edge_meta      : Meta data for each edge
    """
    cdef int i, j
    cdef int polytree_length = polytree.shape[1]
    cdef int last_edge
    cdef int current_node, current_edge
    cdef int edge_start_index
    cdef int[:] linked_sparse
    cdef int[:, :] node_meta, edge_meta
    cdef int TMP_STORAGE = NODE_INDEX

    ############################################################
    # Check if the polytree is ordered by edge
    if( polytree_check == True ):
        current_edge = polytree[EDGE, 0]
        for i in range( polytree_length ):
            assert current_edge <= polytree[EDGE, i]
            current_edge = polytree[EDGE, i]

    ############################################################
    # If the user didn't pass in the number of nodes, search for it
    with nogil:
        if( n_nodes == -1 ):
            n_nodes = 0
            for i in range( polytree_length ):
                current_node = polytree[NODE, i]
                if( current_node > n_nodes ):
                    n_nodes = current_node
            n_nodes += 1

    ############################################################
    # Allocate memory for the data structures
    linked_sparse = np.full( polytree_length, INVALID_INDEX, dtype=np.int32 )
    edge_meta     = np.empty( ( 2, polytree[EDGE, polytree_length - 1] ), dtype=np.int32 )
    node_meta     = np.full( ( 2, n_nodes ), INVALID_INDEX, dtype=np.int32 )

    with nogil:

        ############################################################
        # Create the linked list that points from edge to edge
        # Go backwards through the polytree and store the next index for every index
        for i in range( polytree_length ):
            j = polytree_length - i - 1
            current_node = polytree[NODE, j]
            if( node_meta[TMP_STORAGE, current_node] != INVALID_INDEX ):
                linked_sparse[j] = node_meta[TMP_STORAGE, current_node]
            node_meta[TMP_STORAGE, current_node] = j

        ############################################################
        # Fill in node_meta
        for i in range( polytree_length ):
            current_node = polytree[NODE, i]

            # Increment the number of edges for the node and check if
            # this edge is the first edge
            node_meta[N_EDGES, current_node] += 1
            if( node_meta[NODE_INDEX, current_node] == INVALID_INDEX ):
                node_meta[NODE_INDEX, current_node] = i

        ############################################################
        # Fill in edge_meta
        edge_start_index = 0
        for i in range( polytree_length ):
            current_edge = polytree[EDGE, i]

            # Update edge_meta by storing the start of the edge
            # and the number of nodes in the previous edge
            if( current_edge != last_edge ):
                edge_meta[EDGE_INDEX, current_edge] = i
                edge_meta[N_NODES, last_edge] = i - edge_start_index
                edge_start_index = i
                last_edge = current_edge

        edge_meta[N_NODES, current_edge] = polytree_length - edge_start_index

    return np.asarray( linked_sparse ), np.asarray( node_meta ), np.asarray( edge_meta )

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void polytreeMessagePassing( int[:, :]    polytree,
                                  int[:]       linked_sparse,
                                  int[:, :]    node_meta,
                                  int[:, :]    edge_meta,
                                  int[:]       messages,
                                  vector[int]& batch_sizes ) nogil except *:
    """ Message passing over polytrees.  This will ideally be a clique tree.
        Intuitively, each node within an edge is the link to other edges.

        Args:
            polytree      : This contains the adjacency matrix for the hypergraph.
            linked_sparse : Linked list that points to the next index
                            that a node at index i in polytree
                            is located at
            node_meta     : Meta data for each node
            edge_meta     : Meta data for each edge

        Returns (Fills reference):
            messages      : The correct order that the edges are to be processed
            batch_sizes   : The number of elements per batch.  Elements within a batch
                            can be processed in parallel.
    """
    with gil:
        assert messages.shape[0] == polytree.shape[1]
        assert batch_sizes.size() == 0

    # The final messages will have exactly as many elements as the polytree.
    cdef int[:] message_counts
    cdef int last_message_index = 0
    cdef int message_index = 0

    cdef int recorded_index
    cdef int i, j
    cdef int tree_index
    cdef int progress
    cdef int n_edges
    cdef int node_index
    cdef int related_node
    cdef int edge_start
    cdef int family_index
    cdef int family_edge
    cdef int current_node, current_edge

    with gil:
        message_counts = np.zeros( messages, dtype=np.int32 )
    ############################################################
    # Initialize the semaphore.  For a node in an edge, have to wait
    # for all of the relatives in OTHER edges that node is a part of.
    # This means that for every other edge and for every relative in that
    # edge, need to increment the count by the number of edges the relative is
    # a part of that is not edge.
    for i in range( polytree.shape[1] ):
        current_edge = polytree[EDGE, i]
        current_node = polytree[NODE, i]
        node_index = node_meta[NODE_INDEX, current_node]

        family_index = linked_sparse[node_index]
        while( family_index != INVALID_INDEX ):
            family_edge = polytree[EDGE, family_index]

            # Check if this family of current_node is current_edge
            if( family_edge != current_edge ):

                # Loop over related nodes in family_edge
                edge_start = edge_meta[EDGE_INDEX, family_edge]
                for tree_index in range( edge_start, edge_start + edge_meta[N_NODES, family_edge] ):
                    related_node = polytree[NODE, tree_index]

                    # Increase the message counts for every other family that
                    # a relative is a part of
                    message_counts[i] += node_meta[N_EDGES, related_node] - 1

            # Find the next family
            family_index = linked_sparse[family_index]

    # ############################################################
    # # Initialize the semaphore.  We have to wait for every other node in every
    # # edge AND every other node in related edges to be finished
    # for i in range( polytree.shape[1] ):
    #     current_node = polytree[NODE, i]
    #     node_index = node_meta[NODE_INDEX, current_node]

    #     family_index = linked_sparse[node_index]
    #     while( family_index != INVALID_INDEX ):
    #         family_edge = polytree[EDGE, family_index]

    #         # Increment by the number of relatives
    #         message_counts[i] += edge_meta[N_NODES, family_edge]

    #         # Find the next family
    #         family_index = linked_sparse[family_index]

    #     # Subtract off the number of times current_node appeared
    #     message_counts[i] -= node_meta[N_EDGES, current_node]

    ############################################################
    # Find the base case edges (the ones that unique nodes)
    for current_node in range( node_meta.shape[1] ):
        n_edges = node_meta[N_EDGES, current_node]

        # Cliques with unique nodes are base case cliques
        if( n_edges == 1 ):
            node_index = node_meta[NODE_INDEX, current_node]

            # Record the base case nodes
            messages[message_index] = node_index
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
            current_edge = polytree[EDGE, recorded_index]
            current_node = polytree[NODE, recorded_index]
            node_index = node_meta[NODE_INDEX, current_node]

            # Decrement counts for nodes related to current_node
            # but not if they are in current_edge
            family_index = linked_sparse[node_index]
            while( family_index != INVALID_INDEX ):
                family_edge = polytree[EDGE, family_index]

                # Check if this family of current_node is current_edge
                if( family_edge != current_edge ):

                    # Loop over related nodes in family_edge
                    edge_start = edge_meta[EDGE_INDEX, family_edge]
                    for tree_index in range( edge_start, edge_start + edge_meta[N_NODES, family_edge] ):
                        related_node = polytree[NODE, tree_index]

                        # Update the structures for the related node
                        if( related_node != current_node ):
                            message_counts[tree_index] -= 1

                            # If the next node is ready, record it
                            if( message_counts[tree_index] == 0 ):
                                messages[message_index] = tree_index
                                message_index += 1

                # Find the next edge
                family_index = linked_sparse[family_index]

        ############################################################
        # Check if we're done
        if( progress == message_index ):
            break

        # Record the last back size
        batch_sizes.push_back( message_index - progress )

        # Update the last_message_index
        last_message_index = progress














