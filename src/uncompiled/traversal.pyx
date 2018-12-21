import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set
from libcpp.pair cimport pair
from compiled.definitions cimport *

"""
This file contains an optimized implementation of bredth first search
and message passing over polytrees.  It uses a preprocessed
sparse polytree for O(1) acesses everywhere and constant size buffers.

See definitions.pxd for an overview
"""

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef preprocessSparseGraphForTraversal( int[:, :] edge_parents,
                                         int[:, :] edge_children ):
    """ Pre-process sparse graph so that we can traverse it fast

    Args:
        edge_parents : Parents for each edge
        edge_parents : Children for each edge

    Returns:
        pp_edge_parents  : Pre-processed edge parents array
        pp_edge_children : Pre-processed edge children array
        node_meta        : Meta data about for node
        edge_meta        : Meta data about for edge
        graph_meta       : Meta data about the graph
    """
    cdef int[:] graph_meta          = np.zeros( 5, dtype=np.int32 )
    cdef int[:, :] pp_edge_parents  = np.empty( ( 3, edge_parents.shape[1] ), dtype=np.int32 )
    cdef int[:, :] pp_edge_children = np.empty( ( 2, edge_children.shape[1] ), dtype=np.int32 )
    cdef int edge_parents_length    = edge_parents.shape[1]
    cdef int edge_children_length   = edge_children.shape[1]
    cdef int n_edges                = edge_parents[EDGE, edge_parents_length - 1] + 1
    cdef int n_nodes
    cdef int[:, :] node_meta
    cdef int[:, :] edge_meta
    cdef int[:]    node_last_index
    cdef int i
    cdef int cum_sum
    cdef int current_edge
    cdef int current_node
    cdef int edge_start_index

    # Copy over the edge_parents and edge_children data
    cdef int[:, :] tmp = edge_parents
    with nogil:
        for i in range( edge_parents_length ):
            pp_edge_parents[NODE, i] = tmp[NODE, i]
            pp_edge_parents[EDGE, i] = tmp[EDGE, i]

    tmp = edge_children
    with nogil:
        for i in range( edge_children_length ):
            pp_edge_children[NODE, i] = tmp[NODE, i]
            pp_edge_children[EDGE, i] = tmp[EDGE, i]

        # Find the number of nodes by finding the largest node index
        n_nodes = 0
        for i in range( edge_parents_length ):
            if( pp_edge_parents[NODE, i] > n_nodes ):
                n_nodes = pp_edge_parents[NODE, i]
        for i in range( pp_edge_children.shape[ 1 ] ):
            if( pp_edge_children[NODE, i] > n_nodes ):
                n_nodes = pp_edge_children[NODE, i]
        n_nodes += 1

    # Initialize the remaining data structures
    node_meta        = np.empty( ( 3, n_nodes ), dtype=np.int32 )
    edge_meta        = np.zeros( ( 4, n_edges ), dtype=np.int32 )
    node_last_index  = np.empty( n_nodes, dtype=np.int32 )

    with nogil:
        # Fill in edge_meta.

        # First, record the index of each edge in edge_parents
        # and count the number of parents
        current_edge = 0
        cum_sum      = 0
        for i in range( edge_parents_length ):

            # Have we found the next edge?
            if( pp_edge_parents[EDGE, i] != current_edge ):

                # Count the number of parents for the previous edge
                if( current_edge == 0 ):
                    edge_meta[N_PARENTS, current_edge] = i
                else:
                    edge_meta[N_PARENTS, current_edge] = i - cum_sum
                cum_sum += edge_meta[N_PARENTS, current_edge]

                # Increment and record the index of the new edge
                current_edge += 1
                edge_meta[EDGE_PRNT_IDX, current_edge] = i

        # Calculate the number of parents for the last edge
        edge_meta[N_PARENTS, n_edges - 1] = edge_parents_length - cum_sum

        # Second, record the index of each edge in edge_children
        # and count the nubmer of children
        current_edge = 0
        cum_sum      = 0
        for i in range( edge_children_length ):

            # Have we found the next edge?
            if( pp_edge_children[EDGE, i] != current_edge ):

                # Count the number of parents for the previous edge
                if( current_edge == 0 ):
                    edge_meta[N_CHILDREN, current_edge] = i
                else:
                    edge_meta[N_CHILDREN, current_edge] = i - cum_sum
                cum_sum += edge_meta[N_CHILDREN, current_edge]

                # Increment and record the index of the new edge
                current_edge += 1
                edge_meta[EDGE_CHDN_IDX, current_edge] = i

        # Calculate the number of parents for the last edge
        edge_meta[N_CHILDREN, n_edges - 1] = edge_children_length - cum_sum

        #############################################################
        # Fill in node_meta.

        # Fill node_meta with dummy values
        node_meta[ : ] = -1

        # Find the index of the first child edge in edge_parents for each node.
        # Only do this for the earliest edge
        for i in range( edge_parents_length ):
            if( node_meta[NODE_PRNT_IDX, pp_edge_parents[NODE, i]] == -1 ):
                node_meta[NODE_PRNT_IDX, pp_edge_parents[NODE, i]] = edge_meta[EDGE_PRNT_IDX, pp_edge_parents[EDGE, i]]

        # Find the index of the parent edge in edge_children for each node
        for i in range( edge_children_length ):
            node_meta[NODE_CHDN_IDX, pp_edge_children[NODE, i]] = edge_meta[EDGE_CHDN_IDX, pp_edge_children[EDGE, i]]

        #############################################################

        # Complete the linked list in edge_parents.
        # Do this by working backwards and storing the last known index for
        # every node.
        # Also count the number of child edges each node has
        node_last_index[:] = -1
        node_meta[N_CHILD_EDGES, :] = 0
        for i in range( edge_parents_length ):

            # Iterate backwards
            j = edge_parents_length - 1 - i
            current_node = pp_edge_parents[NODE, j]

            # Increment the number of child edges
            node_meta[N_CHILD_EDGES, current_node] += 1

            # Search for the previous occurence and update
            pp_edge_parents[NEXT_FMLY, j] = node_last_index[current_node]
            node_last_index[current_node] = j

        #############################################################
        # Fill in graph meta

        for i in range( n_nodes ):
            if( node_meta[NODE_CHDN_IDX, i] == -1 ):
                graph_meta[N_ROOTS] += 1
            elif( node_meta[NODE_PRNT_IDX, i] == -1 ):
                graph_meta[N_LEAVES] += 1

            if( node_meta[N_CHILD_EDGES, i] > graph_meta[MAX_CHILD_EDGES] ):
                graph_meta[MAX_CHILD_EDGES] = node_meta[N_CHILD_EDGES, i]

        for i in range( n_edges ):
            if( edge_meta[N_PARENTS, i] > graph_meta[MAX_PARENTS] ):
                graph_meta[MAX_PARENTS] = edge_meta[N_PARENTS, i]

            if( edge_meta[N_PARENTS, i] > graph_meta[MAX_CHILDREN] ):
                graph_meta[MAX_CHILDREN] = edge_meta[N_PARENTS, i]


    return np.asarray( pp_edge_parents ), \
           np.asarray( pp_edge_children ), \
           np.asarray( node_meta ), \
           np.asarray( edge_meta ), \
           np.asarray( graph_meta )

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void getParents( const int[:, :] edge_parents,
                      const int[:, :] edge_children,
                      const int[:, :] node_meta,
                      const int[:, :] edge_meta,
                      const int[:]    graph_meta,
                      int             node,
                      int[:]          parents ) nogil except *:
    """ Finds parents of node.

        Args:
            edge_parents  : Info about edges' parent nodes
            edge_children : Info about edges' child nodes
            node_meta     : Meta data about nodes
            edge_meta     : Meta data about the edges
            graph_meta    : Meta data about the graph
            node          : Node to evaluate
            parents       : The parents of node.  Assumed to be the correct size

        Returns:
            None
    """
    cdef int i
    cdef int j = 0
    cdef int edge_children_index
    cdef int parent_edge
    cdef int n_parents
    cdef int edge_parents_index

    # Find the node's parent edge
    edge_children_index = node_meta[NODE_CHDN_IDX, node]
    parent_edge         = edge_children[EDGE, edge_children_index]
    n_parents           = edge_meta[N_PARENTS, parent_edge]

    # Find the index in edge_parents for parent edge
    edge_parents_index = edge_meta[EDGE_PRNT_IDX, parent_edge]

    # Populate the parents array
    for i in range( edge_parents_index, edge_parents_index + n_parents ):
        parents[j] = edge_parents[NODE, i]
        j += 1

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void getSiblings( const int[:, :] edge_parents,
                       const int[:, :] edge_children,
                       const int[:, :] node_meta,
                       const int[:, :] edge_meta,
                       const int[:]    graph_meta,
                       int             node,
                       int[:]          siblings ) nogil except *:
    """ Finds siblings of node.

        Args:
            edge_parents  : Info about edges' parent nodes
            edge_children : Info about edges' child nodes
            node_meta     : Meta data about nodes
            edge_meta     : Meta data about the edges
            graph_meta    : Meta data about the graph
            node          : Node to evaluate
            siblings      : The siblings of node.  Assumed to be the correct size

        Returns:
            None
    """
    cdef int i
    cdef int j = 0
    cdef int edge_children_index
    cdef int parent_edge
    cdef int n_children

    # Find the node's parent edge
    edge_children_index = node_meta[NODE_CHDN_IDX, node]
    parent_edge         = edge_children[EDGE, edge_children_index]
    n_children          = edge_meta[N_CHILDREN, parent_edge]

    # Populate the siblings array
    for i in range( edge_children_index, edge_children_index + n_children ):
        if( edge_children[NODE, i] == node ):
            continue
        siblings[j] = edge_children[NODE, i]
        j += 1

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void getMates( const int[:, :] edge_parents,
                    const int[:, :] edge_children,
                    const int[:, :] node_meta,
                    const int[:, :] edge_meta,
                    const int[:]    graph_meta,
                    int             node,
                    int             edge,
                    int[:]          mates,
                    int             known_preceding_edge_index=-1 ) nogil except *:
    """ Finds mates of node at some edge

        Args:
            edge_parents               : Info about edges' parent nodes
            edge_children              : Info about edges' child nodes
            node_meta                  : Meta data about nodes
            edge_meta                  : Meta data about the edges
            graph_meta                 : Meta data about the graph
            node                       : Node to evaluate
            edge                       : Edge to get mates at
            mates                      : The mates of node.  Assumed to be the correct size
            known_preceding_edge_index : An index that is known to precede edge in the linked list

        Returns:
            None
    """
    cdef int i
    cdef int j = 0
    cdef int edge_parents_index
    cdef int n_parents

    # Find the index in edge_parents for edge
    edge_parents_index = edge_meta[EDGE_PRNT_IDX, edge]
    n_parents          = edge_meta[N_PARENTS, edge]

    # Populate the mates array
    for i in range( edge_parents_index, edge_parents_index + n_parents ):
        if( edge_parents[NODE, i] == node ):
            continue
        mates[j] = edge_parents[NODE, i]
        j += 1

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void getChildren( const int[:, :] edge_parents,
                       const int[:, :] edge_children,
                       const int[:, :] node_meta,
                       const int[:, :] edge_meta,
                       const int[:]    graph_meta,
                       int             node,
                       int             edge,
                       int[:]          children,
                       int             known_preceding_edge_index=-1 ) nogil except *:
    """ Finds children of node at some edge

        Args:
            edge_parents               : Info about edges' parent nodes
            edge_children              : Info about edges' child nodes
            node_meta                  : Meta data about nodes
            edge_meta                  : Meta data about the edges
            graph_meta                 : Meta data about the graph
            node                       : Node to evaluate (Not actually needed)
            edge                       : Edge to get children at
            children                   : The children of node.  Assumed to be the correct size
            known_preceding_edge_index : An index that is known to precede edge in the linked list

        Returns:
            None
    """
    # Look up where edge is in edge_parents
    cdef int i
    cdef int j = 0
    cdef int edge_children_index
    cdef int n_children

    # Find the location of edge in edge_children
    edge_children_index = edge_meta[EDGE_CHDN_IDX, edge]
    n_children = edge_meta[N_CHILDREN, edge]

    # Populate the children array
    for i in range( edge_children_index, edge_children_index + n_children ):
        children[j] = edge_children[NODE, i]
        j += 1

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void getChildrenEdges( const int[:, :] edge_parents,
                            const int[:, :] edge_children,
                            const int[:, :] node_meta,
                            const int[:, :] edge_meta,
                            const int[:]    graph_meta,
                            int             node,
                            int[ : ]        children_edges ) nogil except *:
    """ Returns the child edges of node

        Args:
            edge_parents   : Info about edges' parent nodes
            edge_children  : Info about edges' child nodes
            node_meta      : Meta data about nodes
            edge_meta      : Meta data about the edges
            graph_meta     : Meta data about the graph
            node           : Node to evaluate
            children_edges : The child of node.  Assumed to be the correct size

        Returns:
            None
    """
    cdef int i
    cdef int j = 0
    cdef int edge_parents_index
    cdef int n_child_edges

    # Find the index of the first child edge
    edge_parents_index = node_meta[NODE_PRNT_IDX, node]
    n_child_edges      = node_meta[N_CHILD_EDGES, node]

    # Just return if this is a leaf
    if( n_child_edges == 0 ):
        return

    # Move onto the correct node
    while( edge_parents[NODE, edge_parents_index] != node ):
        edge_parents_index += 1

    # Traverse the linked list and retrieve the edges
    for i in range( n_child_edges ):
        children_edges[ j ] = edge_parents[EDGE, edge_parents_index]
        j += 1

        edge_parents_index = edge_parents[NEXT_FMLY, edge_parents_index]

@cython.boundscheck( False )
@cython.wraparound( False )
cdef int getEdgeParentIndex( const int[:, :] edge_parents,
                             const int[:, :] edge_children,
                             const int[:, :] node_meta,
                             const int[:, :] edge_meta,
                             const int[:]    graph_meta,
                             int             node,
                             int             edge ) nogil except *:
    """ Returns the child edges of node

        Args:
            edge_parents   : Info about edges' parent nodes
            edge_children  : Info about edges' child nodes
            node_meta      : Meta data about nodes
            edge_meta      : Meta data about the edges
            graph_meta     : Meta data about the graph
            node           : Node to evaluate
            edge           : A child edge of node

        Returns:
            Index in edge_parents corresponding to (edge,node)
    """
    cdef int index

    index = edge_meta[EDGE_PRNT_IDX, edge]
    while( edge_parents[NODE, index] != node ):
        index += 1
    return index

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void updateNextEdgesForForwardPass( const int[:, :] edge_parents,
                                         const int[:, :] edge_children,
                                         const int[:, :] node_meta,
                                         const int[:, :] edge_meta,
                                         const int[:]    graph_meta,
                                         int[:]          edge_parents_left,
                                         vector[int]&    next_edges,
                                         int[:]          child_edges_buffer,
                                         int[:]          last_traversed_nodes ) nogil except *:
    """ Update the

        Args:
            edge_parents         : Info about edges' parent nodes
            edge_children        : Info about edges' child nodes
            node_meta            : Meta data about nodes
            edge_meta            : Meta data about the edges
            graph_meta           : Meta data about the graph
            edge_parents_left    : How many parents are have not been reached yet.
                                   >0 means blocked, 0 means ready, -1 means traversed.
            next_edges           : Contains the edges that are ready.  Will be modified to
                                   have the next ready edges at the end of this function.
            child_edges_buffer   : Buffer to put the child edges
            last_traversed_nodes : The nodes that were just traversed

        Returns:
            None
    """
    cdef int i
    cdef int j
    cdef int edge
    cdef int node
    cdef int n_child_edges

    next_edges.clear()

    # Find the first set of ready edges
    for i in range( last_traversed_nodes.shape[0] ):
        node = last_traversed_nodes[i]

        # Iterate over each child edge
        n_child_edges = node_meta[N_CHILD_EDGES, node]

        # Check if this child is a leaf
        if( n_child_edges != 0 ):

            # Retrive the child edges
            getChildrenEdges( edge_parents,
                              edge_children,
                              node_meta,
                              edge_meta,
                              graph_meta,
                              node,
                              child_edges_buffer )

            # Loop over the child edges and find the ones that are ready
            for j in range( n_child_edges ):
                edge = child_edges_buffer[j]
                edge_parents_left[edge] -= 1
                if( edge_parents_left[edge] == 0 ):
                    next_edges.push_back( edge )


@cython.boundscheck( False )
@cython.wraparound( False )
cdef void forwardPassStep( const int[:, :] edge_parents,
                           const int[:, :] edge_children,
                           const int[:, :] node_meta,
                           const int[:, :] edge_meta,
                           const int[:]    graph_meta,
                           int[:]          edge_parents_left,
                           vector[int]&    next_edges,
                           int[:]          output_order,
                           int&            current_output_index,
                           int[:]          child_edges_buffer ) nogil except *:
    """ Perform a step of bredth first search.

        Args:
            edge_parents         : Info about edges' parent nodes
            edge_children        : Info about edges' child nodes
            node_meta            : Meta data about nodes
            edge_meta            : Meta data about the edges
            graph_meta           : Meta data about the graph
            edge_parents_left    : How many parents are have not been reached yet.
                                   >0 means blocked, 0 means ready, -1 means traversed.
            next_edges           : Contains the edges that are ready.  Will be modified to
                                   have the next ready edges at the end of this function.
            output_order         : The solution array
            current_output_index : The index of the most recent node in output_order
            child_edges_buffer   : Buffer to put the child edges

        Returns:
            None
    """
    cdef int i
    cdef int j
    cdef int child
    cdef int edge
    cdef int n_children
    cdef int n_child_edges
    cdef int original_index = current_output_index

    # Retrieve the children of the edges that are ready
    for i in range( next_edges.size() ):
        edge = next_edges[i]
        n_children = edge_meta[N_CHILDREN, edge]

        # Retrieve the children
        getChildren( edge_parents,
                     edge_children,
                     node_meta,
                     edge_meta,
                     graph_meta,
                     -1,
                     edge,
                     output_order[current_output_index : current_output_index + n_children] )
        (&current_output_index)[0] += n_children

        # Mark this edge as traversed
        edge_parents_left[edge] -= 1

    # Find the next edges to go down
    updateNextEdgesForForwardPass( edge_parents,
                                   edge_children,
                                   node_meta,
                                   edge_meta,
                                   graph_meta,
                                   edge_parents_left,
                                   next_edges,
                                   child_edges_buffer,
                                   output_order[original_index : current_output_index ] )

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef forwardPass( const int[:, :] edge_parents,
                   const int[:, :] edge_children,
                   const int[:, :] node_meta,
                   const int[:, :] edge_meta,
                   const int[:]    graph_meta ):
    """ Bredth first search on the graph.
        TODO:  Clean up this code, it is implemented poorly

        Args:
            edge_parents  : Info about edges' parent nodes
            edge_children : Info about edges' child nodes
            node_meta     : Meta data about nodes
            edge_meta     : Meta data about the edges
            graph_meta    : Meta data about the graph

        Returns:
            output_order : The order that nodes are visited during a bredth first search
            batch_sizes  : The number of elements per batch.  Elements within a batch
                           can be processed in parallel
    """
    cdef int[:] edge_parents_left
    cdef int[:] output_order
    cdef int[:] child_edges_buffer
    cdef int i
    cdef int j
    cdef int node
    cdef int edge
    cdef int current_output_index = 0
    cdef int n_child_edges
    cdef vector[int] next_edges
    cdef vector[int] batch_sizes

    # Allocate a buffer for the child edges so that we don't have to
    # do it multiple times in the future
    child_edges_buffer = np.empty( graph_meta[MAX_CHILD_EDGES], dtype=np.int32 )

    # Initialize the output data structure and edge counter
    output_order      = np.empty( node_meta.shape[1], dtype=np.int32 )
    edge_parents_left = np.empty( edge_meta.shape[1], dtype=np.int32 )

    with nogil:

        # Populate the edge_parents_left array
        for edge in range( edge_meta.shape[1] ):
            edge_parents_left[edge] = edge_meta[N_PARENTS, edge]

        # Find the roots
        for node in range( node_meta.shape[1] ):
            if( node_meta[NODE_CHDN_IDX, node] == -1 ):
                output_order[current_output_index] = node
                current_output_index += 1

        # The first batch is as large as the number of roots
        batch_sizes.push_back( current_output_index )

        # Find the next edges to go down
        updateNextEdgesForForwardPass( edge_parents,
                                       edge_children,
                                       node_meta,
                                       edge_meta,
                                       graph_meta,
                                       edge_parents_left,
                                       next_edges,
                                       child_edges_buffer,
                                       output_order[0 : current_output_index ] )

        # Run over the remaining nodes
        while( next_edges.size() > 0 ):
            forwardPassStep( edge_parents,
                             edge_children,
                             node_meta,
                             edge_meta,
                             graph_meta,
                             edge_parents_left,
                             next_edges,
                             output_order,
                             current_output_index,
                             child_edges_buffer )
            batch_sizes.push_back( current_output_index - batch_sizes.back() )

    return np.asarray( output_order ), batch_sizes

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void initializeMessagePassingCounts( const int[:, :]           edge_parents,
                                          const int[:, :]           edge_children,
                                          const int[:, :]           node_meta,
                                          const int[:, :]           edge_meta,
                                          const int[:]              graph_meta,
                                          int[:]                    edge_parents_buffer,
                                          int[:]                    edge_children_buffer,
                                          int[:]                    u_count,
                                          int[:]                    v_count ) nogil except *:
    """ Initialize the counts for the message passing algorithm

        Args:
            edge_parents         : Info about edges' parent nodes
            edge_children        : Info about edges' child nodes
            node_meta            : Meta data about nodes
            edge_meta            : Meta data about the edges
            graph_meta           : Meta data about the graph
            edge_parents_buffer  : Buffer to store the result of a parents query
            edge_children_buffer : Buffer to store the result of a children query
            u_count              : When to proceed on U calculation for node
            v_count              : When to proceed on V calculation for (node,edge)

        Returns:
            None
    """
    cdef int i
    cdef int j
    cdef int edge
    cdef int node
    cdef int parent
    cdef int sibling
    cdef int mate
    cdef int child
    cdef int edge_children_index
    cdef int parent_edge
    cdef int n_parents
    cdef int n_siblings
    cdef int n_mates
    cdef int n_children

    # These are the places where we need a U value for:
    #  - U for all parents
    #  - V for all parents over all child edges except node's parent edge
    #  - V for all siblings over all child edges
    for node in range( u_count.shape[ 0 ] ):

        # Get information about the parents and siblings
        edge_children_index = node_meta[NODE_CHDN_IDX, node]

        # Check if the node is a root
        if( edge_children_index != -1 ):
            parent_edge         = edge_children[EDGE, edge_children_index]
            n_parents           = edge_meta[N_PARENTS, parent_edge]
            n_siblings          = edge_meta[N_CHILDREN, parent_edge] - 1
            getParents( edge_parents,
                        edge_children,
                        node_meta,
                        edge_meta,
                        graph_meta,
                        node,
                        edge_parents_buffer )
            getSiblings( edge_parents,
                         edge_children,
                         node_meta,
                         edge_meta,
                         graph_meta,
                         node,
                         edge_children_buffer )

            # Increment by the number of parents and also increment by the number
            # of total child edges over the parents that does not include node's parent edge.
            # This is equivalent to adding the total number of down edges across
            # every parent
            for i in range( n_parents ):
                parent = edge_parents_buffer[i]
                u_count[node] += node_meta[N_CHILD_EDGES, parent]

            # Increment by the number of down edges over every sibling or 1 if
            # the child is a leaf
            for i in range( n_siblings ):
                sibling = edge_children_buffer[i]
                if( node_meta[N_CHILD_EDGES, sibling] == 0 ):
                    u_count[node] += 1
                else:
                    u_count[node] += node_meta[N_CHILD_EDGES, sibling]

    # These are the places where we need a V value for:
    #  - U for all mates from edge
    #  - V for all mates over all child edges for mate except for edge
    #  - V for all children from e over all child edges for child
    for i in range( v_count.shape[ 0 ] ):
        edge = edge_parents[EDGE, i]
        node = edge_parents[NODE, i]

        # Get information about the mates and children
        n_mates    = edge_meta[N_PARENTS, edge] - 1
        n_children = edge_meta[N_CHILDREN, edge]
        getMates( edge_parents,
                  edge_children,
                  node_meta,
                  edge_meta,
                  graph_meta,
                  node,
                  edge,
                  edge_parents_buffer )
        getChildren( edge_parents,
                     edge_children,
                     node_meta,
                     edge_meta,
                     graph_meta,
                     node,
                     edge,
                     edge_children_buffer )

        # Increment by the number of mates and also increment by the number
        # of total child edges over the mates that does not include node's parent edge.
        # This is equivalent to adding the total number of down edges across
        # every mate
        for j in range( n_mates ):
            mate = edge_parents_buffer[j]
            v_count[i] += node_meta[N_CHILD_EDGES, mate]

        # Increment by the number of down edges over every child, or 1 if
        # the child is a leaf
        for j in range( n_children ):
            child = edge_children_buffer[j]
            if( node_meta[N_CHILD_EDGES, child] == 0 ):
                v_count[i] += 1
            else:
                v_count[i] += node_meta[N_CHILD_EDGES, child]

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void baseCaseMessagePassing( const int[:, :]           edge_parents,
                                  const int[:, :]           edge_children,
                                  const int[:, :]           node_meta,
                                  const int[:, :]           edge_meta,
                                  const int[:]              graph_meta,
                                  int[:]                    u_output_order,
                                  int[:, :]                 v_output_order,
                                  int&                      u_index,
                                  int&                      v_index,
                                  int&                      last_u_index,
                                  int&                      last_v_index,
                                  int[:]                    u_count,
                                  int[:]                    v_count,
                                  int[:]                    edge_parents_buffer,
                                  int[:]                    edge_children_buffer ) nogil except *:
    """ Base case for message passing.  Retrieve roots only.
        Don't need to do anything about the leaves because they
        have a constant value

        Args:
            edge_parents         : Info about edges' parent nodes
            edge_children        : Info about edges' child nodes
            node_meta            : Meta data about nodes
            edge_meta            : Meta data about the edges
            graph_meta           : Meta data about the graph
            u_output_order       : The order nodes are visited for computing U
            v_output_order       : The order (edge,node) are visited in edge_parents for computing V
            u_index              : The last valid index in u_output_order
            v_index              : The last valid index in v_output_order
            last_u_index         : u_index before the previous iteration
            last_v_index         : v_index before the previous iteration
            u_count              : When to proceed on U calculation for node
            v_count              : When to proceed on V calculation for (node,edge)
            edge_parents_buffer  : Buffer to store the result of a parents query
            edge_children_buffer : Buffer to store the result of a children query

        Returns:
            None
    """
    cdef int node

    # Find the roots and add them to the output
    for node in range( node_meta.shape[1] ):
        if( node_meta[NODE_CHDN_IDX, node] == -1 ):
            u_output_order[u_index] = node
            (&u_index)[0] += 1

    # Find the leaves and add the next iteration of nodes if possible
    for node in range( node_meta.shape[1] ):

        if( node_meta[NODE_PRNT_IDX, node] == -1 ):
            parentEdgeUpdate( edge_parents,
                              edge_children,
                              node_meta,
                              edge_meta,
                              graph_meta,
                              u_output_order,
                              v_output_order,
                              u_index,
                              v_index,
                              u_count,
                              v_count,
                              edge_parents_buffer,
                              edge_children_buffer,
                              node )

    # Remember the last u and v index
    (&last_u_index)[0] = 0
    (&last_v_index)[0] = 0

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void parentEdgeUpdate( const int[:, :]           edge_parents,
                            const int[:, :]           edge_children,
                            const int[:, :]           node_meta,
                            const int[:, :]           edge_meta,
                            const int[:]              graph_meta,
                            int[:]                    u_output_order,
                            int[:, :]                 v_output_order,
                            int&                      u_index,
                            int&                      v_index,
                            int[:]                    u_count,
                            int[:]                    v_count,
                            int[:]                    edge_parents_buffer,
                            int[:]                    edge_children_buffer,
                            int                       node ) nogil except *:
    """ Decrement u_count and v_count after a V computation at node.
        Also update the final order if possible.
        This function decrements v_count at (parent_edge, parent) for each parent
        and decrements u_count at every sibling

        Args:
            edge_parents         : Info about edges' parent nodes
            edge_children        : Info about edges' child nodes
            node_meta            : Meta data about nodes
            edge_meta            : Meta data about the edges
            graph_meta           : Meta data about the graph
            u_output_order       : The order nodes are visited for computing U
            v_output_order       : The order (edge,node) are visited in edge_parents for computing V
            u_index              : Last valid index in u_output_order
            v_index              : Last valid index in v_output_order
            u_count              : When to proceed on U calculation for node
            v_count              : When to proceed on V calculation for (node,edge)
            edge_parents_buffer  : Buffer to store the result of a parents query
            edge_children_buffer : Buffer to store the result of a children query
            node                 : Node that we just did a computation at

        Returns:
            None
    """
    cdef int i
    cdef int edge_children_index
    cdef int parent_edge
    cdef int n_parents
    cdef int n_siblings
    cdef int parent
    cdef int sibling
    cdef int index_in_v

    # Get the parent edge
    edge_children_index = node_meta[NODE_CHDN_IDX, node]

    # Check if this node is a root
    if( edge_children_index == -1 ):
        return

    # Retrieve the parents and siblings
    parent_edge = edge_children[EDGE, edge_children_index]
    n_parents   = edge_meta[N_PARENTS, parent_edge]
    n_siblings  = edge_meta[N_CHILDREN, parent_edge] - 1
    getParents( edge_parents,
                edge_children,
                node_meta,
                edge_meta,
                graph_meta,
                node,
                edge_parents_buffer )
    getSiblings( edge_parents,
                 edge_children,
                 node_meta,
                 edge_meta,
                 graph_meta,
                 node,
                 edge_children_buffer )

    # Decrement from v_count for each parent at parent_edge and find the
    # ready elements
    for i in range( n_parents ):
        parent = edge_parents_buffer[i]
        index_in_v = getEdgeParentIndex( edge_parents,
                                         edge_children,
                                         node_meta,
                                         edge_meta,
                                         graph_meta,
                                         parent,
                                         parent_edge )
        v_count[index_in_v] -= 1
        if( v_count[index_in_v] == 0 ):
            v_output_order[v_index,NODE] = edge_parents[NODE,index_in_v]
            v_output_order[v_index,EDGE] = edge_parents[EDGE,index_in_v]
            (&v_index)[0] += 1

    # Decrement from u_count for each sibling and find the
    # ready elements
    for i in range( n_siblings ):
        sibling = edge_children_buffer[i]
        u_count[sibling] -= 1
        if( u_count[sibling] == 0 ):
            u_output_order[u_index] = sibling
            (&u_index)[0] += 1

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void childEdgeUpdate( const int[:, :]           edge_parents,
                           const int[:, :]           edge_children,
                           const int[:, :]           node_meta,
                           const int[:, :]           edge_meta,
                           const int[:]              graph_meta,
                           int[:]                    u_output_order,
                           int[:, :]                 v_output_order,
                           int&                      u_index,
                           int&                      v_index,
                           int[:]                    u_count,
                           int[:]                    v_count,
                           int[:]                    edge_parents_buffer,
                           int[:]                    edge_children_buffer,
                           int                       edge,
                           int                       node ) nogil except *:
    """ Decrement u_count and v_count after a U or V computation at node.
        Also update the final order if possible.
        This function decrements v_count at (edge, mate) for each mate
        and decrements u_count at every child of edge

        Args:
            edge_parents         : Info about edges' parent nodes
            edge_children        : Info about edges' child nodes
            node_meta            : Meta data about nodes
            edge_meta            : Meta data about the edges
            graph_meta           : Meta data about the graph
            u_output_order       : The order nodes are visited for computing U
            v_output_order       : The order (edge,node) are visited in edge_parents for computing V
            u_index              : Last valid index in u_output_order
            v_index              : Last valid index in v_output_order
            u_count              : When to proceed on U calculation for node
            v_count              : When to proceed on V calculation for (node,edge)
            edge_parents_buffer  : Buffer to store the result of a parents query
            edge_children_buffer : Buffer to store the result of a children query
            edge                 : The child edge of node
            node                 : Node that we just did a computation at

        Returns:
            None
    """
    cdef int i
    cdef int n_mates
    cdef int n_children
    cdef int mate
    cdef int child
    cdef int index_in_v

    # Retrieve the mates and children for this child edge
    n_mates    = edge_meta[N_PARENTS, edge] - 1
    n_children = edge_meta[N_CHILDREN, edge]
    getMates( edge_parents,
                 edge_children,
                 node_meta,
                 edge_meta,
                 graph_meta,
                 node,
                 edge,
                 edge_parents_buffer )
    getChildren( edge_parents,
                 edge_children,
                 node_meta,
                 edge_meta,
                 graph_meta,
                 node,
                 edge,
                 edge_children_buffer )

    # Decrement from v_count for each mate at edge and find the
    # ready elements
    for i in range( n_mates ):
        mate       = edge_parents_buffer[i]
        index_in_v = getEdgeParentIndex( edge_parents,
                                         edge_children,
                                         node_meta,
                                         edge_meta,
                                         graph_meta,
                                         mate,
                                         edge )
        v_count[index_in_v] -= 1
        if( v_count[index_in_v] == 0 ):
            v_output_order[v_index,NODE] = edge_parents[NODE,index_in_v]
            v_output_order[v_index,EDGE] = edge_parents[EDGE,index_in_v]
            (&v_index)[0] += 1

    # Decrement from u_count for each child and find the
    # ready elements
    for i in range( n_children ):
        child = edge_children_buffer[i]
        u_count[child] -= 1
        if( u_count[child] == 0 ):
            u_output_order[u_index] = child
            (&u_index)[0] += 1

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void messagePassingStep( const int[:, :]           edge_parents,
                              const int[:, :]           edge_children,
                              const int[:, :]           node_meta,
                              const int[:, :]           edge_meta,
                              const int[:]              graph_meta,
                              int[:]                    u_output_order,
                              int[:, :]                 v_output_order,
                              int&                      u_index,
                              int&                      v_index,
                              int&                      last_u_index,
                              int&                      last_v_index,
                              int[:]                    u_count,
                              int[:]                    v_count,
                              vector[pair[int,int]]&    batch_sizes,
                              int[:]                    edge_parents_buffer,
                              int[:]                    edge_children_buffer,
                              int[:]                    child_edges_buffer ) nogil except *:
    """ Perform 1 iteration of message passing

        Args:
            edge_parents         : Info about edges' parent nodes
            edge_children        : Info about edges' child nodes
            node_meta            : Meta data about nodes
            edge_meta            : Meta data about the edges
            graph_meta           : Meta data about the graph
            u_output_order       : The order nodes are visited for computing U
            v_output_order       : The order (edge,node) are visited in edge_parents for computing V
            u_index              : Last valid index in u_output_order
            v_index              : Last valid index in v_output_order
            last_u_index         : u_index before the previous iteration
            last_v_index         : v_index before the previous iteration
            u_count              : When to proceed on U calculation for node
            v_count              : When to proceed on V calculation for (node,edge)
            batch_sizes          : The number of elements per batch.  Elements within a batch
                                   can be processed in parallel.  First element is the elements
                                   to take from U and second is the elements to take from V
            edge_parents_buffer  : Buffer to store the result of a parents query
            edge_children_buffer : Buffer to store the result of a children query
            child_edges_buffer   : Buffer to store the result of a children edge query

        Returns:
            None
    """
    cdef int i
    cdef int j
    cdef int edge
    cdef int comp_edge
    cdef int node
    cdef int parent
    cdef int sibling
    cdef int mate
    cdef int child
    cdef int n_child_edges
    cdef int n_parents
    cdef int n_siblings
    cdef int n_mates
    cdef int n_children
    cdef int edge_children_index
    cdef int parent_edge
    cdef int index_in_v

    cdef int initial_u_index =  u_index
    cdef int initial_v_index =  v_index

    # For every completed U computation:
    #  - Decrement u_count for all children
    #  - Decrement v_count for all mates over all child edges that node and
    #    the mate are a part of
    for i in range( last_u_index, initial_u_index ):
        node          = u_output_order[i]
        n_child_edges = node_meta[N_CHILD_EDGES, node]

        # Check if this child is a leaf
        if( n_child_edges != 0 ):

            # Retrieve the child edges
            getChildrenEdges( edge_parents,
                              edge_children,
                              node_meta,
                              edge_meta,
                              graph_meta,
                              node,
                              child_edges_buffer )

            # Loop over the child edges
            for j in range( n_child_edges ):
                edge = child_edges_buffer[j]

                # Decrement
                childEdgeUpdate( edge_parents,
                                 edge_children,
                                 node_meta,
                                 edge_meta,
                                 graph_meta,
                                 u_output_order,
                                 v_output_order,
                                 u_index,
                                 v_index,
                                 u_count,
                                 v_count,
                                 edge_parents_buffer,
                                 edge_children_buffer,
                                 edge,
                                 node )

    # For every completed V computation:
    #  - Decrement u_count for children that come from a different edge
    #    than the one just completed
    #  - Decrement v_count for all parents at node's up edge
    #  - Decrement u_count for all siblings
    #  - Decrement v_count for mates that come from a different edge
    #    than the one just completed
    for i in range( last_v_index, initial_v_index ):
        node = v_output_order[i, NODE]
        edge = v_output_order[i, EDGE]
        n_child_edges = node_meta[N_CHILD_EDGES, node]

        # Retrieve the child edges
        getChildrenEdges( edge_parents,
                          edge_children,
                          node_meta,
                          edge_meta,
                          graph_meta,
                          node,
                          child_edges_buffer )

        # Loop over the all of the child edges
        for j in range( n_child_edges ):
            comp_edge = child_edges_buffer[j]

            # Skip edge
            if( comp_edge == edge ):
                continue

            childEdgeUpdate( edge_parents,
                             edge_children,
                             node_meta,
                             edge_meta,
                             graph_meta,
                             u_output_order,
                             v_output_order,
                             u_index,
                             v_index,
                             u_count,
                             v_count,
                             edge_parents_buffer,
                             edge_children_buffer,
                             comp_edge,
                             node )

        parentEdgeUpdate( edge_parents,
                          edge_children,
                          node_meta,
                          edge_meta,
                          graph_meta,
                          u_output_order,
                          v_output_order,
                          u_index,
                          v_index,
                          u_count,
                          v_count,
                          edge_parents_buffer,
                          edge_children_buffer,
                          node )

    # Note the batch size
    batch_sizes.push_back( pair[int,int]( initial_u_index - last_u_index, initial_v_index - last_v_index ) )

    # Remember the last u and v index
    (&last_u_index)[0] = initial_u_index
    (&last_v_index)[0] = initial_v_index

# Can't include these in the pxd file for some reason
@cython.boundscheck( False )
@cython.wraparound( False )
cdef vector[pair[int, int]] fastMessagePassing( const int[:, :]           edge_parents,
                                                const int[:, :]           edge_children,
                                                const int[:, :]           node_meta,
                                                const int[:, :]           edge_meta,
                                                const int[:]              graph_meta,
                                                int[:]                    u_output_order,
                                                int[:, :]                 v_output_order,
                                                int[:]                    u_count,
                                                int[:]                    v_count,
                                                int[:]                    edge_parents_buffer,
                                                int[:]                    edge_children_buffer,
                                                int[:]                    child_edges_buffer ) nogil except *:
    """ C++ guts of polytreeMessagePassing.  Returns the batch sizes

        Args:
            edge_parents         : Info about edges' parent nodes
            edge_children        : Info about edges' child nodes
            node_meta            : Meta data about nodes
            edge_meta            : Meta data about the edges
            graph_meta           : Meta data about the graph
            u_output_order       : The order nodes are visited for computing U
            v_output_order       : The order (edge,node) are visited in edge_parents for computing V
            u_count              : When to proceed on U calculation for node
            v_count              : When to proceed on V calculation for (node,edge)
            edge_parents_buffer  : Buffer to store the result of a parents query
            edge_children_buffer : Buffer to store the result of a children query
            child_edges_buffer   : Buffer to store the result of a children edge query

        Returns:
            batch_sizes          : The number of elements per batch.  Elements within a batch
                                   can be processed in parallel.  First element is the elements
                                   to take from U and second is the elements to take from V
    """
    cdef int progress
    cdef int u_index = 0
    cdef int v_index = 0
    cdef int last_u_index = 0
    cdef int last_v_index = 0
    cdef vector[pair[int, int]] batch_sizes

    # Initialize the blocking arrays
    initializeMessagePassingCounts( edge_parents,
                                    edge_children,
                                    node_meta,
                                    edge_meta,
                                    graph_meta,
                                    edge_parents_buffer,
                                    edge_children_buffer,
                                    u_count,
                                    v_count )

    # Initialize the algorithm with the roots and leaves
    baseCaseMessagePassing( edge_parents,
                            edge_children,
                            node_meta,
                            edge_meta,
                            graph_meta,
                            u_output_order,
                            v_output_order,
                            u_index,
                            v_index,
                            last_u_index,
                            last_v_index,
                            u_count,
                            v_count,
                            edge_parents_buffer,
                            edge_children_buffer )

    while( True ):

        progress = u_index + v_index

        # Run single step
        messagePassingStep( edge_parents,
                            edge_children,
                            node_meta,
                            edge_meta,
                            graph_meta,
                            u_output_order,
                            v_output_order,
                            u_index,
                            v_index,
                            last_u_index,
                            last_v_index,
                            u_count,
                            v_count,
                            batch_sizes,
                            edge_parents_buffer,
                            edge_children_buffer,
                            child_edges_buffer )

        # Break when we've visited all of the nodes
        if( progress == u_index + v_index ):
            break

    # Check to see if the algroithm failed
    if( u_index != u_output_order.shape[0] or
        v_index != v_output_order.shape[0] ):
        # Make this mean that we failed
        batch_sizes.clear()

    return batch_sizes

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef polytreeMessagePassing( int[:, :] edge_parents,
                              int[:, :] edge_children,
                              int[:, :] node_meta,
                              int[:, :] edge_meta,
                              int[:]    graph_meta ):
    """ Message passing algorithm for polytree

        Args:
            edge_parents  : Info about edges' parent nodes
            edge_children : Info about edges' child nodes
            node_meta     : Meta data about nodes
            edge_meta     : Meta data about the edges
            graph_meta    : Meta data about the graph

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

    cdef int[:] edge_parents_buffer  = np.empty( graph_meta[MAX_PARENTS], dtype=np.int32 )
    cdef int[:] edge_children_buffer = np.empty( graph_meta[MAX_CHILDREN], dtype=np.int32 )
    cdef int[:] child_edges_buffer   = np.empty( graph_meta[MAX_CHILD_EDGES], dtype=np.int32 )

    # We want a u for every node
    u_count        = np.zeros( node_meta.shape[1], dtype=np.int32 )
    u_output_order = np.zeros( node_meta.shape[1], dtype=np.int32 )

    # We want a v for every [parent, child edge] combo
    v_count        = np.zeros( edge_parents.shape[1], dtype=np.int32 )
    v_output_order = np.zeros( ( edge_parents.shape[1], 2 ), dtype=np.int32 )

    batch_sizes = fastMessagePassing( edge_parents,
                                      edge_children,
                                      node_meta,
                                      edge_meta,
                                      graph_meta,
                                      u_output_order,
                                      v_output_order,
                                      u_count,
                                      v_count,
                                      edge_parents_buffer,
                                      edge_children_buffer,
                                      child_edges_buffer )

    return np.asarray( u_output_order ), np.asarray( v_output_order ), batch_sizes