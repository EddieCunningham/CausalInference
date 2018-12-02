import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libcpp.vector cimport vector

##########################################################################################

"""
edge_parents: Contains the parents for each edge and a linked list between nodes
index       [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]

EDGE        [ 0, 1, 1, 2, 3, 3, 4, 4, 4, 4 ]  Edge index (child edge of node)
NODE        [ 5, 2, 4, 4, 2, 1, 3, 5, 2, 0 ]  Node index
NEXT_FAMILY [ 7, 4, 3,-1, 8,-1,-1,-1,-1,-1 ]  The next index in this array where node appears.  -1 if last edge for node

#############################

edge_children: Contains the children for each edge.  Has n_nodes elements
index        [ 0, 1, 2, 3, 4, 5, 6, 7, 8 ]

EDGE         [ 0, 1, 1, 2, 3, 3, 4, 4, 4 ]  Edge index (parent edge of node)
NODE         [ 5, 2, 4, 4, 2, 1, 3, 5, 2 ]  Node index

#############################

node_meta: Location of node in edge_children and edge_parents
index (node)        [ 0, 1, 2, 3, 4, 5 ]

NODE_CHILDREN_INDEX [ ................ ]  Index of parent edge in edge_children.  -1 if root
NODE_PARENTS_INDEX  [ ................ ]  Index of first child edge in edge_parents.  -1 if leaf
N_CHILD_EDGES       [ ................ ]  The number of child edges

#############################

edge_meta: Location of edge in edge_parents
index (edge)        [ 0, 1, 2, 3, 4 ]

EDGE_PARENT_INDEX   [ ............. ]  Index of edge in edge_parents
EDGE_CHILDREN_INDEX [ ............. ]  Index of edge in edge_children
N_PARENTS           [ ............. ]  Number of parents
N_CHILDREN          [ ............. ]  Number of children

#############################

graph_meta: Meta data about the number of roots and leaves
[ N_ROOTS, N_LEAVES, MAX_CHILD_EDGES ]
MAX_CHILD_EDGES : Record this so that we don't have to allocate memory every time we want to
                  fill a child edge array.  This shouldn't be too large

"""

# Enums are apparently python objects....
cdef int EDGE       = 0 # EDGE
cdef int NODE       = 1 # NODE
cdef int NEXT_FMLY  = 2 # NEXT_FAMILY

cdef int NODE_CHDN_IDX = 0 # NODE_CHILDREN_INDEX
cdef int NODE_PRNT_IDX = 1 # NODE_PARENTS_INDEX
cdef int N_CHILD_EDGES = 2 # N_CHILD_EDGES

cdef int EDGE_PRNT_IDX = 0 # EDGE_PARENT_INDEX
cdef int EDGE_CHDN_IDX = 1 # EDGE_CHILDREN_INDEX
cdef int N_PARENTS     = 2 # N_PARENTS
cdef int N_CHILDREN    = 3 # N_CHILDREN

cdef int N_ROOTS         = 0 # N_ROOTS
cdef int N_LEAVES        = 1 # N_LEAVES
cdef int MAX_CHILD_EDGES = 2 # MAX_CHILD_EDGES

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cpdef preprocessSparseGraphForTraversal( np.ndarray[int, ndim=2] edge_parents,
                                         np.ndarray[int, ndim=2] edge_children ):
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
    cdef int[:] graph_meta          = np.zeros( 3, dtype=np.int32 )
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

    return np.asarray( pp_edge_parents ), \
           np.asarray( pp_edge_children ), \
           np.asarray( node_meta ), \
           np.asarray( edge_meta ), \
           np.asarray( graph_meta )

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void getParents( int[:, :] edge_parents,
                      int[:, :] edge_children,
                      int[:, :] node_meta,
                      int[:, :] edge_meta,
                      int[:]    graph_meta,
                      int       node,
                      int[:]    parents ) nogil except *:
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
cdef void getSiblings( int[:, :] edge_parents,
                       int[:, :] edge_children,
                       int[:, :] node_meta,
                       int[:, :] edge_meta,
                       int[:]    graph_meta,
                       int       node,
                       int[:]    siblings ) nogil except *:
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
cdef void getMates( int[:, :] edge_parents,
                    int[:, :] edge_children,
                    int[:, :] node_meta,
                    int[:, :] edge_meta,
                    int[:]    graph_meta,
                    int       node,
                    int       edge,
                    int[:]    mates,
                    int       known_preceding_edge_index=-1 ) nogil except *:
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
cdef void getChildren( int[:, :] edge_parents,
                       int[:, :] edge_children,
                       int[:, :] node_meta,
                       int[:, :] edge_meta,
                       int[:]    graph_meta,
                       int       node,
                       int       edge,
                       int[:]    children,
                       int       known_preceding_edge_index=-1 ) nogil except *:
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
cdef void getChildrenEdges( int[:, :] edge_parents,
                            int[:, :] edge_children,
                            int[:, :] node_meta,
                            int[:, :] edge_meta,
                            int[:]    graph_meta,
                            int       node,
                            int[ : ]  children_edges ) nogil except *:
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

    # Move onto the correct node
    while( edge_parents[NODE, edge_parents_index] != node ):
        edge_parents_index += 1

    # Traverse the linked list and retrieve the edges
    for i in range( n_child_edges ):
        children_edges[ j ] = edge_parents[EDGE, edge_parents_index]
        j += 1

        edge_parents_index = edge_parents[NEXT_FMLY, edge_parents_index]

###################################################################################

@cython.boundscheck( False )
@cython.wraparound( False )
cdef void updateNextEdgesForForwardPass( int[:, :]    edge_parents,
                                         int[:, :]    edge_children,
                                         int[:, :]    node_meta,
                                         int[:, :]    edge_meta,
                                         int[:]       graph_meta,
                                         int[:]       edge_parents_left,
                                         vector[int]& next_edges,
                                         int[:]       child_edges_buffer,
                                         int[:]       last_traversed_nodes ) nogil except *:
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
            output_order         : The solution array
            current_output_index : The index of the most recent node in output_order
            child_edges_buffer   : Buffer to put the child edges

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
cdef void forwardPassStep( int[:, :]    edge_parents,
                           int[:, :]    edge_children,
                           int[:, :]    node_meta,
                           int[:, :]    edge_meta,
                           int[:]       graph_meta,
                           int[:]       edge_parents_left,
                           vector[int]& next_edges,
                           int[:]       output_order,
                           int&         current_output_index,
                           int[:]       child_edges_buffer ) nogil except *:
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
cpdef forwardPass( int[:, :] edge_parents,
                   int[:, :] edge_children,
                   int[:, :] node_meta,
                   int[:, :] edge_meta,
                   int[:]    graph_meta ):
    """ Bredth first search on the graph

        Args:
            edge_parents  : Info about edges' parent nodes
            edge_children : Info about edges' child nodes
            node_meta     : Meta data about nodes
            edge_meta     : Meta data about the edges
            graph_meta    : Meta data about the graph

        Returns:
            output_order : The order that nodes are visited during a bredth first search
            batch_sizes  : The number of elements per batch.  Each batch can be processed
                           in parallel
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
cpdef accessorTest( np.ndarray[int, ndim=2] edge_parents,
                    np.ndarray[int, ndim=2] edge_children,
                    np.ndarray[int, ndim=2] node_meta,
                    np.ndarray[int, ndim=2] edge_meta,
                    np.ndarray[int, ndim=1] graph_meta ):
    """ A test that prints out the full family for every node

        Args:
            edge_parents  : Info about edges' parent nodes
            edge_children : Info about edges' child nodes
            node_meta     : Meta data about nodes
            edge_meta     : Meta data about the edges
            graph_meta    : Meta data about the graph

        Returns:
            None
    """
    cdef int node
    cdef int edge
    cdef int edge_children_index
    cdef int parent_edge
    cdef int n_parents
    cdef int n_siblings
    cdef int[:] parents
    cdef int[:] siblings
    cdef int[:] mates
    cdef int[:] children
    cdef int[:] child_edges

    # Print the parents, siblings, mates and children for each node
    for node in range( node_meta.shape[1] ):

        # Get the parent edge
        edge_children_index = node_meta[NODE_CHDN_IDX, node]

        # Check if this node is a root
        if( edge_children_index != -1 ):

            parent_edge         = edge_children[EDGE, edge_children_index]
            n_parents           = edge_meta[N_PARENTS, parent_edge]
            n_siblings          = edge_meta[N_CHILDREN, parent_edge] - 1
            parents             = np.zeros( n_parents, dtype=np.int32 )
            siblings            = np.zeros( n_siblings, dtype=np.int32 )

            # Retrieve the parents and siblings
            getParents( edge_parents,
                        edge_children,
                        node_meta,
                        edge_meta,
                        graph_meta,
                        node,
                        parents )

            getSiblings( edge_parents,
                         edge_children,
                         node_meta,
                         edge_meta,
                         graph_meta,
                         node,
                         siblings )
        else:
            parents  = np.zeros( 0, dtype=np.int32 )
            siblings = np.zeros( 0, dtype=np.int32 )

        # Iterate over each child edge
        n_child_edges = node_meta[N_CHILD_EDGES, node]

        # Check if this node is a leaf
        if( n_child_edges != 0 ):
            child_edges = np.zeros( n_child_edges, dtype=np.int32 )
            getChildrenEdges( edge_parents,
                              edge_children,
                              node_meta,
                              edge_meta,
                              graph_meta,
                              node,
                              child_edges )

            for edge in child_edges:

                # Populate the mates and siblings arrays
                n_mates    = edge_meta[N_PARENTS, edge] - 1
                n_children = edge_meta[N_CHILDREN, edge]
                mates      = np.zeros( n_mates, dtype=np.int32 )
                children   = np.zeros( n_children, dtype=np.int32 )

                # Retrieve the mates and children
                getMates( edge_parents,
                          edge_children,
                          node_meta,
                          edge_meta,
                          graph_meta,
                          node,
                          edge,
                          mates )

                getChildren( edge_parents,
                             edge_children,
                             node_meta,
                             edge_meta,
                             graph_meta,
                             node,
                             edge,
                             children )
        else:
            mates    = np.zeros( 0, dtype=np.int32 )
            children = np.zeros( 0, dtype=np.int32 )

        # Print the results
        print( 'node', node,
               'parents', np.asarray( parents ),
               'siblings', np.asarray( siblings ),
               'mates', np.asarray( mates ),
               'children', np.asarray( children ) )