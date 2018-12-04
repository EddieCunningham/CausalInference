
"""
This file contains an optimized implementation of bredth first search
and message passing over polytree.  It uses a sparse polytree
format and does preprocessing to make everything fast.  Below is a small
example (graph9) of each of the data structures used throughout this file.

#############################

edge_parents: Contains the parents for each edge and a linked list between nodes
index        [ 0  1  2  3  4  5  6  7  8]

EDGE         [ 0  0  1  1  2  2  3  3  3]  Edge index (child edge of node)
NODE         [ 7  8  0  1  1  2  2  3  4]  Node index
NEXT_FAMILY  [-1 -1 -1  4 -1  6 -1 -1 -1]  The next index in this array where node appears.  -1 if last edge for node

#############################

edge_children: Contains the children for each edge.  Has n_nodes elements
index  [0 1 2 3 4]

EDGE   [0 1 2 3 3]  Edge index (parent edge of node)
NODE   [0 3 4 5 6]  Node index

#############################

node_meta: Location of node in edge_children and edge_parents
index (node)         [ 0  1  2  3  4  5  6  7  8]

NODE_CHILDREN_INDEX  [ 0 -1 -1  1  2  3  3 -1 -1]  Index of parent edge in edge_children.  -1 if root
NODE_PARENTS_INDEX   [ 2  2  4  6  6 -1 -1  0  0]  Index of first child edge in edge_parents.  -1 if leaf
N_CHILD_EDGES        [ 1  2  2  1  1  0  0  1  1]  The number of child edges

#############################

edge_meta: Location of edge in edge_parents
index (edge)         [0 1 2 3]

EDGE_PARENT_INDEX    [0 2 4 6]  Index of edge in edge_parents
EDGE_CHILDREN_INDEX  [0 1 2 3]  Index of edge in edge_children
N_PARENTS            [2 2 2 3]  Number of parents
N_CHILDREN           [1 1 1 2]  Number of children

#############################

graph_meta: Meta data about the number of roots and leaves
[ N_ROOTS, N_LEAVES, MAX_CHILD_EDGES, MAX_PARENTS, MAX_SIBLINGS, MAX_MATES, MAX_CHILDREN ]
[4 2 2 3 3]
"""

# Can't access memoryviews with enums, so using this ugly way
# of just hard coding values
cdef extern from 'definitions.h' namespace 'cython_accessors':
    cdef int EDGE            # EDGE
    cdef int NODE            # NODE
    cdef int NEXT_FMLY       # NEXT_FAMILY

    cdef int NODE_CHDN_IDX   # NODE_CHILDREN_INDEX
    cdef int NODE_PRNT_IDX   # NODE_PARENTS_INDEX
    cdef int N_CHILD_EDGES   # N_CHILD_EDGES

    cdef int EDGE_PRNT_IDX   # EDGE_PARENT_INDEX
    cdef int EDGE_CHDN_IDX   # EDGE_CHILDREN_INDEX
    cdef int N_PARENTS       # N_PARENTS
    cdef int N_CHILDREN      # N_CHILDREN

    cdef int N_ROOTS         # N_ROOTS
    cdef int N_LEAVES        # N_LEAVES
    cdef int MAX_CHILD_EDGES # MAX_CHILD_EDGES
    cdef int MAX_PARENTS     # MAX_PARENTS
    cdef int MAX_CHILDREN    # MAX_CHILDREN
