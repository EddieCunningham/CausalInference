import cython
import numpy as np
cimport numpy as np
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

# @cython.boundscheck( False )
# @cython.wraparound( False )
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
                                                int[:]                    child_edges_buffer ) nogil except *