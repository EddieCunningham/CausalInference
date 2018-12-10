from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

cdef struct WorkTicket:

    # The node this ticket is for
    int node

    # The nodes to evaluate the potential over
    vector[int] potential

    # Incoming u nodes
    vector[pair[int,vector[int]]]            u_nodes

    # Incoming v nodes
    vector[pair[int,vector[pair[int, int]]]] v_nodes

    # Which nodes to integrate out at this step
    vector[int] nodes_to_integrate

    # If this is for loopy propogation belief
    bool loopy