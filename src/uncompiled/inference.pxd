from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

cdef struct WorkTicket:

    # The nodes this ticket is for
    vector[int] nodes

    # Need to know the computation type to ensure compatibility
    int computation_type

    # The nodes to evaluate the potential over
    vector[vector[int]] potentials

    # Incoming u nodes for every node
    vector[vector[pair[int,vector[int]]]]            u_nodes

    # Incoming v nodes
    vector[vector[pair[int,vector[pair[int, int]]]]] v_nodes

    # Which nodes to integrate out at this step
    vector[vector[int]] nodes_to_integrate

    # If we are looping
    bool loopy