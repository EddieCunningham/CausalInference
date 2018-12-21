import numpy as np
from compiled.inference import *
from host.data.example_graphs import *


def multiplyTerms( *terms, contract=None ):
    # Basically np.einsum but in log space
    pass

######################################################################

def integrate( integrand, axes ):
    pass

######################################################################

def aWork( full_n_parents, u, vs, order ):
    term = multiplyTerms( terms=( u, *vs ) )
    return extendAxes( term, order, full_n_parents )

######################################################################

def bWorkFBS( transition, emission ):
    return multiplyTerms( terms=( transition, emission ) )

def bWork( full_n_parents, transition, emission, vs ):
    emission = extendAxes( emission, full_n_parents, full_n_parents + 1 )
    vs = [ extendAxes( v, full_n_parents, full_n_parents + 1 ) for v in vs ]
    integrand = multiplyTerms( terms=( transition, emission, *vs ) )
    return integrate( integrand, axes=[ full_n_parents ] )

######################################################################

def uWork( parent_us, parent_vs, sibling_vs, potential ):
    sibling_bs = multiplyTerms( 'bijk->bi', potential, *sibling_vs )
    integrand  = multiplyTerms( 'bijk->bi', potential, *parent_us, *parent_vs, *sibling_bs )
    return integrate( 'bi', integrand )

def vWork( mate_us, mate_vs, child_vs, potential ):
    child_bs  = multiplyTerms( 'bijk->bi', potential, *child_vs )
    integrand = multiplyTerms( 'bijk->bi', *mate_us, *mate_vs, *child_bs )
    return integrate( 'bi', integrand )

######################################################################

def messagePassing( graphs ):
    work_tickets = getGraphWorkTickets( graphs )

    for tickets in work_tickets:

        # All of these can be done in parallel
        for ticket in tickets:

            # Collect all of the data objects
            ticket.nodes

            # Distribute the work
            uWork()
            vWork()

            # Update the data structures
