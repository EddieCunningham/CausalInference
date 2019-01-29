from host.src.markov_network import MarkovNetwork
from host.src.junction_tree import JunctionTree
from host.src.discrete_network import DiscreteNetwork
import time
import networkx as nx
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

__all__ = [ 'allTensorflowTests' ]

def create_graph():
    # graph = DiscreteNetwork( nx.karate_club_graph() )
    graph = DiscreteNetwork( nx.generators.balanced_tree( 3, 3 ) )

    # graph = DiscreteNetwork( nx.karate_club_graph() )
    # graph = DiscreteNetwork( nx.circular_ladder_graph( 7 ) )
    # graph.remove_nodes_from( [ 0, 5 ] )
    # graph.draw()
    print( 'Number of nodes', len( list( graph.nodes ) ) )
    print( 'Number of edges', len( list( graph.edges ) ) )

    # Set the state sizes
    state_sizes = dict( [ ( node, 2 ) for node in graph.nodes ] )
    # state_sizes = dict( [ ( node, np.random.randint( 3, 8 ) ) for node in graph.nodes ] )
    graph.set_state_sizes( state_sizes )

    # Set the clique potentials
    # DON'T USE A BIG GRAPH!!!! THIS IS JUST FOR TESTING
    max_cliques = [ tuple( sorted( x ) ) for x in nx.find_cliques( graph )  ]
    clique_sizes = [ tuple( [ state_sizes[node] for node in max_clique ] ) for max_clique in max_cliques ]
    potentials = dict( [ ( max_clique, tf.Variable( np.random.random( clique_size ) ) ) for max_clique, clique_size in zip( max_cliques, clique_sizes ) ] )
    graph.set_potentials( potentials )

    ######################################################

    print( 'Starting to find best elimination order' )
    order, fastest_time = graph.find_best_elimination_order( n_iters=1 )
    print( 'fastest_time', fastest_time )

    nx.write_yaml( graph, './host/tf_graph.yaml' )

def evidence_test():

    graph = nx.read_yaml( './host/tf_graph.yaml' )

    instructions = graph.get_computation_instructions( graph.best_elimination_order )

    comp_start = time.time()
    for _ in range( 10 ):

        graph.perform_message_passing( *instructions )

    print( 'Total computation time', time.time() - comp_start )

def allTensorflowTests():

    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()

    ######################################################

    graph = DiscreteNetwork( nx.generators.balanced_tree( 3, 3 ) )
    state_sizes = dict( [ ( node, 2 ) for node in graph.nodes ] )
    graph.set_state_sizes( state_sizes )

    # Set the clique potentials
    max_cliques = [ tuple( sorted( x ) ) for x in nx.find_cliques( graph )  ]
    clique_sizes = [ tuple( [ state_sizes[node] for node in max_clique ] ) for max_clique in max_cliques ]
    potentials = dict( [ ( max_clique, tf.Variable( np.random.random( clique_size ) ) ) for max_clique, clique_size in zip( max_cliques, clique_sizes ) ] )
    graph.set_potentials( potentials )

    print( 'Starting to find best elimination order' )
    order, fastest_time = graph.find_best_elimination_order( n_iters=1 )
    print( 'fastest_time', fastest_time )

    ######################################################

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    ######################################################
    ######################################################
    ######################################################
    ######################################################
    ######################################################

    # create_graph()
    # evidence_test()
    print( 'Completed all tensorflow tests' )