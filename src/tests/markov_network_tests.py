from host.src.markov_network import MarkovNetwork
from host.src.junction_tree import JunctionTree
from host.src.discrete_network import DiscreteNetwork
import time
import networkx as nx
import numpy as np

__all__ = [ 'allMarkovNetworkTests' ]

def test_to_bayesian_network():
    graph = MarkovNetwork.some_graph()
    digraph = graph.to_bayesian_network( node_ordering=sorted( graph.nodes ) )
    digraph.draw( output_folder='/app/host' )

def sparse_graph_test():
    graph = MarkovNetwork.some_graph()
    sparse = graph.to_sparse()
    graph_back = MarkovNetwork.from_sparse( sparse )
    print( graph.summary )
    print( graph_back.summary )

def message_passing_test():
    graph = MarkovNetwork.some_graph()
    graph.draw()
    messages = graph.message_passing_order( loopy=True, loopy_iters=0 )

    total_messages = set()
    for m in messages:
        message_set = set()
        for b in m:
            message_set.add( tuple( b ) )

        print( 'message_set', message_set )
        total_messages = total_messages.union( message_set )

    print( total_messages )

def junction_tree_test():
    jt = JunctionTree.example_junction_tree()
    jt.draw()
    rip = jt.has_running_intersection_property()
    print( 'Satisfies rip:', rip, 'Expect True' )

    jt = JunctionTree.example_non_junction_tree()
    jt.draw()
    rip = jt.has_running_intersection_property()
    print( 'Satisfies rip:', rip, 'Expect False' )

def junction_tree_conversion_test():
    graph = MarkovNetwork( nx.karate_club_graph() )
    graph.draw( output_folder='/app/host', output_name='graph' )

    junction_tree, _ = graph.junction_tree()

    is_jt = JunctionTree.is_junction_tree( graph, junction_tree )

    print( 'is_jt', is_jt )

def junction_tree_inference_test():
    # graph = DiscreteNetwork( nx.karate_club_graph() )
    graph = DiscreteNetwork( nx.generators.balanced_tree( 2, 2 ) )

    # graph = DiscreteNetwork( nx.karate_club_graph() )
    # graph = DiscreteNetwork( nx.circular_ladder_graph( 7 ) )
    # graph.remove_nodes_from( [ 0, 5 ] )
    graph.draw()
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
    potentials = dict( [ ( max_clique, np.random.random( clique_size ) ) for max_clique, clique_size in zip( max_cliques, clique_sizes ) ] )
    graph.set_potentials( potentials )

    ######################################################
    start = time.time()

    # Run variable elimination
    order, max_clique_potential_instructions, junction_tree = graph.variable_elimination( potentials,
                                                                                          return_maximal_cliques=True,
                                                                                          draw=False,
                                                                                          return_junction_tree=True )

    # Create the junction tree and the computation instructions
    junction_tree.draw()
    instructions = junction_tree.shafer_shenoy_inference_instructions()

    # Generate the instructions to do inference
    supernode_potential_instructions = graph.parse_max_clique_potential_instructions( max_clique_potential_instructions, junction_tree.nodes )
    separators, computation_instructions = graph.parse_inference_instructions( instructions )

    # Generate the contractions list.  This makes the log_einsum calls fast
    contraction_lists = graph.generate_contractions( supernode_potential_instructions, separators, computation_instructions )

    ######################################################

    # Run inference.
    comp_start = time.time()
    graph.perform_message_passing( supernode_potential_instructions, separators, computation_instructions, contraction_lists )

    print( 'Total computation time', time.time() - comp_start )
    print( 'Total time', time.time() - start )

def junction_tree_best_inference_test():
    graph = DiscreteNetwork( nx.karate_club_graph() )
    # graph = DiscreteNetwork( nx.generators.balanced_tree( 2, 2 ) )

    # graph = DiscreteNetwork( nx.karate_club_graph() )
    # graph = DiscreteNetwork( nx.circular_ladder_graph( 7 ) )
    # graph.remove_nodes_from( [ 0, 5 ] )
    graph.draw()
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
    potentials = dict( [ ( max_clique, np.random.random( clique_size ) ) for max_clique, clique_size in zip( max_cliques, clique_sizes ) ] )
    graph.set_potentials( potentials )

    ######################################################

    order, fastest_time = graph.find_best_elimination_order( n_iters=1 )
    print( 'fastest_time', fastest_time )
    instructions = graph.get_computation_instructions( order )

    comp_start = time.time()
    for _ in range( 10 ):

        graph.perform_message_passing( *instructions )

    print( 'Total computation time', time.time() - comp_start )

def evidence_test():
    # graph = DiscreteNetwork( nx.karate_club_graph() )
    # graph = DiscreteNetwork( nx.generators.balanced_tree( 3, 3 ) )

    # graph = DiscreteNetwork( nx.karate_club_graph() )
    graph = DiscreteNetwork( nx.circular_ladder_graph( 7 ) )
    graph.remove_nodes_from( [ 0, 5 ] )
    graph.draw()
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
    potentials = dict( [ ( max_clique, np.random.random( clique_size ) ) for max_clique, clique_size in zip( max_cliques, clique_sizes ) ] )
    graph.set_potentials( potentials )

    ######################################################

    # Add evidence
    nodes_with_evidence = []
    data = []
    for node in graph.nodes:
        if( node != 0 ):
            continue
        # if( 0.4 > np.random.random() ):
            # continue

        p = np.arange( state_sizes[node] )**2
        p = p / p.sum()
        n_possible_states = np.random.choice( state_sizes[node], 1, p=p )[0]
        possible_states   = np.random.choice( state_sizes[node], n_possible_states, replace=False )

        print( 'node', node )
        print( 'possible_states', possible_states )

        nodes_with_evidence.append( node )
        data.append( possible_states )

    graph.add_evidence( nodes_with_evidence, data )

    ######################################################

    order, fastest_time = graph.find_best_elimination_order( n_iters=1 )
    print( 'fastest_time', fastest_time )

    instructions = graph.get_computation_instructions( order )

    comp_start = time.time()
    for _ in range( 0 ):

        graph.perform_message_passing( *instructions )

    print( 'Total computation time', time.time() - comp_start )

def allMarkovNetworkTests():
    # test_to_bayesian_network()
    # sparse_graph_test()
    # message_passing_test()
    # junction_tree_test()
    # junction_tree_conversion_test()
    # junction_tree_inference_test()
    # junction_tree_best_inference_test()
    evidence_test()
    print( 'Completed all markov network tests' )