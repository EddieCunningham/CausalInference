from host.src.markov_network import MarkovNetwork
from host.src.junction_tree import JunctionTree
import time
import networkx as nx

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

def cutset_test():
    graph = MarkovNetwork.some_graph()
    graph.draw( output_folder='/app/host', output_name='graph' )
    cutset = graph.get_cutset()

    cut_graph = graph.copy()
    cut_graph.remove_nodes_from( cutset )
    cut_graph.draw( output_folder='/app/host', output_name='cut_graph' )

    cluster_graph = graph.cutset_clustering( cutset )

def message_passing_test():
    graph = MarkovNetwork.some_graph()
    graph.draw()
    messages = graph.message_passing_order( loopy=True, loopy_iters=0 )

    total_messages = set()
    for m in messages:
        message_set = set()
        for b in m.tolist():
            message_set.add( tuple( b ) )

        print( 'message_set', message_set )
        total_messages = total_messages.union( message_set )

    print( total_messages )

def junction_tree_test():
    jt = JunctionTree.example_junction_tree()
    jt.draw()
    rip = JunctionTree.satisfies_running_intersection_property( jt )
    print( 'Satisfies rip:', rip, 'Expect True' )

    jt = JunctionTree.example_non_junction_tree()
    jt.draw()
    rip = JunctionTree.satisfies_running_intersection_property( jt )
    print( 'Satisfies rip:', rip, 'Expect False' )

def junction_tree_conversion_test():
    graph = MarkovNetwork( nx.karate_club_graph() )
    graph.draw( output_folder='/app/host', output_name='graph' )

    junction_tree = graph.junction_tree()

    is_jt = JunctionTree.is_junction_tree( graph, junction_tree )

    print( 'is_jt', is_jt )

def junction_tree_inference_test():
    graph = MarkovNetwork( nx.circular_ladder_graph( 5 ) )
    graph.remove_nodes_from( [ 0, 5 ] )
    graph.draw()

    junction_tree = graph.junction_tree()
    junction_tree.draw()



    print( junction_tree.nodes )

def allMarkovNetworkTests():
    # test_to_bayesian_network()
    # sparse_graph_test()
    # cutset_test()
    # message_passing_test()
    # junction_tree_test()
    # junction_tree_conversion_test()
    junction_tree_inference_test()
    print( 'Completed all markov network tests' )