from host.src.graph import Graph
from compiled.traversal import *
from compiled.inference import *
from host.data.example_graphs import *
import time
from collections import deque

__all__ = [ 'allInferenceTests' ]

def polytreeTest():
    graphs = [ polyTree1(),
               polyTree2(),
               polyTree3(),
               polyTree4(),
               polyTree5(),
               polyTree6(),
               polyTree7() ]

    graphs = [ polyTree2() ]


    sparses = [ graph.toSparse() for graph in graphs ]
    edge_parents, edge_children = Graph.combineSparse( sparses )
    if( False ):
        big_graph = Graph.fromSparse( edge_parents, edge_children )
        big_graph.draw( output_folder='/app/host' )

    start = time.time()
    big_sparse = preprocessSparseGraphForTraversal( edge_parents.astype( np.int32 ), edge_children.astype( np.int32 ) )
    edge_parents, edge_children, node_meta, edge_meta, graph_meta = big_sparse
    feedback_set = np.array( [ 0, 4, 1 ], dtype=np.int32 )
    u_order, v_order, batch_sizes = inferenceInstructions( edge_parents,
                                                           edge_children,
                                                           node_meta,
                                                           edge_meta,
                                                           graph_meta,
                                                           feedback_set )
    end = time.time()

    print( 'Cython way took', end - start )

    print( 'There are', node_meta.shape[1], 'nodes' )
    print( 'There are', edge_meta.shape[1], 'edges' )

    print( 'batch_sizes', batch_sizes )

    u_deque = deque( u_order )
    v_deque = deque( v_order )

    for i, ( u_batch_size, v_batch_size ) in enumerate( batch_sizes ):

        u_nodes           = [ u_deque.popleft() for _ in range( u_batch_size ) ]
        v_edges_and_nodes = [ v_deque.popleft() for _ in range( v_batch_size ) ]

        print( '\nbatch number', i )
        print( 'u_batch_size', u_batch_size )
        print( 'v_batch_size', v_batch_size )
        print( 'u_nodes', u_nodes )
        print( 'v_edges_and_nodes', v_edges_and_nodes )


def allInferenceTests():
    polytreeTest()