from host.src.graph import Graph
from compiled.traversal import *
from host.data.example_graphs import *
import time

def processGraphs( graphs ):
    sparses = [ graph.toSparse() for graph in graphs ]
    edge_parents, edge_children = Graph.combineSparse( sparses )
    return preprocessSparseGraphForTraversal( edge_parents.astype( np.int32 ), edge_children.astype( np.int32 ) )

def treeAccessorTest():

    graphs = [ polyTree1(),
               polyTree2(),
               polyTree3(),
               polyTree4(),
               polyTree5(),
               polyTree6(),
               polyTree7(),
               graph1(),
               graph2(),
               graph3(),
               graph7(),
               graph8(),
               graph9(),
               graph10(),
               graph11(),
               graph12(),
               graph13() ]

    for graph in graphs:
        # graph.draw( output_folder='/app/host' )

        edge_parents, edge_children = graph.toSparse()
        print( 'edge_parents', edge_parents )
        print( 'edge_children', edge_children )
        pp_graph = preprocessSparseGraphForTraversal( edge_parents.astype( np.int32 ), edge_children.astype( np.int32 ) )
        pp_edge_parents, pp_edge_children, node_meta, edge_meta, graph_meta = pp_graph
        print( 'pp_edge_parents\n', pp_edge_parents )
        print( 'pp_edge_children\n', pp_edge_children )
        print( 'node_meta\n', node_meta )
        print( 'edge_meta\n', edge_meta )
        print( 'graph_meta\n', graph_meta )

        accessorTest( pp_edge_parents, pp_edge_children, node_meta, edge_meta, graph_meta )

def forwardPassTest():
    graphs = [ polyTree1(),
               polyTree2(),
               polyTree3(),
               polyTree4(),
               polyTree5(),
               polyTree6(),
               polyTree7(),
               graph1(),
               graph2(),
               graph3(),
               graph7(),
               graph8(),
               graph9(),
               graph10(),
               graph11(),
               graph12(),
               graph13() ]
    # graphs = [ polyTree1(),
    #            polyTree2() ]

    sparses = [ graph.toSparse() for graph in graphs ]
    edge_parents, edge_children = Graph.combineSparse( sparses )
    big_sparse = preprocessSparseGraphForTraversal( edge_parents.astype( np.int32 ), edge_children.astype( np.int32 ) )

    cython_order, batch_sizes = forwardPass( *big_sparse )

    print( cython_order )
    print( batch_sizes )


    start = time.time()
    big_sparse = processGraphs( graphs )
    cython_order, batch_sizes = forwardPass( *big_sparse )
    end = time.time()
    print( 'Cython way took', end - start )

    big_sparse = processGraphs( graphs )
    start = time.time()
    cython_order, batch_sizes = forwardPass( *big_sparse )
    end = time.time()
    print( 'Cython way (no preprocess) took', end - start )

    start = time.time()
    for graph in graphs:
        python_order = [ x for x in graph.forwardPass() ]
    end = time.time()
    print( 'Python way took', end - start )

    start = time.time()
    for graph in graphs:
        python_order = [ x for x in graph.forwardPass() ]
    end = time.time()
    print( 'Python way (no preprocess) took', end - start )
