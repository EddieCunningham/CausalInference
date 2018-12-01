from host.src.graph import Graph
from compiled.traversal import *
from host.data.example_graphs import *

def test1():

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
        pp_edge_parents, pp_edge_children, node_meta, edge_meta = preprocessSparseGraphForTraversal( edge_parents.astype( np.int32 ),
                                                                                                     edge_children.astype( np.int32 ) )
        print( 'pp_edge_parents\n', pp_edge_parents )
        print( 'pp_edge_children\n', pp_edge_children )
        print( 'node_meta\n', node_meta )
        print( 'edge_meta\n', edge_meta )

        test( pp_edge_parents, pp_edge_children, node_meta, edge_meta )