import networkx as nx
from networkx.algorithms import approximation
import itertools
import numpy as np
from compiled.message_passing import *
from host.src.clique import Clique
from heapq import nsmallest

class MarkovNetwork( nx.Graph ):

    def message_passing_order( self, root_nodes=None, loopy=False, loopy_iters=10, no_batches=False ):
        """ Returns an iterator on how to execute message passing.
            Set loopy to true if the graph has cycles

        Args:
            loopy       - Whether or not to use loopy propagation belief
            loopy_iters - When to terminate loopy propagation belief

        Returns:
            message_batches - The message batches iterator.  Messages
                              in each batch can be computed in parallel
        """
        sparse = self.to_sparse()

        # The message passing algorithm accepts node index values
        sparse_index = dict( [ ( node, i ) for i, node in enumerate( self.nodes ) ] )
        sparse_list = [ sparse_index[ node ] for node in sparse ]

        # Find the roots of the tree
        root_indices = np.array( [] ) if root_nodes is None else np.array( [ sparse_list.index( r ) for r in root_nodes ] )

        # Run the message passing algorithm
        messages, batch_sizes = message_passing( np.array( sparse_list ),
                                                 root_indices,
                                                 n_nodes=len( self.nodes ),
                                                 loopy=loopy,
                                                 loopy_iters=loopy_iters )

        # Build the unpack function depending on whether or not we want parallel batches
        if( no_batches == True ):
            batch_sizes = np.ones( sum( batch_sizes ), dtype=int )
            unpack = lambda node, relative: ( node[0], relative[0] )
        else:
            unpack = lambda nodes, relatives: list( zip( nodes, relatives ) )

        find_relatives = lambda x: x + 1 if x%2 == 0 else x - 1

        # Return the answer
        n_completed = 0
        for batch_size in batch_sizes:
            node_indices = messages[n_completed: n_completed + batch_size]
            relative_indices = np.array( [ find_relatives( i ) for i in node_indices ] )
            nodes     = sparse[ node_indices ]
            relatives = sparse[ relative_indices ]

            # Yield the objects themselves
            yield unpack( nodes, relatives )
            n_completed += batch_size

    @property
    def summary( self ):
        """ Quick summary of the network

        Args:
            None

        Returns:
            summary - String of what to print
        """

        summary = ( '\nNumber of nodes: %d'
                    '\nNumber of edges: %d'
                    '\nNodes: %s'
                    '\nEdges: %s' )%( self.number_of_nodes(),
                                      self.size(),
                                      str( np.array( self.nodes ) ),
                                      str( np.array( self.edges ) ) )
        return summary

    @staticmethod
    def some_forest():
        """ Convenient way to create a forest

        Args:
            None

        Returns:
            graph - Some graph
        """
        graph = MarkovNetwork()

        graph.add_edge( 0, 1 )
        graph.add_edge( 1, 2 )
        graph.add_edge( 3, 2 )
        graph.add_edge( 3, 4 )
        graph.add_edge( 5, 2 )
        graph.add_edge( 6, 5 )
        graph.add_edge( 7, 6 )
        graph.add_edge( 8, 6 )
        graph.add_edge( 2, 9 )

        return graph

    @staticmethod
    def some_graph():
        """ Convenient way to create a graph

        Args:
            None

        Returns:
            graph - Some graph
        """
        graph = MarkovNetwork()
        graph.add_edges_from( Clique.edges_for( [ 0, 1, 4 ] ) )
        graph.add_edges_from( Clique.edges_for( [ 2, 3, 5 ] ) )
        graph.add_edges_from( Clique.edges_for( [ 4, 5, 6 ] ) )
        graph.add_edges_from( Clique.edges_for( [ 4, 5, 7 ] ) )

        return graph

    def to_bayesian_network( self, node_ordering=None ):
        """ Convert this network to a DAG with a certain node ordering

        Args:
            node_ordering - Node ordering.  If None, will be however the nodes
            are ordered under the hood

        Returns:
            digraph - The bayesian network
        """
        from host.src.bayesian_network import BayesianNetwork

        node_ordering = list( self.nodes ) if node_ordering is None else node_ordering
        node_indices = dict( [ ( node, i ) for i, node in enumerate( node_ordering ) ] )

        digraph = BayesianNetwork()

        for node1, node2 in self.edges:
            if( node_indices[node1] < node_indices[node2] ):
                digraph.add_edge( node1, node2 )
            else:
                digraph.add_edge( node2, node1 )

        return digraph

    def to_sparse( self ):
        """ Returns the sparse representation.  This is found
            by keeping the neighbors of nodes in a vertex cover

        Args:
            None

        Returns:
            sparse - The sparse hypergraph representation
                                of the network
        """
        return np.array( self.edges ).ravel()

    @staticmethod
    def from_sparse( sparse ):
        """ Returns the sparse hypergraph representation

        Args:
            sparse - The sparse hypergraph representation
                                of the network

        Returns:
            graph - The MarkovNetwork represented by the hypergraph
        """
        sparse_it = iter( sparse.tolist() )
        edge_list = [ ( it, next( sparse_it ) ) for it in sparse_it ]
        return MarkovNetwork( edge_list )

    def draw( self,
              output_folder='/app/host',
              output_name='graph',
              file_format='png',
              position_program='graphviz',
              prog='neato',
              node_size=1000,
              with_labels=True,
              labels=None,
              **kwargs ):
        """ Draw the markov network.  Optionally, pass an order to
            convert the graph into a directed graph.  This can make
            the graph more visually appealing.

        Args:
            node_ordering - If None, will be however the nodes
            are ordered under the hood

        Returns:
            render - The graphviz render
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        plt.switch_backend('agg')

        # Generate the dave file
        output_folder = output_folder if output_folder[-1] != '/' else output_folder[:-1]
        output_name = output_name if '/' not in output_name else output_name.replace( '/', '' )
        file_format = file_format.replace( '.', '' )

        # Compute the positions of each node
        min_x, min_y, max_x, max_y = [ 0, 0, 0, 0 ]
        pos = {}
        if( position_program == 'graphviz' ):
            pos = nx.drawing.nx_agraph.graphviz_layout( self, prog=prog )
            for node, ( x, y ) in pos.items():
                min_x = min( x, min_x )
                min_y = min( y, min_y )
                max_x = max( x, max_x )
                max_y = max( y, max_y )
        else:
            assert 0, 'Invalid layout'

        fig, ax = plt.subplots( 1, 1 )
        nx.draw( self, ax=ax, pos=pos, node_size=node_size, with_labels=with_labels, labels=labels, **kwargs )

        # Make the graph fit in view
        ax.set_xlim( min_x - 50, max_x + 50 )
        ax.set_ylim( min_y - 50, max_y + 50 )

        plt.savefig( '%s/%s.%s'%( output_folder, output_name, file_format ) )

    # def get_cutset( self ):
    #     """ Find a cutset of this network.  This is a set of nodes
    #         that, when removed, leave the network without any cycles.
    #         Using bad algorithm for the moment.

    #     Args:
    #         None

    #     Returns:
    #         cutset - A cutset for this node
    #     """
    #     assert 0

    #     reduced = self.copy()

    #     # Remove a minimum spanning tree from the graph until there are no cycles.
    #     # This is so that we are left with nodes that comprise cycles
    #     # TODO: DO NOT DELETE CONNECTED COMPONENTS WITH NO CYCLES
    #     cycles = nx.cycle_basis( reduced )
    #     while( len( cycles ) > 0 ):

    #         mst = MarkovNetwork( nx.minimum_spanning_tree( reduced ) )
    #         reduced.remove_edges_from( mst.edges )
    #         cycles = nx.cycle_basis( reduced )

    #     # This function will find the vertex cover of each connected component
    #     def cover_connected_components( reduced_graph ):
    #         vertex_cover = list( approximation.min_weighted_dominating_set( reduced ) )

    #         node_with_most_neighbors = -1
    #         most_neighbors = 0
    #         for node in vertex_cover:
    #             neighbors = list( reduced.neighbors( node ) )
    #             if( len( neighbors ) > 0 and len( neighbors ) > most_neighbors ):
    #                 node_with_most_neighbors = node
    #                 most_neighbors = len( neighbors )

    #         return node_with_most_neighbors

    #     # The cutset will be the vertex cover nodes of each connected component
    #     # in the resulting graph
    #     cutset = []
    #     node_with_most_neighbors = cover_connected_components( reduced )
    #     while( node_with_most_neighbors != -1 ):
    #         cutset.append( node_with_most_neighbors )
    #         reduced.remove_nodes_from( [ node_with_most_neighbors ] )
    #         node_with_most_neighbors = cover_connected_components( reduced )

    #     return cutset

    # def cutset_clustering( self, cutset ):
    #     """ Triangulate the graph using a cutset.  Not really that great

    #     Args:
    #         cutset - The cutset nodes

    #     Returns:
    #         cluster_graph - Cluster graph generated using cutset.
    #     """

    #     assert 0

    #     print( 'Cutset:', cutset )

    #     # Find the loopy message passing order
    #     message_batches = self.message_passing_order( loopy=True, loopy_iters=1 )

    #     # Initialize the dependence sets
    #     for node, data in self.nodes.data():
    #         data['cutset_dependence'] = {}
    #         for neighbor in self.neighbors( node ):
    #             data['cutset_dependence'][neighbor] = set()

    #     # Go through the messages and keep track of which cutset nodes are visited
    #     for messages in message_batches:

    #         print()

    #         for in_node, out_node in messages:

    #             print( in_node, out_node )

    #             # in_node is a summary of the rest of the graph
    #             # out_node now summarizes that part of the graph and in_node

    #             if( in_node in cutset ):
    #                 # The out_node clearly depends on in_node
    #                 self.nodes[out_node]['cutset_dependence'][in_node].add( in_node )
    #             elif( out_node in cutset ):
    #                 continue
    #             else:
    #                 # Otherwise, update the out_node
    #                 in_dep  = self.nodes[in_node]['cutset_dependence']
    #                 out_dep = self.nodes[out_node]['cutset_dependence']

    #                 union = set()
    #                 for i, ( neighbor, accumulated ) in enumerate( in_dep.items() ):
    #                     if( neighbor == out_node ):
    #                         continue
    #                     union = union.union( accumulated )

    #                 intersection = set()
    #                 for i, ( neighbor, accumulated ) in enumerate( in_dep.items() ):
    #                     if( neighbor == out_node ):
    #                         continue

    #                     if( len( intersection ) == 0 ):
    #                         intersection = accumulated
    #                     else:
    #                         intersection = intersection.intersection( accumulated )

    #                 out_dep[in_node] = union - intersection

    #     # Return a graph where each node is unioned with the cutset node
    #     for node, data in self.nodes.data():
    #         print( 'node', node )
    #         for neighbor, accumulated in data['cutset_dependence'].items():
    #             print( 'neighbor', neighbor, 'accumulated', accumulated )

    def elimination_heuristic( self, graph, heuristic='min_fill_in', top_choices=4, selector='inverse_fill_in' ):
        """ Find the next node to eliminate.  Do this by evaluating the heuristic
            over all of the nodes and taking the top choices.  Then, draw the node
            according to some distribution. (https://youtu.be/PXLNbueWCug?t=4046)

        Args:
            graph         - The current induced graph
            heuristic     - The search heuristic
            top_choices   - How many of the best nodes to consider

        Returns:
            node - The next node to eliminate in graph
        """
        assert heuristic in [ 'min_fill_in' ]
        # assert heuristic in [ 'min_fill_in', 'min_size', 'min_weight' ]
        assert selector in [ 'inverse_fill_in' ]

        def min_fill( node ):
            neighbors = list( graph.neighbors( node ) )
            clique_size = ( len( neighbors )**2 - len( neighbors ) ) / 2
            connecting_edges = graph.subgraph( neighbors ).number_of_edges()
            return clique_size - connecting_edges

        # Find the best nodes to use
        best_nodes = nsmallest( top_choices, graph.nodes, key=min_fill )

        # Sample the next node
        if( selector == 'inverse_fill_in' ):
            probs = np.array( [ min_fill( node ) for node in best_nodes ] )
            probs[probs!=0] = 1 / probs[probs!=0]
            probs[probs==0] = 999999999999999999
            probs /= probs.sum()

        index = np.random.choice( len( best_nodes ), 1, p=probs )[0]

        return best_nodes[index]

    def junction_tree( self, **elimination_args ):
        """ Construct a junction tree over this graph

        Args:
            elimination_args - See elimination_heuristic

        Returns:
            junction_tree - The junction tree
        """
        from host.src.junction_tree import JunctionTree

        # Run the elimination algorithm to find the elimination cliques
        maximal_cliques = []
        induced_graph = self.copy()
        while( len( induced_graph.nodes ) > 0 ):

            # Find the next node to eliminate
            node = self.elimination_heuristic( induced_graph, **elimination_args )

            neighbors = list( induced_graph.neighbors( node ) )
            elimination_clique = Clique( set( [ node ] + neighbors ) )

            # Check to see if the current elimination clique is maximal.
            # If it is not a subset of a previous maxclique, then it is maximal (https://youtu.be/7SB67giDEsE?t=2738)
            is_maximal_clique = True
            for max_clique in maximal_cliques:
                if( elimination_clique.is_subset( max_clique ) ):
                    is_maximal_clique = False
                    break

            if( is_maximal_clique ):
                maximal_cliques.append( elimination_clique )

            # Remove the current node and add in the fill-in edges
            induced_graph.remove_nodes_from( [ node ] )
            induced_graph.add_edges_from( Clique.edges_for( neighbors ) )

        # Construct a cluster graph from the maxcliques and weight the edges with the intersection size
        # (https://youtu.be/TddbmU9dHgA?t=4544)
        cluster_graph = MarkovNetwork()

        for i, max_clique1 in enumerate( maximal_cliques ):
            for j, max_clique2 in enumerate( maximal_cliques ):
                if( i == j ):
                    continue

                intersection = max_clique1.intersection( max_clique2 )

                # A cluster graph has edges when the clusters have intersection
                if( len( intersection ) > 0 ):
                    cluster_graph.add_edge( max_clique1, max_clique2, weight=len( intersection ) )

        # Finally, find a maximal spanning tree for the cluster graph
        junction_tree = JunctionTree( nx.maximum_spanning_tree( cluster_graph ) )

        sparse = junction_tree.to_sparse()

        return junction_tree

    def set_clique_potentials( self, potential_map ):
        """ Set the clique potentials for all of the nodes.  Will only go over the max cliques because any
            other clique can be found by marginalizing irrelevant values from the maximal cliques.

        Args:
            potential_map - A dictionary of { max_clique: function } where function( node ) returns the potential

        Returns:
            None
        """
        for max_clique, potential_func in potential_map.items():
            assert isinstance( max_clique, Clique )
            self.potentials[max_clique] = potential_func

    def get_clique_potential( self, nodes, is_maximal_clique=False ):
        """ Get the clique potentials for any node.  Will find all of the maximal cliques
            that the nodes are a part of.  Assumes that the nodes parameter is passed in
            responsibly so that the find_cliques function doesn't run forever

        Args:
            nodes             - A list of nodes that we want the potentials for
            is_maximal_clique - If the nodes passed in are known to be a maximal clique

        Returns:
            potential - The first potential that all nodes in nodes are a part of
        """

        if( not is_maximal_clique ):

            # Find the induced subgraph
            induced_subgraph = self.subgraph( nodes )

            # Find all of the maximal cliques in subgraph.
            max_clique = None
            for clique in nx.find_cliques( induced_graph ):
                if( sum( [ node in clique for node in nodes ] ) == 1 ):
                    max_clique = Clique( clique )
                    break
        else:
            max_clique = Clique( nodes )

        # Find the correct potential
        return self.potentials[max_clique]
