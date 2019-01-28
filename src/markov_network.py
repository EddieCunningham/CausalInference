import networkx as nx
from networkx.algorithms import approximation
import itertools
import numpy as np
from compiled.message_passing import *
from host.src.clique import Clique
from heapq import nsmallest
from collections import namedtuple
from .red_black_tree import RedBlackTree, NodeData

class MarkovNetwork( nx.Graph ):

    Message = namedtuple( 'Message', [ 'incoming', 'outgoing' ] )
    MaximalCliqueInstructions = namedtuple( 'MaximalCliqueInstructions', [ 'integrate_out', 'elimination_nodes', 'factors_to_combine' ] )

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

    def build_clique( self, nodes ):
        """ Build a clique over nodes.  Subclasses of MarkovNetwork should overload this
        Args:
            nodes - A list of nodes to build a clique over
        Returns:
            clique - The clique object
        """
        return Clique( nodes )

    @property
    def best_elimination_order( self ):
        if( not hasattr( self, '_best_elimination_order' ) ):
            self.best_elimination_order = None
        return self._best_elimination_order

    @best_elimination_order.setter
    def best_elimination_order( self, value ):
        self._best_elimination_order = value

    def variable_elimination( self,
                              clique_factorization=None,
                              order=None,
                              return_maximal_cliques=False,
                              heuristic='min_fill_in',
                              top_choices=4,
                              selector='inverse_fill_in',
                              draw=False ):
        """ Run variable elimination using a heuristic.
            Currently only implemented min_fill_in and top_choices=N.
            TODO:  Implement the other variable elimination options

        Args:
            clique_factorization   - The known cliques that make up the joint distribution
            order                  - A known elimination order.  If unknown, will use a heuristic to find one
            return_maximal_cliques - Whether or not to return the maximal cliques of the induced graph
            heuristic              - The search heuristic
            top_choices            - How many of the best nodes to consider
            selector               - How to sample from the top choices
            draw                   - Whether or not to draw the induced graphs

        Returns:
            order               - The elimination order
            factor_instructions - The instructions on how to create the factors
        """

        elimination_args = dict( heuristic=heuristic, top_choices=top_choices, selector=selector )

        def fill_in_to_weight( value ):
            if( value == 0 ):
                return 50
            return 10 / value**2

        def n_fill_in( graph, node ):
            neighbors = list( graph.neighbors( node ) )
            clique_size = ( len( neighbors )**2 - len( neighbors ) ) / 2
            connecting_edges = graph.subgraph( neighbors ).number_of_edges()
            return clique_size - connecting_edges

        def weight_for_node( graph, node ):
            fill_in = n_fill_in( graph, node )
            return fill_in_to_weight( fill_in )

        def update_elimination_queue( elimination_queue, graph, node, node_to_data, data_to_node ):
            if( node in node_to_data ):
                # Remove the node
                data = node_to_data[ node ]
                elimination_queue.remove( data )
                del node_to_data[ node ]
                del data_to_node[ data ]
            # Add the updated weight
            weight = weight_for_node( induced_graph, node )
            data = NodeData( weight )
            node_to_data[ node ] = data
            data_to_node[ data ] = node
            elimination_queue.add( data )

        # Run the elimination algorithm to find the elimination cliques
        maximal_cliques = []
        factor_instructions = []
        induced_graph = self.copy()
        should_find_order = order is None

        # Create the elimination queue
        if( should_find_order ):
            order = []
            node_to_data = {}
            data_to_node = {}
            elimination_queue = RedBlackTree()
            for node in induced_graph.nodes:
                update_elimination_queue( elimination_queue, induced_graph, node, node_to_data, data_to_node )

        factors = set()
        i = 0
        while( len( induced_graph.nodes ) > 0 ):

            if( draw == True ):
                induced_graph.draw( output_name='induced_graph_%d'%( i ) )

            i += 1

            # Find the next node to eliminate and remove node from the elimination queue
            if( should_find_order ):
                data = elimination_queue.sample()
                node = data_to_node[ data ]
                elimination_queue.remove( data )
                del node_to_data[ node ]
                del data_to_node[ data ]

                # Keep track of the elimination order
                order.append( node )
            else:
                node = order.pop( 0 )

            # Find the neighbors and make the tuple of nodes in the elimination clique
            neighbors = tuple( sorted( list( induced_graph.neighbors( node ) ) ) )
            elimination_nodes = tuple( sorted( [ node ] + list( neighbors ) ) )

            # Go through all of the cliques in the current scope of nodes and see which
            # ones we need in order to compute the next factor
            current_scope = induced_graph.subgraph( elimination_nodes )
            all_cliques = nx.enumerate_all_cliques( current_scope )

            # Find the cliques that are needed to compute the next factor
            factors_to_combine = []
            for clique in all_cliques:
                sorted_clique = tuple( sorted( clique ) )
                if( clique_factorization is None or ( ( sorted_clique in clique_factorization ) or ( sorted_clique in factors ) ) ):
                    factors_to_combine.append( sorted_clique )

            # Add the terms used to the next factor
            factors.add( neighbors )

            # Update the instructions
            factor_instructions.append( MarkovNetwork.MaximalCliqueInstructions( node, elimination_nodes, factors_to_combine ) )

            # Check to see if the current elimination clique is maximal.
            # There is probably a better way to do this
            if( return_maximal_cliques ):
                is_maximal_clique = True
                elimination_clique = self.build_clique( elimination_nodes )
                for max_clique in maximal_cliques:
                    if( elimination_clique.is_subset( max_clique ) ):
                        is_maximal_clique = False
                        break

                if( is_maximal_clique ):
                    maximal_cliques.append( elimination_clique )

            # Remove the current node and add in the fill-in edges
            induced_graph.remove_nodes_from( [ node ] )
            induced_graph.add_edges_from( Clique.edges_for( neighbors ) )

            if( should_find_order ):
                # Re-calculate the fill in weights for each of the neighbors
                for neighbor in neighbors:
                    update_elimination_queue( elimination_queue, induced_graph, neighbor, node_to_data, data_to_node )

        if( return_maximal_cliques ):
            return order, factor_instructions, maximal_cliques

        return order, factor_instructions

    def junction_tree( self, triangulated_graph_maximal_cliques ):
        """ Construct a junction tree using the maximal cliques of a triangulated graph.
            Call this after getting the maximal cliques from variable elimination

        Args:
            triangulated_graph_maximal_cliques - The max cliques of a triangulated graph

        Returns:
            junction_tree - The junction tree
        """
        from host.src.junction_tree import JunctionTree

        # Construct a cluster graph from the maxcliques and weight the edges with the intersection size
        # (https://youtu.be/TddbmU9dHgA?t=4544)
        cluster_graph = MarkovNetwork()

        for i, max_clique1 in enumerate( triangulated_graph_maximal_cliques ):
            for j, max_clique2 in enumerate( triangulated_graph_maximal_cliques ):
                if( i == j ):
                    continue

                intersection = max_clique1.intersection( max_clique2 )

                # A cluster graph has edges when the clusters have intersection
                if( len( intersection ) > 0 ):
                    cluster_graph.add_edge( max_clique1, max_clique2, weight=len( intersection ) )

        cluster_graph.draw( output_name='cluster_graph' )

        # Finally, find a maximal spanning tree for the cluster graph
        return JunctionTree( nx.maximum_spanning_tree( cluster_graph ) )
