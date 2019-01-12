import networkx as nx
import numpy as np
from .markov_network import MarkovNetwork
from host.src.clique import Clique
import itertools

class JunctionTree( MarkovNetwork ):

    @property
    def super_nodes( self ):
        return self.nodes

    @property
    def leaves( self ):
        return [ node for node in self.nodes if len( list( self.neighbors( node ) ) ) == 1 ]

    @staticmethod
    def example_junction_tree():
        A = MarkovNetwork()
        A.add_edges_from( Clique.edges_for( [ 1, 2, 3 ] ) )

        B = MarkovNetwork()
        B.add_edges_from( Clique.edges_for( [ 2, 3 ] ) )

        C = MarkovNetwork()
        C.add_edges_from( Clique.edges_for( [ 2, 3, 6 ] ) )

        D = MarkovNetwork()
        D.add_edges_from( Clique.edges_for( [ 6, 7 ] ) )

        E = MarkovNetwork()
        E.add_edges_from( Clique.edges_for( [ 3, 6, 7 ] ) )

        F = MarkovNetwork()
        F.add_edges_from( Clique.edges_for( [ 6, 7, 8 ] ) )

        G = MarkovNetwork()
        G.add_edges_from( Clique.edges_for( [ 3, 4, 5 ] ) )

        H = MarkovNetwork()
        H.add_edges_from( Clique.edges_for( [ 4, 5, 9 ] ) )

        jt = JunctionTree()
        jt.add_edge( A, B )
        jt.add_edge( B, C )
        jt.add_edge( B, G )
        jt.add_edge( G, H )
        jt.add_edge( C, E )
        jt.add_edge( E, D )
        jt.add_edge( E, F )

        nx.set_node_attributes( jt, values={ A: { 'name': 'A' },
                                             B: { 'name': 'B' },
                                             C: { 'name': 'C' },
                                             D: { 'name': 'D' },
                                             E: { 'name': 'E' },
                                             F: { 'name': 'F' },
                                             G: { 'name': 'G' },
                                             H: { 'name': 'H' } },
                                     name='name' )
        return jt

    @staticmethod
    def example_non_junction_tree():
        A = MarkovNetwork()
        A.add_edges_from( Clique.edges_for( [ 1, 2, 3 ] ) )

        B = MarkovNetwork()
        B.add_edges_from( Clique.edges_for( [ 2, 3 ] ) )

        C = MarkovNetwork()
        C.add_edges_from( Clique.edges_for( [ 2, 3, 6 ] ) )

        D = MarkovNetwork()
        D.add_edges_from( Clique.edges_for( [ 6, 7 ] ) )

        E = MarkovNetwork()
        E.add_edges_from( Clique.edges_for( [ 3, 6, 7 ] ) )

        F = MarkovNetwork()
        F.add_edges_from( Clique.edges_for( [ 6, 7, 8 ] ) )

        G = MarkovNetwork()
        G.add_edges_from( Clique.edges_for( [ 3, 4, 6 ] ) )

        H = MarkovNetwork()
        H.add_edges_from( Clique.edges_for( [ 4, 5, 9 ] ) )

        jt = JunctionTree()
        jt.add_edge( A, B )
        jt.add_edge( B, C )
        jt.add_edge( B, G )
        jt.add_edge( G, H )
        jt.add_edge( C, E )
        jt.add_edge( E, D )
        jt.add_edge( E, F )

        nx.set_node_attributes( jt, values={ A: { 'name': 'A' },
                                             B: { 'name': 'B' },
                                             C: { 'name': 'C' },
                                             D: { 'name': 'D' },
                                             E: { 'name': 'E' },
                                             F: { 'name': 'F' },
                                             G: { 'name': 'G' },
                                             H: { 'name': 'H' } },
                                     name='name' )
        return jt

    def draw( self,
              output_folder='/app/host',
              output_name='juntion_tree',
              file_format='png' ):
        """ Wrapper around the draw function so that we print out nice junction trees

        Args:
            node_ordering - If None, will be however the nodes
            are ordered under the hood

        Returns:
            render - The graphviz render
        """
        # Convert each of the nodes into scalar objects
        node_map = dict( [ ( cluster, i ) for i, cluster in enumerate( self.super_nodes ) ] )

        # Re-create the scalar graph
        as_graph = MarkovNetwork()
        for cluster1, cluster2 in self.edges:
            as_graph.add_edge( node_map[cluster1], node_map[cluster2] )

        # Mark the labels for each node
        labels = {}
        for cluster, i in node_map.items():
            labels[i] = str( cluster.nodes )

        return as_graph.draw( output_folder=output_folder, output_name=output_name, file_format=file_format, labels=labels )

    def to_cluster_graph( self ):
        """ Convert the nodes in this tree to sets

        Args:
            None

        Returns:
            cluster_tree - A cluster_tree that must be composed of unordered clusters of nodes
        """
        graph = MarkovNetwork()
        for cluster1, cluster2 in self.edges:
            graph.add_edge( set( cluster1.nodes ), set( cluster2.nodes ) )
        return graph

    @classmethod
    def satisfies_running_intersection_property( cls, cluster_tree ):
        """ Check the running intersection property.

        Args:
            cluster_tree - A cluster_tree that must be composed of super_nodes of nodes

        Returns:
            Whether or not this cluser graph satisfies the rip
        """

        # Start from every leaf and go inwards.
        message_passing_order = cluster_tree.message_passing_order( no_batches=True )

        union_at_nodes = {}
        for super_node, adjacent_super_node in message_passing_order:

            # Build up the union tree so that we can check the running intersection property
            if( super_node not in union_at_nodes ):
                union_at_nodes[super_node] = set( super_node.nodes )

            # Check if the rip holds
            intersection_of_union_nodes = union_at_nodes[super_node].intersection( set( adjacent_super_node.nodes ) )
            intersection_nodes = set( super_node.nodes ).intersection( set( adjacent_super_node.nodes ) )

            # This must hold for the rip to hold
            if( intersection_nodes != intersection_of_union_nodes ):
                return False

            # Update the union of the super_nodes
            if( adjacent_super_node not in union_at_nodes ):
                union_at_nodes[adjacent_super_node] = set( adjacent_super_node.nodes ).union( union_at_nodes[super_node] )
            else:
                union_at_nodes[adjacent_super_node] = union_at_nodes[adjacent_super_node].union( union_at_nodes[super_node] )

        return True

    @classmethod
    def is_junction_tree( cls, original_graph, tree ):
        """ Check to see if tree is actually a junction tree

        Args:
            original_graph - The original graph
            tree           - The (potential) junction tree of tree

        Returns:
            Whether or not tree is a junction tree of original_graph
        """
        # Must be a tree
        if( nx.is_tree( tree ) == False ):
            print( 'Not a tree' )
            return False

        # Must be a JunctionTree instance
        if( isinstance( tree, JunctionTree ) == False ):
            print( 'Not a JunctionTree' )
            return False

        # Every node must be a cluster
        for cluster in tree.nodes:
            if( isinstance( cluster, MarkovNetwork ) == False ):
                print( 'Doesn\'t have a MarkovNetwork node' )
                return False

        # Must be an edge cover (for any edge in original_graph, both nodes must live in a node in tree)
        cluster_graph = tree.to_cluster_graph()
        for node1, node2 in tree.edges:
            find_this = set( [ node1, node2 ] )

            for cluster in cluster_graph.nodes:
                if( find_this.issubset( cluster ) == False ):
                    print( 'Not an edge cover' )
                    return False

        # Must satisfy the running intersection property
        if( JunctionTree.satisfies_running_intersection_property( tree ) == False ):
            print( 'Doesn\'t satisfy the rip' )
            return False

        return True

    def inference( self ):
        junction_tree = self.junction_tree()

        message_passing_order = junction_tree.message_passing_order()

        for potential_batch in message_passing_order:

            print()
            for potential_in, potential_out in potential_batch:
                print( potential_in.nodes, potential_out.nodes )