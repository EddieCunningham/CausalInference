import networkx as nx
import numpy as np
from .markov_network import MarkovNetwork
from host.src.clique import Clique
import itertools
from collections import namedtuple

class JunctionTree( MarkovNetwork ):

    MessageInstruction = namedtuple( 'MessageInstruction', [ 'message', 'incoming_messages', 'separator_nodes' ] )

    @property
    def super_nodes( self ):
        return self.nodes

    @property
    def leaves( self ):
        return [ node for node in self.nodes if len( list( self.neighbors( node ) ) ) == 1 ]

    @staticmethod
    def example_junction_tree():
        A = Clique( [ 1, 2, 3 ] )
        B = Clique( [ 2, 3 ] )
        C = Clique( [ 2, 3, 6 ] )
        D = Clique( [ 6, 7 ] )
        E = Clique( [ 3, 6, 7 ] )
        F = Clique( [ 6, 7, 8 ] )
        G = Clique( [ 3, 4, 5 ] )
        H = Clique( [ 4, 5, 9 ] )
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
        A = Clique( [ 1, 2, 3 ] )
        B = Clique( [ 2, 3 ] )
        C = Clique( [ 2, 3, 6 ] )
        D = Clique( [ 6, 7 ] )
        E = Clique( [ 3, 6, 7 ] )
        F = Clique( [ 6, 7, 8 ] )
        G = Clique( [ 3, 4, 6 ] )
        H = Clique( [ 4, 5, 9 ] )
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

    def has_running_intersection_property( self ):
        """ Check the running intersection property.

        Args:
            None

        Returns:
            Whether or not this cluser graph satisfies the rip
        """

        # Start from every leaf and go inwards.
        message_passing_order = self.message_passing_order( no_batches=True )

        union_at_nodes = {}
        visited_pairs = set()
        for super_node, adjacent_super_node in message_passing_order:

            # Don't want to repeat - only want to start at leaves and go inward.
            pair = tuple( sorted( [ super_node, adjacent_super_node ], key=lambda x: hash( x ) ) )
            if( pair in visited_pairs ):
                continue
            visited_pairs.add( pair )

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

        # Every node must be a clique object
        for cluster in tree.nodes:
            if( isinstance( cluster, Clique ) == False ):
                print( 'Doesn\'t have a Clique node' )
                return False

        # Must be an edge cover (for any edge in original_graph, both nodes must live in a node in tree)
        for node1, node2 in original_graph.edges:
            find_this = Clique( [ node1, node2 ] )

            found = False
            for cluster in tree.nodes:
                if( find_this.is_subset( cluster ) == True ):
                    found = True
            if( found == False ):
                print( 'Not an edge cover' )
                return False

        # Must satisfy the running intersection property
        if( tree.has_running_intersection_property() == False ):
            print( 'Doesn\'t satisfy the rip' )
            return False

        return True

    def is_consistent( self ):
        """ Check to see if inference worked.  This is done by seeing if
            pairs of nodes at each edge marginalize to the same thing.
            Because of the RIP, locally consistency implies global consistency

        Args:
            None

        Returns:
            Whether or not tree is consistent
        """
        pass

    def shafer_shenoy_inference_instructions( self ):
        """ Build the set of instructions of how to do inference.  This is useful so
            that we can save off the instructions for any particular graph.
            This function will return the instructions needed to do shafer shenoy
            message passing.  Here we have messages for the separator potentials.

        Args:
            None

        Returns:
            instructions - The instructions on how to perform inference
        """
        message_instructions = []

        message_passing_order = self.message_passing_order()

        # Loop over every message batch
        for potential_batch in message_passing_order:

            # Populate the batches with the messages that we should combine
            batch = []
            for potential_in, potential_out in potential_batch:

                nodes_in        = tuple( potential_in.nodes )
                nodes_out       = tuple( potential_out.nodes )
                separator_nodes = tuple( potential_in.intersection( potential_out ) )

                # Each edge represents a message
                message           = MarkovNetwork.Message( nodes_in, nodes_out )
                incoming_messages = [ MarkovNetwork.Message( tuple( neighbor.nodes ), nodes_in ) for neighbor in self.neighbors( nodes_in ) if neighbor != nodes_out ]

                batch.append( JunctionTree.MessageInstruction( message, incoming_messages, separator_nodes ) )

            message_instructions.append( batch )

        # At the end we want to compute the smoothed clique potentials.  This can be done asynchronously
        batch = []
        for clique in self.nodes:

            nodes             = tuple( clique.nodes )
            message           = MarkovNetwork.Message( nodes, nodes )
            incoming_messages = [ MarkovNetwork.Message( tuple( neighbor.nodes ), nodes ) for neighbor in self.neighbors( nodes ) ]
            separator_nodes   = nodes

            batch.append( JunctionTree.MessageInstruction( message, incoming_messages, separator_nodes ) )
        message_instructions.append( batch )

        return message_instructions

    def recombination_instructions( self ):
        """ How to combine messages to

        Args:
            None

        Returns:
            instructions - The instructions on how to perform inference
        """
        pass