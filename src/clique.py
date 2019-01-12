import networkx as nx
from networkx.algorithms import approximation
import itertools
import numpy as np

class Clique():

    def __init__( self, nodes ):
        """ Nodes must be hashable

        Args:
            nodes - An iterable of hashed nodes

        Returns:
            None
        """
        self.nodes = sorted( nodes, key=lambda x: hash( x ) )
        self.hashes = np.array( [ hash( node ) for node in self.nodes ] )
        self.hash = hash( tuple( self.hashes ) )

    def __hash__( self ):
        """ A simple way to hash nodes.

        Args:
            nodes - Must be an iterable of nodes

        Returns:
            hash - The has for nodes
        """
        return self.hash

    def __str__( self ):
        return str( self.nodes )

    def __repr__( self ):
        return str( self )

    def is_subset( self, other_clique ):
        """ Check if this clique is a subset of the other clique

        Args:
            other_clique - The potential sepset of this clique

        Returns:
            Whether or not this clique is contained in other_clique
        """
        assert isinstance( other_clique, Clique )
        # Can optimize this later
        return set( self.hashes ).issubset( other_clique.hashes )

    def intersection( self, other_clique ):
        """ Check if this node shared nodes with other_clique

        Args:
            other_clique - The potential sepset of this clique

        Returns:
            Whether or not the cliques share nodes
        """
        assert isinstance( other_clique, Clique )
        intersection_indices = np.in1d( self.hashes, other_clique.hashes )
        return [ self.nodes[i] for i in intersection_indices ]

    @staticmethod
    def edges_for( nodes ):
        """ Return all of the edges between nodes

        Args:
            nodes - Nodes that make up clique

        Returns:
            edges - A list of ( node1, node2 ) for each edge
        """
        return list( itertools.combinations( nodes, 2 ) )

    @property
    def edges( self ):
        return Clique.edges_for( self.nodes )

    @property
    def number_of_edges( self ):
        return len( self.nodes ) * ( len( self.nodes ) - 1 ) / 2

    def to_graph( self ):
        """ Return a complete graph that represents this clique

        Args:
            None

        Returns:
            complete_graph - The networkx graph of this
        """
        complete_graph = nx.Graph()
        complete_graph.add_edges_from( self.edges )
        return complete_graph

    def potential( self ):
        """ Return some potential over this clique

        Args:
            None

        Returns:
            potential over clique
        """
        assert 0, 'Implement this in a subclass if you want to do inference'

    def potential_complexity( self ):
        """ Evaluates how expensive it is to compute the potential for max_clique

        Args:
            max_clique - The max_clique

        Returns:
            None
        """
        assert 0, 'Implement this in a subclass if you want to do inference'
