import networkx as nx
from networkx.algorithms import approximation
import itertools
import numpy as np
from collections import OrderedDict

__all__ = [ 'Clique', 'DiscreteClique' ]

class Clique():

    def __init__( self, nodes, computation_type=None ):
        """ Nodes must be hashable and the hash must stay the same for the
            lifetime of this object!!!!!!

        Args:
            nodes            - An iterable of hashed nodes
            computation_type - What kind of computation goes into evaluating
                               the potential.  Useful for determining batches

        Returns:
            None
        """
        self.nodes  = sorted( nodes, key=lambda x: hash( x ) )
        self.hash_map = dict( [ ( hash( node ), i ) for i, node in enumerate( self.nodes ) ] )
        self.hash_keys = set( [ hash( node ) for node in self.nodes ] )
        # self.hashes = np.array( [ hash( node ) for node in self.nodes ] )
        self.hash   = hash( tuple( self.nodes ) )
        self.computation_type = computation_type
        self.potential = None

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

    def __eq__( self, other ):
        if( isinstance( other, Clique ) ):
            return other.hash == self.hash
        return self.hash == Clique( other ).hash

    def is_subset( self, other_clique ):
        """ Check if this clique is a subset of the other clique

        Args:
            other_clique - The potential sepset of this clique

        Returns:
            Whether or not this clique is contained in other_clique
        """
        assert isinstance( other_clique, Clique )
        return self.hash_keys.issubset( other_clique.hash_keys )

    def intersection( self, other_clique ):
        """ Nodes shared with other_clique

        Args:
            other_clique - The potential sepset of this clique

        Returns:
            Whether or not the cliques share nodes
        """
        assert isinstance( other_clique, Clique )

        intersection = self.hash_keys.intersection( other_clique.hash_keys )
        return [ self.nodes[self.hash_map[h]] for h in intersection ]

    def union( self, other_clique ):
        """ Nodes shared with other_clique

        Args:
            other_clique - The potential sepset of this clique

        Returns:
            Whether or not the cliques share nodes
        """
        assert isinstance( other_clique, Clique )

        union = self.hash_keys.union( other_clique.hash_keys )
        return [ self.nodes[self.hash_map[h]] if h in self.hash_map else other_clique.nodes[other_clique.hash_map[h]] for h in union ]

    def difference( self, other_clique ):
        """ The nodes in this clique that are not in other_clique

        Args:
            other_clique - The potential sepset of this clique

        Returns:
            The nodes in this clique that are not in other_clique
        """
        assert isinstance( other_clique, Clique )

        difference = self.hash_keys.difference( other_clique.hash_keys )
        return [ self.nodes[self.hash_map[h]] for h in difference ]

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

    def set_potential( self, potential ):
        """ Set the potential function over the nodes in this clique

        Args:
            potential - Some potential over all the nodes

        Returns:
            potential over clique
        """
        self.potential = potential

    def potential_complexity( self ):
        """ Evaluates how expensive it is to compute the potential for max_clique

        Args:
            max_clique - The max_clique

        Returns:
            None
        """
        assert 0, 'Implement this in a subclass if you want to do inference'

##############################################################################################################

class DiscreteClique( Clique ):

    def __init__( self, nodes, state_sizes ):
        """ A clique in a discrete network.  Pass in the state sizes

        Args:
            nodes            - An iterable of hashed nodes
            state_sizes      - The number of values that each node can take

        Returns:
            None
        """
        super().__init__( nodes )
        self.state_sizes = np.array( [ state_sizes[self.nodes.index( node )] for node in nodes ] )
        self.computation_type = tuple( self.state_sizes )

    def set_potential( self, potential ):
        """ Set the potential function over the nodes in this clique

        Args:
            potential - Some potential over all the nodes.  Must be a numpy array

        Returns:
            None
        """
        assert isinstance( potential, np.ndarray )
        self.potential = potential

    def potential_complexity( self ):
        """ Evaluates how expensive it is to compute the potential for max_clique.
            This is going to be the number of elements in the potential for this clique

        Args:
            None

        Returns:
            None
        """
        return np.prod( np.array( self.potential.shape ) )
