import numpy as np
import itertools
from .draw import _DrawingMixin
from collections import deque

__all__ = [ 'Graph' ]

class Edges():
    def __init__( self ):
        self.parent_edge = None
        self.child_edges = []

class _PolytreeBaseMixin():

    def __init__( self ):
        """ This is the base class for polytrees

        Args:
            None

        Returns:
            None
        """
        self.nodes = set()
        self.edge_children = list()
        self.edge_parents = list()
        self.modified = True

        self.evidence = {}
        self.bayesian_intervention = {}
        self.causal_intervention = {}
        self.guess_intervention = {}

    ######################################################################

    def addEdge( self, parents, children ):
        """ Main method to build a graph.  Parents and children can
            be anything, but must be hashable.  It is important to note
            that the order of the parents used in any subsequent algorithm
            is the same as how the parents were passed in here!

        Args:
            None

        Returns:
            None
        """
        assert isinstance( parents, list ) or isinstance( parents, tuple )
        assert isinstance( children, list ) or isinstance( children, tuple )
        for node in parents + children:
            self.nodes.add( node )

        self.edge_children.append( children )
        self.edge_parents.append( parents )
        self.modified = True

    ######################################################################

    @property
    def roots( self ):
        """ Convinient way to access the roots of the graph.  Auto updated
            as needed
        """
        if( hasattr( self, '_roots' ) == False or self.modified ):
            self._roots = self.nodes - set( list( itertools.chain( *self.edge_children ) ) )
            self._leaves = self.nodes - set( list( itertools.chain( *self.edge_parents ) ) )
            self.modified = False
        return self._roots

    @property
    def leaves( self ):
        """ Convinient way to access the leaves of the graph.  Auto updated
            as needed
        """
        if( hasattr( self, '_leaves' ) == False or self.modified ):
            self._roots = self.nodes - set( list( itertools.chain( *self.edge_children ) ) )
            self._leaves = self.nodes - set( list( itertools.chain( *self.edge_parents ) ) )
            self.modified = False
        return self._leaves

    @property
    def tree( self ):
        """ A tree of this data structure
        """
        if( hasattr( self, '_tree' ) == False or self.modified ):
            self._tree = {}
            for e, ( parents, children ) in enumerate( zip( self.edge_parents, self.edge_children ) ):
                for parent in parents:
                    if( parent not in self._tree ):
                        self._tree[parent] = Edges()
                    if( e not in self._tree[parent].child_edges ):
                        self._tree[parent].child_edges.append( e )

                for child in children:
                    if( child not in self._tree ):
                        self._tree[child] = Edges()
                    if( self._tree[child].parent_edge is not None ):
                        assert self._tree[child].parent_edge == e
                    else:
                        self._tree[child].parent_edge = e

        return self._tree

    ######################################################################

    def getParents( self, node ):
        """ Get the parents for node

        Args:
            node : Query node

        Returns:
            list : A list containing the parents of node
        """
        parent_edge = self.tree[node].parent_edge
        return self.edge_parents[parent_edge] if parent_edge is not None else []

    def getChildren( self, node ):
        """ Get the children for node

        Args:
            node : Query node

        Returns:
            list : A nested list containing the children of node at each child edge
        """
        ans = []
        for child_edge in self.tree[node].child_edges:
            ans.append( self.edge_children[child_edge] )
        return ans

    ######################################################################

    def forwardPass( self ):
        """ Generator function that performs a bredth first search
            of the graph

        Args:
            None

        Returns:
            Each node, visited when its parents are visited
        """
        edge_semaphores = np.array( [ len( e ) for e in self.edge_parents ] )

        # Get the first edges to start with
        for edge, parents in enumerate( self.edge_parents ):
            edge_semaphores[edge] -= len( set.intersection( self.roots, set( parents ) ) )

        for root in self.roots:
            yield root

        edges = np.arange( edge_semaphores.shape[ 0 ], dtype=int )

        done_edges = edge_semaphores == 0
        q = deque( edges[done_edges] )
        while( len( q ) > 0 ):

            edge = q.popleft()
            for child in self.edge_children[edge]:
                yield child
                for child_edge in self.tree[child].child_edges:
                    edge_semaphores[child_edge] -= 1

            now_done = ( edge_semaphores == 0 ) & ( ~done_edges )
            q.extend( edges[now_done] )
            done_edges |= now_done

    def backwardPass( self ):
        """ Generator function that performs a reversed bredth first search
            of the graph

        Args:
            None

        Returns:
            Each node, visited when its children are visited
        """
        edge_semaphores = np.array( [ len( e ) for e in self.edge_children ] )

        # Get the first edges to start with
        for edge, children in enumerate( self.edge_children ):
            edge_semaphores[edge] -= len( set.intersection( self.leaves, set( children ) ) )

        for leaf in self.leaves:
            yield leaf

        edges = np.arange( edge_semaphores.shape[0], dtype=int )

        done_edges = edge_semaphores == 0
        q = deque( edges[done_edges] )
        while( len( q ) > 0 ):

            edge = q.popleft()
            for parent in self.edge_parents[edge]:
                yield parent
                if( self.tree[parent].parent_edge is not None ):
                    edge_semaphores[self.tree[parent].parent_edge] -= 1

            now_done = ( edge_semaphores == 0 ) & ( ~done_edges )
            q.extend( edges[now_done] )
            done_edges |= now_done

    ######################################################################

    def toSparse( self ):
        """ Converts the graph into its sparse representation.  It
            uniquely identifies edges and nodes and then holds arrays
            of parents for edges and children for edges.
            The return shape will be [ 2, : ]

        Args:
            None

        Returns:
            edge_parents_sparse  : Parents for edges
            edge_children_sparse : Children for edges
        """
        nodes = list( self.nodes )

        edge_parents_sparse, edge_children_sparse = [], []

        # Create the child edges
        for i, node_list in enumerate( self.edge_parents ):
            for j, node in enumerate( node_list ):
                edge_parents_sparse.append( [ i, nodes.index( node ) ] )

        # Create the parent edges
        for i, node_list in enumerate( self.edge_children ):
            for j, node in enumerate( node_list ):
                edge_children_sparse.append( [ i, nodes.index( node ) ] )

        return np.array( edge_parents_sparse ).T, np.array( edge_children_sparse ).T

    @staticmethod
    def combineSparse( sparse_graphs ):
        """ Combines sparse graphs into one big, unconnected graph

        Args:
            sparse_graphs : A list of sparse graphs

        Returns:
            edge_parents_sparse  : Parents for edges
            edge_children_sparse : Children for edges
        """
        edge_parents, edge_children = [], []
        total_edges, total_nodes = 0, 0
        for ep, ec in sparse_graphs:
            # See how many nodes and edges are in this sparse graph
            n_edges = max( ep[0, -1], ec[0, -1] ) + 1
            n_nodes = max( np.max( ep[1, :] ), np.max( ec[1, :] ) ) + 1

            # Adjust their indices
            ep[0, :] += total_edges
            ec[0, :] += total_edges
            ep[1, :] += total_nodes
            ec[1, :] += total_nodes

            # Increment the number of nodes and edges
            total_edges += n_edges
            total_nodes += n_nodes

            # Add the to the graph
            edge_parents.append( ep )
            edge_children.append( ec )

        # Concatenate the arrays
        edge_parents = np.hstack( edge_parents )
        edge_children = np.hstack( edge_children )
        return edge_parents, edge_children

    @staticmethod
    def fromSparse( edge_parents_sparse, edge_children_sparse ):
        """ Turn sparse format into graph

        Args:
            edge_parents_sparse  : Parents for edges
            edge_children_sparse : Children for edges

        Returns:
            graph : The graph
        """
        edges = {}

        for e, parent in edge_parents_sparse.T:
            if( e not in edges ):
                edges[e] = [ [ parent ], [] ]
            else:
                edges[e][0].append( parent )

        for e, child in edge_children_sparse.T:
            edges[e][1].append( child )

        graph = Graph()
        for e in sorted( edges.keys() ):
            graph.addEdge( edges[e][0], edges[e][1] )

        return graph

##########################################################################

class Graph( _PolytreeBaseMixin, _DrawingMixin ):
    pass