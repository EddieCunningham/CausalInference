import networkx
import numpy as np

class _AlgorithmMixin():

    def __init__( self ):
        """ Algorithm mixin to do stuff

        Args:
            None

        Returns:
            None
        """
        super().__init__()
        self.feedback_set = []

    def findCutSet( self ):
        pass

    @staticmethod
    def combineCutSets( graphs ):
        cut_sets = [ np.array( graph.findCutSet() ) for graph in graphs ]
        reindexed_cut_sets = []
        total_nodes = 0

        for i, graph in enumerate( graphs ):
            reindexed_cut_sets.append( graph.findCutSet() + total_nodes )
            total_nodes += len( graph.nodes )

        return np.hstack( reindexed_cut_sets )