import networkx as nx
import numpy as np
from .draw import _DrawingMixin

class BayesianNetwork( nx.DiGraph ):

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

    def draw( self,
              output_folder='.',
              output_name='graph',
              file_format='png' ):
        graph = nx.nx_agraph.to_agraph( self )

        output_folder = output_folder if output_folder[-1] != '/' else output_folder[:-1]
        output_name = output_name if '/' not in output_name else output_name.replace( '/', '' )
        file_format = file_format.replace( '.', '' )

        graph.draw( '%s/%s.%s'%( output_folder, output_name, file_format ), prog='dot' )