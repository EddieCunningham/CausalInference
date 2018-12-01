import graphviz

class _DrawingMixin():
    """ This class is meant to be a mixin to the main Graph class in graph.py
    """

    @property
    def edge_style( self ):
        """ Hyper edge style
        """
        if( hasattr( self, '_edge_style' ) == False ):
            self.edge_style = dict( label='',
                                    width='0.25',
                                    height='0.25',
                                    fontcolor='white',
                                    style='filled',
                                    fillcolor='black',
                                    fixedsize='true',
                                    fontsize='6' )
        return self._edge_style

    @edge_style.setter
    def edge_style( self, val ):
        self._edge_style = val

    ######################################################################

    @property
    def node_style( self ):
        """ Node style
        """
        if( hasattr( self, '_node_style' ) == False ):
            self.node_style = dict( label='', fixedsize='true' )
        return self._node_style

    @node_style.setter
    def node_style( self, val ):
        self._node_style = val

    ######################################################################

    @property
    def highlight_node_style( self ):
        """ Highlighted node style
        """
        if( hasattr( self, '_highlight_edge_style' ) == False ):
            self.highlight_node_style = dict( label='',
                                              fontcolor='white',
                                              style='filled',
                                              fillcolor='blue' )
        return self._highlight_edge_style

    @highlight_node_style.setter
    def highlight_node_style( self, val ):
        self._highlight_edge_style = val

    ######################################################################

    def draw( self,
              render=True,
              horizontal=False,
              labels=True,
              styles={},
              node_to_style_key={},
              edge_to_style_key={},
              file_format='png',
              output_folder='.',
              output_name='graph' ):
        """ Draw this graph

        Args:
            render            : Render this graph as a <file_format>?
            horizontal        : Draw the graph horizontally
            labels            : Include the node and edge numbers
            styles            : Style dict for each node and edge
            node_to_style_key : Map between node and style choice
            edge_to_style_key : Map between edge and style choice
            file_format       : What file format to render as
            output_folder     : Where to render to
            output_name       : Render file name

        Returns:
            Graphviz object
        """
        getEdgeStyle = lambda ne: styles[ edge_to_style_key[ ne ] ] if ne in edge_to_style_key else self.node_style

        d = graphviz.Digraph( format=file_format, filename=output_name, directory=output_folder )
        if( horizontal == True ):
            d.attr( rankdir='LR' )

        for e, ( parents, children ) in enumerate( zip( self.edge_parents, self.edge_children ) ):
            for p in parents:
                d.edge( '%d '%( p ), '%d'%( e ), **getEdgeStyle( ( p, e ) ) )
            for c in children:
                d.edge( '%d'%( e ), '%d '%( c ), **getEdgeStyle( ( e, c ) ) )

            if( labels == True ):
                d.node( '%d'%( e ), **self.edge_style )
            else:
                d.node( '', **self.edge_style )

        for n, style_key in node_to_style_key.items():
            if( labels == True ):
                d.node( '%d '%( n ), **styles[ style_key ] )
            else:
                d.node( '', **styles[ style_key ] )

        if( render ):
            d.render( cleanup=True )

        return d
