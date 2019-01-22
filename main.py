import numpy as np
from host.src.tests.markov_network_tests import *
# from host.src.tests.traversal_tests import *
# from host.src.tests.inference_tests import *
# from host.src.tests.message_passing_tests import *

allMarkovNetworkTests()
# allMessagePassingTests()
# allTraversalTests()
# allInferenceTests()

# from host.data.networkx_generator_examples import draw_all
# draw_all()

# from host.src.red_black_tree import *

# data_list = [ 0, 0, 2, 8, 4, 7, 3, 7, 1, 0, 3, 4.1, 4.1, 5.1, 7, 2, 8 ]
# data = [ NodeData( d ) for d in data_list ]

# rb_tree = RedBlackTree()

# for d in data:
#     rb_tree.add( d )


# samples = np.array( [ rb_tree.sample().data for _ in range( 1000 ) ] )
# samples.sort()

# print( samples )

# print( 'root', rb_tree.root )
# for node, left, right in rb_tree:
#     print( '{%s,%d} | {%s,%d}, {%s,%d}'%( node.value, node.total_value, left.value, left.total_value, right.value, right.total_value ) )
    # print()
    # print( node.value, left.value, right.value )
    # print( node.left_value_sum, node.right_value_sum )
