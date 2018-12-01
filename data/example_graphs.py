from host.src.graph import Graph

def polyTree1():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2 ] )
    return graph

def polyTree2():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 4 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 5 ] )
    graph.addEdge( parents=[ 4, 5 ], children=[ 6, 7 ] )

    return graph

def polyTree3():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 3 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 4 ] )

    return graph

def polyTree4():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1, 2 ], children=[ 4, 5 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 6 ] )

    return graph

def polyTree5():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 6, 7 ] )
    graph.addEdge( parents=[ 2 ], children=[ 8 ] )
    graph.addEdge( parents=[ 3, 4, 5 ], children=[ 9, 10, 11 ] )
    graph.addEdge( parents=[ 7, 8 ], children=[ 12 ] )
    graph.addEdge( parents=[ 6, 9 ], children=[ 13 ] )
    graph.addEdge( parents=[ 15, 16 ], children=[ 17 ] )
    graph.addEdge( parents=[ 13, 17 ], children=[ 14 ] )

    return graph

def polyTree6():
    graph = Graph()

    graph.addEdge( parents=[ 0 ], children=[ 1 ] )

    return graph

def polyTree7():
    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2 ] )
    graph.addEdge( parents=[ 2 ], children=[ 3 ] )
    graph.addEdge( parents=[ 3 ], children=[ 4 ] )

    return graph

def polyTree8():
    graph = Graph()

    graph.addEdge( parents=[ 0 ], children=[ 1 ] )
    graph.addEdge( parents=[ 1 ], children=[ 2 ] )
    graph.addEdge( parents=[ 2 ], children=[ 3 ] )
    graph.addEdge( parents=[ 3 ], children=[ 4 ] )
    graph.addEdge( parents=[ 4 ], children=[ 5 ] )
    graph.addEdge( parents=[ 5 ], children=[ 6 ] )
    graph.addEdge( parents=[ 6 ], children=[ 7 ] )
    graph.addEdge( parents=[ 7 ], children=[ 8 ] )
    graph.addEdge( parents=[ 8 ], children=[ 9 ] )
    graph.addEdge( parents=[ 9 ], children=[ 10 ] )
    graph.addEdge( parents=[ 10 ], children=[ 11 ] )
    graph.addEdge( parents=[ 11 ], children=[ 12 ] )
    graph.addEdge( parents=[ 12 ], children=[ 13 ] )
    graph.addEdge( parents=[ 13 ], children=[ 14 ] )
    graph.addEdge( parents=[ 14 ], children=[ 15 ] )
    return graph

def graph1():

    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2, 3 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 4 ] )

    # fbs = np.array( [ 2 ] )

    # return graph, fbs
    return graph

def graph2():

    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 3 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 4 ] )
    graph.addEdge( parents=[ 2, 3, 4 ], children=[ 5, 6 ] )

    # fbs = np.array( [ 2, 4 ] )

    # return graph, fbs
    return graph

def graph3():

    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2, 3, 9 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 4 ] )
    graph.addEdge( parents=[ 1, 2, 3, 8 ], children=[ 5, 6 ] )
    graph.addEdge( parents=[ 1, 2, 4, 5, 6, 9 ], children=[ 7, 10 ] )

    # fbs = np.array( [ 1, 2, 3, 4, 6 ] )

    # return graph, fbs
    return graph

def graph4():

    graph = Graph()

    graph.addEdge( parents=[ 0 ], children=[ 1 ] )
    graph.addEdge( parents=[ 1 ], children=[ 0 ] )

    # fbs = np.array( [ 0 ] )

    assert 0, 'This graph has a feedback cycle'

    # return graph, fbs
    return graph

def graph5():

    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2, 3, 9 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 4 ] )
    graph.addEdge( parents=[ 1, 2, 3, 8 ], children=[ 5, 6 ] )
    graph.addEdge( parents=[ 1, 2, 4, 5, 6, 9 ], children=[ 7, 10 ] )
    graph.addEdge( parents=[ 7, 10 ], children=[ 0 ] )

    # fbs = np.array( [ 0, 1, 2, 3, 4, 6 ] )

    assert 0, 'This graph has a feedback cycle'

    # return graph, fbs
    return graph

def graph6():

    graph = Graph()

    graph.addEdge( parents=[ 0, 1 ], children=[ 2 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 0 ] )

    # fbs = np.array( [ 0 ] )

    assert 0, 'This graph has a feedback cycle'

    # return graph, fbs
    return graph

def graph7():

    graph = Graph()

    graph.addEdge( parents=[ 0 ], children=[ 1, 2 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 3 ] )

    # fbs = np.array( [ 2 ] )

    # return graph, fbs
    return graph

def graph8():

    graph = Graph()

    graph.addEdge( parents=[ 0 ], children=[ 1, 2 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 3 ] )

    # fbs = np.array( [ 2 ] )

    # return graph, fbs
    return graph

def graph9():

    graph = Graph()

    graph.addEdge( parents=[ 7, 8 ], children=[ 0 ] )
    graph.addEdge( parents=[ 0, 1 ], children=[ 3 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 4 ] )
    graph.addEdge( parents=[ 2, 3, 4 ], children=[ 5, 6 ] )

    # fbs = np.array( [ 2, 4 ] )

    # return graph, fbs
    return graph

def graph10():

    graph = Graph()

    graph.addEdge( parents=[ 0 ], children=[ 1, 2 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 3 ] )
    graph.addEdge( parents=[ 3 ], children=[ 4, 5 ] )
    graph.addEdge( parents=[ 4, 5 ], children=[ 6 ] )

    # fbs = np.array( [ 1, 4 ] )

    # return graph, fbs
    return graph

def graph11():
    graph = Graph()
    graph.addEdge( parents=[ 0 ], children=[ 1 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 3, 4, 5 ] )
    graph.addEdge( parents=[ 3, 4 ], children=[ 6 ] )
    graph.addEdge( parents=[ 6 ], children=[ 7 ] )
    graph.addEdge( parents=[ 5, 6 ], children=[ 8 ] )

    # fbs = np.array( [ 3, 5 ] )

    # return graph, fbs
    return graph

def graph12():
    graph = Graph()
    graph.addEdge( parents=[ 0, 1, 2 ], children=[ 6, 7 ] )
    graph.addEdge( parents=[ 3 ], children=[ 8 ] )
    graph.addEdge( parents=[ 4, 5 ], children=[ 9 ] )
    graph.addEdge( parents=[ 5, 6 ], children=[ 10 ] )
    graph.addEdge( parents=[ 6 ], children=[ 11 ] )
    graph.addEdge( parents=[ 7, 8 ], children=[ 12 ] )
    graph.addEdge( parents=[ 11 ], children=[ 13 ] )
    graph.addEdge( parents=[ 11, 12 ], children=[ 14 ] )
    graph.addEdge( parents=[ 9, 10, 11 ], children=[ 15 ] )
    graph.addEdge( parents=[ 13, 14 ], children=[ 16 ] )
    graph.addEdge( parents=[ 8, 14 ], children=[ 17 ] )
    graph.addEdge( parents=[ 15, 16, 17 ], children=[ 18, 19, 20 ] )

    # fbs = np.array( [ 5, 6, 11, 14 ] )

    # return graph, fbs
    return graph

def graph13():
    graph = Graph()
    graph.addEdge( parents=[ 0, 1 ], children=[ 3 ] )
    graph.addEdge( parents=[ 1, 2 ], children=[ 4 ] )
    graph.addEdge( parents=[ 3 ], children=[ 5 ] )
    graph.addEdge( parents=[ 4 ], children=[ 7 ] )
    graph.addEdge( parents=[ 5, 6 ], children=[ 8 ] )
    graph.addEdge( parents=[ 6, 7 ], children=[ 9 ] )

    # fbs = np.array( [ 6 ] )

    # return graph, fbs
    return graph

def graph14():
    graph = Graph()
    graph.addEdge( parents=[ 0 ], children=[ 1, 2 ] )
    graph.addEdge( parents=[ 1 ], children=[ 3 ] )
    graph.addEdge( parents=[ 2 ], children=[ 4 ] )
    graph.addEdge( parents=[ 3, 4 ], children=[ 5 ] )

    # fbs = np.array( [ 1 ] )

    # return graph, fbs
    return graph

def graph15():
    graph = Graph()
    graph.addEdge( parents=[ 0 ], children=[ 1, 2 ] )
    graph.addEdge( parents=[ 1 ], children=[ 3, 4 ] )
    graph.addEdge( parents=[ 2 ], children=[ 5 ] )
    graph.addEdge( parents=[ 3 ], children=[ 6 ] )
    graph.addEdge( parents=[ 4, 6 ], children=[ 7 ] )
    graph.addEdge( parents=[ 4, 5 ], children=[ 8 ] )

    # fbs = np.array( [ 4 ] )

    # return graph, fbs
    return graph

def graph16():
    graph = Graph()
    graph.addEdge( parents=[ 0 ], children=[ 2, 3 ] )
    graph.addEdge( parents=[ 1 ], children=[ 4, 5 ] )
    graph.addEdge( parents=[ 2, 3 ], children=[ 6 ] )
    graph.addEdge( parents=[ 4, 5 ], children=[ 7 ] )
    graph.addEdge( parents=[ 6, 7 ], children=[ 8 ] )
    graph.addEdge( parents=[ 2, 8 ], children=[ 9 ] )
    graph.addEdge( parents=[ 5, 8 ], children=[ 10 ] )

    # fbs = np.array( [ 3, 4, 6, 7 ] )

    # return graph, fbs
    return graph