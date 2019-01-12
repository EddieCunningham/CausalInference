from compiled.message_passing import *
from host.data.example_graphs import *

def getCliquePotential( clique ):
    pass

def goalUseCase():

    graph = randomGraph()
    clique_tree = graph.toCliqueTree()

    for clique in clique_tree:
        clique.setPotential( getCliquePotential( clique ) )

    message_passing_iterator = graph.messagePassing()

    for clique_batch in message_passing_iterator:
        pass
