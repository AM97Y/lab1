import sys

from graph import Graph
from binary import Binary


def run(*args):
    print(*args)
    graph = Graph(args[0][1])
    graph.read_graph()
    graph.get_paths()

    """graph = Binary(args[0][1])
    graph.read_graph()
    graph.boruvka_algorithm
"""

if __name__ == '__main__':
    run(sys.argv)
