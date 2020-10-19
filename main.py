import sys

from graph import Graph


def run(*args):
    print(*args)
    graph = Graph(args[0][1])
    graph.read_graph()
    graph.get_paths()


if __name__ == '__main__':
    run(sys.argv)
