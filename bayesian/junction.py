from itertools import combinations

from recur.graphs import Undirected as Node


class DomainGraph(Node):

    def __init__(self, variable):
        """Node of a domain graph which represents a variable"""

        super().__init__()
        self.variable = variable

    @staticmethod
    def from_tables(tables):
        """Builds domain graphs from a list of tables"""

        # Get the unique variables of the graph and create a node for
        # each of them.
        variables = list(set(v for t in tables for v in t.domain))
        nodes = [DomainGraph(v) for v in variables]

        # Add the link. Because the nodes are Undirected, every add creates
        # a symmetric link.
        for table in tables:
            for n1, n2 in combinations(table.domain, 2):
                nodes[variables.index(n1)].add(nodes[variables.index(n2)])

        # It is possible for tables to generate disconnected graphs. Here we
        # select one node for every one.
        graphs = []
        while len(nodes) > 0:
            graphs.append(nodes[0])
            for node in graphs[-1]:
                nodes.remove(node)

        return graphs
