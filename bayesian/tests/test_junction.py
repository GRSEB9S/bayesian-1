import unittest

from bayesian import Domain, Table, Variable
from bayesian.junction import DomainGraph


class TestDomainGraph(unittest.TestCase):
    """Test the bayesian.junction.DomainGraph class"""

    def test_from_tables_simple(self):
        """Test the from_Tables static method"""

        # Define one table with dependent variables.
        a = Variable('a')
        b = Variable('b')
        table = Table([a, b], [[0.1, 0.2], [0.3, 0.4]])

        # The graph should be two linked nodes.
        graphs = DomainGraph.from_tables([table])
        self.assertEqual(len(graphs), 1)
        n = 0
        for node in graphs[0]:
            n += 1
        self.assertEqual(n, 2)

    def test_from_tables_triangle_plus_one(self):
        """Test the from_tables using a triangle graph and isolated node"""

        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        d = Variable('d')

        table_a = Table(Domain([a, b]))
        table_b = Table(Domain([c, b]))
        table_c = Table(Domain([a, c]))
        table_d = Table(Domain([d]))

        # The graph should be a triangle and one isolated node"""
        graphs = DomainGraph.from_tables([table_a, table_b, table_c, table_d])

        self.assertEqual(len(graphs), 2)
        if graphs[0].is_isolated:
            triangle = graphs[1]
        elif graphs[1].is_isolated:
            triangle = graphs[0]
        else:
            self.assertTrue(False)

        self.assertTrue(triangle.is_simplicial)

    def test_from_tables_empty(self):
        """Test the from_tables static method using an empty graph"""

        graphs = DomainGraph.from_tables([])
        self.assertEqual(len(graphs), 0)

    def test_from_tables_disconnected_graphs(self):
        """Test the from_tables static method"""

        # Define two tables with independent variables.
        a = Variable('a')
        table_a = Table([a], [0.1, 0.9])
        b = Variable('b')
        table_b = Table([b], [0.2, 0.8])

        # The graph for a single table should be a single isolated node.
        graphs = DomainGraph.from_tables([table_a])
        self.assertEqual(len(graphs), 1)
        self.assertTrue(graphs[0].is_isolated)

        # The graph for the two tables should be two isolated nodes.
        graphs = DomainGraph.from_tables([table_a, table_b])
        self.assertEqual(len(graphs), 2)
        self.assertTrue(graphs[0].is_isolated)
        self.assertTrue(graphs[1].is_isolated)
