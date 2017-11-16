from itertools import chain
import unittest

import numpy as np

from bayesian import Domain, Table, Variable
from bayesian.junction import DomainGraph, JunctionTree
import bayesian.tables
from bayesian.tables import Probability


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


class TestJunctionTree(unittest.TestCase):

    def test_from_graph_simple(self):
        """Test the from_graph static method in the simplest case"""

        # Define one table with dependent variables.
        a = Variable('a')
        b = Variable('b')
        table = Table([a, b], [[0.1, 0.2], [0.3, 0.4]])

        junction_trees = JunctionTree.from_tables([table])

        # There should be a single junction tree.
        self.assertEqual(len(junction_trees), 1)
       
        # The junction tree should be a single node with all the variables.
        junction_tree = junction_trees[0]
        nb_nodes = 0
        for node in junction_tree:
            self.assertSetEqual(node.variables,
                                set([a, b]))
            self.assertSetEqual(node.boundary, set())
            self.assertListEqual(node.tables, [table])
            nb_nodes += 1

        self.assertEqual(nb_nodes, 1)

    def test_simplify_simple(self):
        """Test the _simplify method in the simple case"""

        a = Variable('a')
        b = Variable('b')
        table_ab = bayesian.tables.Probability(bayesian.Domain([a, b]), 
                                               [[0.1, 0.2], [0.3, 0.4]])
        table_a = bayesian.tables.Probability(bayesian.Domain([a]), 
                                              [0.1, 0.9])

        junction_trees = JunctionTree.from_tables([table_ab, table_a])
        self.assertEqual(len(junction_trees), 1)
        junction_tree = junction_trees[0]
        junction_tree._simplify()

        # There should be a single table left with the result of the product.
        self.assertEqual(len(junction_tree.tables), 1)
        table = junction_tree.tables[0]
        self.assertFalse(table.updated)

        table.update()
        self.assertTrue(table.updated)
        values = np.array([[0.01, 0.02], [0.27, 0.36]])
        values /= np.sum(values)
        np.testing.assert_array_almost_equal(table.values, values)

    def test_collect_simple(self):
        """Test the _collect method in the simple case"""

        a = Variable('a')
        b = Variable('b')

        table = bayesian.tables.Probability(bayesian.Domain([a, b]), 
                                            [[0.1, 0.2], [0.3, 0.4]])

        parent = JunctionTree(set([b]), set(), [])
        child = JunctionTree(set([a, b]), set([b]), [table])
        parent.add(child)

        parent._collect()

        # The parent has no collect information.
        self.assertIsNone(parent.collect)

        # The child has the table with 'a' marginalized out.
        self.assertIsNotNone(child.collect)
        table = child.collect
        self.assertFalse(table.updated)
        self.assertTupleEqual(table.domain, (b,))

        table.update()
        self.assertTrue(table.updated)
        np.testing.assert_array_almost_equal(table.values, [0.4, 0.6])

    def test_diamond(self):

        symbols = [str(i) for i in range(4)]
        variables = [Variable(s) for s in symbols]
        tables = []
        tables.append(Probability(Domain([variables[0], variables[1]])))
        tables.append(Probability(Domain([variables[0], variables[2]])))
        tables.append(Probability(Domain([variables[1], variables[3]])))
        tables.append(Probability(Domain([variables[2], variables[3]])))

        junction_trees = JunctionTree.from_tables(tables) 

        # There should be a single junction tree.
        self.assertEqual(len(junction_trees), 1)
        junction_tree = junction_trees[0]

        # The junction tree has 2 nodes.
        self.assertEqual(len([n for n in junction_tree]), 2)

        # The head of the tree should contain 3 variables and the boundary 0.
        self.assertEqual(len(junction_tree.variables), 3)
        self.assertEqual(len(junction_tree.boundary), 0)
        self.assertEqual(len(junction_tree.tables), 2)
        
        # The tables should contain only variables that are in the node.
        domains = [t.domain for t in junction_tree.tables]
        self.assertTrue(set(chain(*domains)) == junction_tree.variables)

        # The subtree should contain 3 variables and the boundary 2.
        sub_junction_tree = list(n for n in junction_tree)[1]
        self.assertEqual(len(sub_junction_tree.variables), 3)
        self.assertEqual(len(sub_junction_tree.boundary), 2)
        self.assertEqual(len(sub_junction_tree.tables), 2)

        # The tables should contain only variables that are in the node.
        domains = [t.domain for t in sub_junction_tree.tables]
        self.assertTrue(set(chain(*domains)) == sub_junction_tree.variables)

        # Before simplifying, keep a reference to the old tables.
        junction_tree_tables = junction_tree.tables
        sub_junction_tree_tables = sub_junction_tree.tables

        # Simplifying the tree should reduce the tables to 1 element.
        junction_tree._simplify()
        self.assertEqual(len(junction_tree.tables), 1)
        self.assertTrue(set(junction_tree.tables[0].domain) == junction_tree.variables)
        self.assertEqual(junction_tree.tables[0].left, junction_tree_tables[0])
        self.assertEqual(junction_tree.tables[0].right, junction_tree_tables[1])
        self.assertEqual(len(sub_junction_tree.tables), 1)
        self.assertTrue(set(sub_junction_tree.tables[0].domain) == sub_junction_tree.variables)
        self.assertEqual(sub_junction_tree.tables[0].left, sub_junction_tree_tables[0])
        self.assertEqual(sub_junction_tree.tables[0].right, sub_junction_tree_tables[1])

        # After collecting, the sub junction tree should have a table to
        # share.
        junction_tree._collect()
        self.assertIsNone(junction_tree.collect)
        self.assertIsNotNone(sub_junction_tree.collect) 
        self.assertSetEqual(set(sub_junction_tree.collect.domain), sub_junction_tree.boundary)
        self.assertEqual(sub_junction_tree.collect.table, sub_junction_tree.tables[0]) 

        # After distributing, the sub junction tree should have a table to share.
        junction_tree._distribute()
        self.assertIsNone(junction_tree.distribute)
        self.assertIsNotNone(sub_junction_tree.distribute)
        self.assertSetEqual(set(sub_junction_tree.collect.domain), sub_junction_tree.boundary)
        self.assertEqual(sub_junction_tree.distribute.table, junction_tree.tables[0]) 

        # The marginals should include all tables.
        marginals = []
        visited = set()
        junction_tree._marginals(marginals, visited)



class TestMarginalize(unittest.TestCase):
    """Test the marginalize function"""

    def test_simple_list(self):
        """Test the marginalize function using a list graph"""

        a = Variable('a')
        b = Variable('b')

        table_a = bayesian.tables.Probability(bayesian.Domain([a]),
                                              [0.1, 0.9])
        table_ab = bayesian.tables.Probability(bayesian.Domain([a, b]),
                                               [[0.1, 0.2], [0.3, 0.4]])

        marginals = list(chain(*bayesian.junction.marginalize([table_a, table_ab])))

        # We should have one marginal per variable.
        self.assertEqual(len(marginals), 2)
        variables = [m.domain[0] for m in marginals]
        marginal_a = marginals[variables.index(a)]
        marginal_b = marginals[variables.index(b)]

        # All the marginals should have the same normalisation coefficients.
        for marginal in marginals:
            self.assertAlmostEqual(marginal.normalization, marginals[0].normalization)

        # The marginals should be udpated.
        self.assertTrue(marginal_a.updated)
        values = np.array([0.03, 0.63])
        normalization = values.sum() 
        values /= normalization
        self.assertAlmostEqual(marginal_a.normalization, normalization)
        np.testing.assert_array_almost_equal(marginal_a.values, values)

        self.assertTrue(marginal_b.updated)
        values = np.array([0.28, 0.38])
        normalization = values.sum() 
        values /= normalization
        self.assertAlmostEqual(marginal_b.normalization, normalization)
        np.testing.assert_array_almost_equal(marginal_b.values, values)

    def test_diamond(self):
        """Test the marginalize function using a diamond graph"""

        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        d = Variable('d')

        table_ab = bayesian.tables.Probability(bayesian.Domain([a, b]),
                                               [[0.1, 0.2], [0.3, 0.4]])
        table_ac = bayesian.tables.Probability(bayesian.Domain([a, c]),
                                               [[0.1, 0.2], [0.3, 0.4]])
        table_bd = bayesian.tables.Probability(bayesian.Domain([b, d]),
                                               [[0.1, 0.2], [0.3, 0.4]])
        table_cd = bayesian.tables.Probability(bayesian.Domain([c, d]),
                                               [[0.1, 0.2], [0.3, 0.4]])

        marginals = list(chain(*bayesian.junction.marginalize([table_ab, table_ac, table_bd, table_cd])))

        # One marginal per variable.
        self.assertEqual(len(marginals), 4)
        variables = [m.domain[0] for m in marginals]
        marginal_a = marginals[variables.index(a)]
        marginal_b = marginals[variables.index(b)]
        marginal_c = marginals[variables.index(c)]
        marginal_d = marginals[variables.index(d)]

        # All the marginals should have the same normalisation coefficients.
        for marginal in marginals:
            self.assertAlmostEqual(marginal.normalization, marginals[0].normalization)

        # The marginals should be udpated.
        self.assertTrue(marginal_a.updated)
        values = np.array([
            0.0001 + 0.0004 + 0.0006 + 0.0016 + 0.0006 + 0.0016 + 0.0036 + 0.0064,
            0.0009 + 0.0036 + 0.0036 + 0.0096 + 0.0036 + 0.0096 + 0.0144 + 0.0256])
        normalization = values.sum() 
        values /= normalization
        self.assertAlmostEqual(marginal_a.normalization, normalization)
        np.testing.assert_array_almost_equal(marginal_a.values, values)

        self.assertTrue(marginal_b.updated)
        values = np.array([
            0.0001 + 0.0004 + 0.0006 + 0.0016 + 0.0009 + 0.0036 + 0.0036 + 0.0096,
            0.0006 + 0.0016 + 0.0036 + 0.0064 + 0.0036 + 0.0096 + 0.0144 + 0.0256])
        normalization = values.sum() 
        values /= normalization
        self.assertAlmostEqual(marginal_b.normalization, normalization)
        np.testing.assert_array_almost_equal(marginal_b.values, values)

        self.assertTrue(marginal_c.updated)
        values = np.array([
            0.0001 + 0.0004 + 0.0006 + 0.0016 + 0.0009 + 0.0036 + 0.0036 + 0.0096,
            0.0006 + 0.0016 + 0.0036 + 0.0064 + 0.0036 + 0.0096 + 0.0144 + 0.0256])
        normalization = values.sum() 
        values /= normalization
        self.assertAlmostEqual(marginal_c.normalization, normalization)
        np.testing.assert_array_almost_equal(marginal_c.values, values)

        self.assertTrue(marginal_d.updated)
        values = np.array([
            0.0001 + 0.0006 + 0.0006 + 0.0036 + 0.0009 + 0.0036 + 0.0036 + 0.0144,
            0.0004 + 0.0016 + 0.0016 + 0.0064 + 0.0036 + 0.0096 + 0.0096 + 0.0256])
        normalization = values.sum() 
        values /= normalization
        self.assertAlmostEqual(marginal_d.normalization, normalization)
        np.testing.assert_array_almost_equal(marginal_d.values, values)

    def test_normalization(self):
        """Test that the normalization coefficients are always the same"""

        variable_symbols = 'abcde'
        variables = [bayesian.Variable(s) for s in variable_symbols]

        table_1 = bayesian.tables.Probability(
            bayesian.Domain(variables[:-1]),
            np.random.rand(2, 2, 2, 2))

        table_2 = bayesian.tables.Probability(
            bayesian.Domain(variables[1:]),
            np.random.rand(2, 2, 2, 2))

        table_a = bayesian.tables.Probability(
            bayesian.Domain([variables[0]]),
            [0.5, 0.5])
        table_e = bayesian.tables.Probability(
            bayesian.Domain([variables[-1]]),
            [0.5, 0.5])

        marginals = list(chain(*bayesian.junction.marginalize(
            [table_1, table_2, table_a, table_e])))

        for _ in range(100):

            table_a.values = np.random.rand(2)
            table_e.values = np.random.rand(2)
    
            for marginal in marginals:
                marginal.update()

            for marginal in marginals:
                self.assertAlmostEqual(marginals[0].normalization,
                                       marginal.normalization)
