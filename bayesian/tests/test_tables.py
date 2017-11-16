from functools import reduce
from itertools import combinations
import unittest

import numpy as np

import  bayesian
import  bayesian.tables
from bayesian import Domain, Variable 
from bayesian.tables import Probability, Product, Marginal


class TestMarginal(unittest.TestCase):
    """Test the Marginal class"""

    def test_update_simple(self):
        """Test the udpate method in the simple case"""

        a = bayesian.Variable('a') 
        b = bayesian.Variable('b') 

        table = bayesian.tables.Probability(bayesian.Domain([a, b]),
                                              [[0.1, 0.9], [0.4, 0.6]])

        result = bayesian.tables.Marginal(table, a)
        self.assertFalse(result.updated)

        result.update()
        self.assertTrue(result.updated)
        np.testing.assert_array_almost_equal(result.values, [0.25, 0.75])

        table.values = [[0.2, 0.8], [0.4, 0.6]]
        self.assertFalse(result.updated)

        result.update()
        self.assertTrue(result.updated)
        np.testing.assert_array_almost_equal(result.values, [0.3, 0.7])


class TestProduct(unittest.TestCase):
    """Test the Product class"""

    def test_update_simple(self):
        """Test the update method in the simple case"""

        a = bayesian.Variable('a') 
        b = bayesian.Variable('b') 

        table_a = bayesian.tables.Probability(bayesian.Domain([a]), [0.1, 0.9])
        table_b = bayesian.tables.Probability(bayesian.Domain([b]), [0.4, 0.6])

        result = bayesian.tables.Product(table_a, table_b)
        self.assertFalse(result.updated)

        result.update()
        self.assertTrue(result.updated)
        np.testing.assert_array_almost_equal(result.values, [[0.04, 0.06], [0.36, 0.54]])

        table_a.values = [0.2, 0.8]
        self.assertFalse(result.updated)

        result.update()
        self.assertTrue(result.updated)
        np.testing.assert_array_almost_equal(result.values, [[0.08, 0.12], [0.32, 0.48]])


class TestUpdateGraph(unittest.TestCase):
    """Test the update graph used to update marginals"""

    def test_update_order(self):
        """Test the update order by computing a marginal with different trees

        The order of the operations should not affect the values of the
        marginals. This test computes a marginal using 2 different update
        trees and verifies that the result is the same.

        """

        a = bayesian.Variable('a') 
        b = bayesian.Variable('b') 
        c = bayesian.Variable('c') 

        table_a = Probability(Domain([a]), [0.1, 0.9])

        table_ba = Probability(Domain([b, a]), [[0.1, 0.2], [0.3, 0.4]])
        table_bca = Probability(Domain([b, c, a]), [
            [[0.05, 0.10], [0.15, 0.20]],
            [[0.20, 0.15], [0.10, 0.05]]
        ])

        # Compute the marginal by computing all products first.
        product_abc = reduce(Product, [table_a, table_ba, table_bca])
        product_abc.update()
        marginal_1 = product_abc.marginalize(c)

        # Compute the marginal by interweaving products and marginals.
        product_bac = reduce(Product, [table_ba, table_bca])
        product_bac.update()
        temp_marginal = product_bac.marginalize(c)
        marginal_2 = Product(table_a, temp_marginal)
        marginal_2.update()

        np.testing.assert_array_almost_equal(
            marginal_1.values.flat,
            marginal_2.values.flat[bayesian.map(marginal_1.domain, marginal_2.domain)])


    def test_graph_pair(self):
        """Test using 2 graphs joined by a single variable

        This test builds a graph that contains 2 subgraph that are joined
        by a single variable. The marginals for each side requires the
        the marginal of the joining variable for the other side.

        """
        N = 8
        joining_variable = Variable(str(2 * N))

        # Build the left and right trees. Each of them only share a single
        # variable.
        left_symbols = [str(i) for i in range(N)]
        left_variables = [Variable(symbol) for symbol in left_symbols]
        left_tables = [Probability(Domain([v])) for v in left_variables]
        left_joining = Probability(Domain(left_variables + [joining_variable]),
                                   np.random.rand(*((2,) * (N + 1))))

        right_symbols = [str(i + N) for i in range(N)]
        right_variables = [Variable(symbol) for symbol in right_symbols]
        right_tables = [Probability(Domain([v])) for v in right_variables]
        right_joining = Probability(Domain(right_variables + [joining_variable]),
                                    np.random.rand(*((2,) * (N + 1))))

        # Compute the joint for each subgraph.
        left_joint = reduce(Product, left_tables)
        left_joint = Product(left_joint, left_joining)

        right_joint = reduce(Product, right_tables)
        right_joint = Product(right_joint, right_joining)

        # To compute the marginals of the variable on the right side, we
        # need the marginal for the joining variable on the left side.
        left_joining_marginal = Marginal(left_joint, left_variables[0])
        for variable in left_variables[1:]:
            left_joining_marginal = Marginal(left_joining_marginal, variable)
       
        # The result should be a marginal over the joining variable.
        self.assertEqual(len(left_joining_marginal.domain), 1)
        self.assertEqual(left_joining_marginal.domain[0], joining_variable)

        right_full_joint = Product(right_joint, left_joining_marginal)

        # Compute all the marginals for the right side by marginalizing all
        # variables except one.
        marginals = []
        for variable in right_variables:
            marginal = right_full_joint
            for to_marginalize in right_full_joint.domain - variable:
                marginal = Marginal(marginal, to_marginalize)
            marginals.append(marginal)

        # To compute the marginals of the variable on the left side, we
        # need the marginal for the joining variable on the right side.
        right_joining_marginal = Marginal(right_joint, right_variables[0])
        for variable in right_variables[1:]:
            right_joining_marginal = Marginal(right_joining_marginal, variable)

        # The result should be a marginal over the joining variable.
        self.assertEqual(len(right_joining_marginal.domain), 1)
        self.assertEqual(right_joining_marginal.domain[0], joining_variable)

        left_full_joint = Product(left_joint, right_joining_marginal)

        # Compute all the marginals for the left side by marginalizing all
        # variables except one.
        for variable in left_variables:
            marginal = left_full_joint
            for to_marginalize in left_full_joint.domain - variable:
                marginal = Marginal(marginal, to_marginalize)
            marginals.append(marginal)

        # The marginal for the joining variable can be computed using the
        # left or right tree. The result should be the same.
        joining_marginal_left = left_full_joint
        for to_marginalize in left_full_joint.domain - joining_variable:
            joining_marginal_left = Marginal(joining_marginal_left, to_marginalize)
        marginals.append(joining_marginal_left)

        joining_marginal_right = right_full_joint
        for to_marginalize in right_full_joint.domain - joining_variable:
            joining_marginal_right = Marginal(joining_marginal_right, to_marginalize)
        marginals.append(joining_marginal_right)

        np.testing.assert_array_almost_equal(joining_marginal_left.values,
                                             joining_marginal_right.values)
        np.testing.assert_almost_equal(joining_marginal_left.normalization,
                                       joining_marginal_right.normalization)
        
        tables = left_tables + right_tables
        for _ in range(100):

            # Change the values of the input tables and update the marginals.
            for table in tables:
                table.values = np.random.rand(2)

            for marginal in marginals:
                marginal.update()

            for marginal in marginals:
                np.testing.assert_almost_equal(marginal.normalization,
                                               marginals[0].normalization)

    def test_combine_eliminate(self):
        """Test the Product and Marginal recursive update

        This test builds a trees that combines N independent variables into
        a single table and marginalizes each of them out. The result should
        be indentical to the input tables.

        The objective is to test the recusive update calls.

        """

        # Build the test data.
        N = 10
        symbols = [str(i) for i in range(N)]
        variables = [Variable(symbol) for symbol in symbols]
        tables = [Probability(Domain([v])) for v in variables]

        # Combine all tables using a product.
        joint = reduce(Product, tables)

        # Compute all the marginals by marginalizing all variables except
        # one.
        marginals = []
        for variable in variables:
            marginal = joint
            for to_marginalize in joint.domain - variable:
                marginal = Marginal(marginal, to_marginalize)
            marginals.append(marginal)

        # Sort the marginals so they are in the same order as the input
        # tables.
        order = [variables.index(marginal.domain[0]) for marginal in marginals]
        marginals = [marginals[i] for i in order]

        for _ in range(100):

            # Change the values of the input tables and update the marginals.
            for table in tables:
                table.values = np.random.rand(2)

            for marginal in marginals:
                marginal.update()
            
            # Because the variables are independent, the marginals should be equal
            # to the inputs.
            for table, marginal in zip(tables, marginals):
                np.testing.assert_array_almost_equal(table.values, marginal.values)

            for marginal in marginals:
                np.testing.assert_almost_equal(marginal.normalization,
                                               marginals[0].normalization)

    def test_joint_pair(self):
        """Tests using a joint in between all pairs

        This test builds a tree were all pairs of variables are dependent. The
        values of the marginals are diffucult to compute manually, but the
        normalization should be the same for all marginals.

        """

        # Build the test data.
        N = 10
        symbols = [str(i) for i in range(N)]
        variables = [Variable(symbol) for symbol in symbols]
        inputs = [Probability(Domain([v])) for v in variables]
        pairs = [pair for pair in combinations(variables, 2)]
        joints = [Probability(Domain(p), np.random.rand(2, 2)) for p in pairs]
        tables = inputs + joints

        # Build the joint by multiplying all tables. We could do this simply
        # using reduce, but selecting variables at a time builds a more typical
        # update graph.
        selected_tables = [t for t in tables if variables[0] in t.domain]
        joint = reduce(Product, selected_tables)
        used = selected_tables

        for variable in variables[1:]:
            selected_tables = [t for t in tables if variable in t.domain]
            selected_tables = [t for t in selected_tables if t not in used]
            joint = Product(joint, reduce(Product, selected_tables))
            used.extend(selected_tables)

        # Compute all the marginals by marginalizing all variables except
        # one.
        marginals = []
        for variable in variables:
            marginal = joint
            for to_marginalize in joint.domain - variable:
                marginal = Marginal(marginal, to_marginalize)
            marginals.append(marginal)

        # Sort the marginals so they are in the same order as the input
        # tables.
        order = [variables.index(marginal.domain[0]) for marginal in marginals]
        marginals = [marginals[i] for i in order]

        for _ in range(100):

            # Change the values of the input tables and update the marginals.
            for table in inputs:
                table.values = np.random.rand(2)

            for marginal in marginals:
                marginal.update()

            for marginal in marginals:
                np.testing.assert_almost_equal(marginal.normalization,
                                               marginals[0].normalization)
