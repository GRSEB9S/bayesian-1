import unittest

import numpy as np

import  bayesian
import  bayesian.tables


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
