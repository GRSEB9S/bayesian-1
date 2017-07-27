import unittest

import numpy as np

import bayesian
import bayesian._junction

class TestProduct(unittest.TestCase):
    """Test the Product class"""

    def test_init(self):
        """Test the __init__ method"""

        # Create the variables of the table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')

        # The test tables.
        left = bayesian.Table([a, b], [[0.1, 0.2], [0.3, 0.4]])
        right = bayesian.Table([b], [0.4, 0.6])

        product = bayesian._junction.Product(left, right)
        product.update()
        result = product.result

        np.testing.assert_array_almost_equal(
            result._values, np.array([[0.04, 0.12], [0.12, 0.24]])/0.52)
        self.assertAlmostEqual(result._normalization, 0.52)

        # The product should be the same not matter how it is
        # computed, but the order of the variables might not 
        # be the same.
        other_result = left * right
        map = bayesian.map(result.domain, other_result.domain)
        np.testing.assert_array_almost_equal(result._values.flat[map],
                                             other_result._values.flat)


class TestMarginal(unittest.TestCase):
    """Test the Marginal class"""

    def test_update(self):
        """Test the update method"""

        # Create a test table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')
        domain = bayesian.Domain((a, b))
        table = bayesian.Table(domain, [[0.1, 0.2], [0.3, 0.4]])

        marginal = bayesian._junction.Marginal(table, a)
        marginal.update()
        result = marginal.result

        np.testing.assert_array_almost_equal(
            result._values, np.array([0.4, 0.6]))
        self.assertAlmostEqual(result._normalization, 1.0)
