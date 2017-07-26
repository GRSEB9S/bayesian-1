import unittest

import numpy as np

import bayesian
import bayesian._junction

class TestProduct(unittest.TestCase):

    def test_init(self):

        # Create the variables of the table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')

        # The test tables.
        left = bayesian.Table([a, b], [[0.2, 0.8], [0.7, 0.3]])
        right = bayesian.Table([a], [0.1, 0.9])

        product = bayesian._junction.Product(left, right)
        product.update()
        result = product.result

        # The product should be the same not matter how it is
        # computed, but the order of the variables might not 
        # be the same.
        other_result = left * right
        map = bayesian.map(result.domain, other_result.domain)
        np.testing.assert_array_almost_equal(result._values.flat[map],
                                             other_result._values.flat)
