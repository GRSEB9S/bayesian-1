import unittest

import numpy as np

import bayesian


class TestBayesianTable(unittest.TestCase):
    """Tests for the bayesian.Table class"""

    def test_marginalize(self):
        """Test the marginalize method"""

        # Create the variables of the table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')

        # The test table.
        table = bayesian.Table([a, b], [[0.2, 0.8], [0.7, 0.3]])

        # Marginalizing a out of the table yields
        #   0: 0.2 + 0.7 = 0.9
        #   1: 0.8 + 0.3 = 1.1
        new_table = table.marginalize(a)
        new_table.unnormalize()
        self.assertAlmostEqual(new_table[0], 0.9)
        self.assertAlmostEqual(new_table[1], 1.1)

        # Marginalizing b and normalizing yields
        #   0: (0.2 + 0.8) / 2.0 = 0.50
        #   1: (0.7 + 0.3) / 2.0 = 0.50
        new_table = table.marginalize(b)
        self.assertAlmostEqual(new_table[0], 0.5)
        self.assertAlmostEqual(new_table[1], 0.5)

    def test_unnormalize(self):
        """Test the unnormalize method"""

        # Create the variables of the table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')

        # The test tables.
        values1 = np.random.rand(2)
        table1 = bayesian.Table([a], values1)
        values2 = np.random.rand(2,2)
        table2 = bayesian.Table([a, b], values2)

        # The product of the two tables should be normalized.
        table3 = table1 * table2
        self.assertAlmostEqual(np.sum(table3._values), 1.0)

        # Unnormalizing should yield the same results as multiplying the
        # values directly.
        table3.unnormalize()
        np.testing.assert_array_almost_equal(
            table3._values[0, :], values1[0] * values2[0, :])
        np.testing.assert_array_almost_equal(
            table3._values[1, :], values1[1] * values2[1, :])

if __name__ == '__main__':
    unittest.main()
