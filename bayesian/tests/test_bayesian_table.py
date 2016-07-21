import unittest
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
        self.assertAlmostEqual(new_table[0], 0.9)
        self.assertAlmostEqual(new_table[1], 1.1)

        # Marginalizing b and normalizing yields
        #   0: (0.2 + 0.8) / 2.0 = 0.50
        #   1: (0.7 + 0.3) / 2.0 = 0.50
        new_table = table.marginalize(b)
        new_table.normalize()
        self.assertAlmostEqual(new_table[0], 0.5)
        self.assertAlmostEqual(new_table[1], 0.5)

if __name__ == '__main__':
    unittest.main()
