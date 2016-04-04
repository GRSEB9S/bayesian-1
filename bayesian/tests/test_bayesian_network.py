import unittest
import bayesian

class TestBayesianNetwork(unittest.TestCase):
    """Tests for the bayesian.Network class"""

    def test_marginals(self):
        """Test the marginals method"""

        # Create the variables of the table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')
        c = bayesian.Variable('c')

        # The test tables.
        table_a = bayesian.Table([a], [0.15, 0.85])
        table_ab = bayesian.Table([a, b], [[0.2, 0.8],[0.7, 0.3]])
        table_abc = bayesian.Table([a, b, c], [
            [[0.5, 0.5], [0.3, 0.7]],
            [[0.25, 0.75], [0.9, 0.1]]
        ])

        # Create the network.
        network = bayesian.Network()
        network.add_table(table_a)
        network.add_table(table_ab)
        network.add_table(table_abc)

        # Compute the marginals with normalization.
        marginals = network.marginals()
        
        # The marginals computed in a batch should be equal to the ones 
        # computed individually.
        for marginal in marginals:
            individual_marginal = network.marginal(marginal.domain[0])
            self.assertAlmostEqual(individual_marginal[0], marginal[0])
            self.assertAlmostEqual(individual_marginal[1], marginal[1])

        # Compute the marginals without normalization.
        marginals = network.marginals(False)
        
        # The marginals computed in a batch should be equal to the ones 
        # computed individually.
        for marginal in marginals:
            individual_marginal = network.marginal(marginal.domain[0], False)
            self.assertAlmostEqual(individual_marginal[0], marginal[0])
            self.assertAlmostEqual(individual_marginal[1], marginal[1])

if __name__ == '__main__':
    unittest.main()
