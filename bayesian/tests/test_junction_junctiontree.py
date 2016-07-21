import unittest
import time

import bayesian
import bayesian.tests.networks


class TestGraphJunctionTree(unittest.TestCase):
    """Test the JunctionTree class"""

    def test_marginals(self):
        """Test the marginals method"""

        # Get the available test networks.
        networks = [
            bayesian.tests.networks.Pyramid(),
            bayesian.tests.networks.CarStartProblem()
        ]

        # Test them one by one.
        for network in networks:
            self._test_marginals_network(network)

    def _test_marginals_network(self, network):
        """Test the marginals method using a test network"""

        # Compute the marginals using the junction tree.
        junction_tree = bayesian.JunctionTree(network)
        junction_tree_marginals = junction_tree.marginals()

        # The marginals computed should be equal to the marginals computed
        # one by one.
        for junction_tree_marginal in junction_tree_marginals:
            marginal = network.marginal(junction_tree_marginal.domain[0])
            self.assertAlmostEqual(junction_tree_marginal[0], marginal[0])
            self.assertAlmostEqual(junction_tree_marginal[1], marginal[1])

        # The results should also hold for unnormalized probabilities.
        junction_tree = bayesian.JunctionTree(network)
        junction_tree_marginals = junction_tree.marginals()
        for junction_tree_marginal in junction_tree_marginals:
            marginal = network.marginal(junction_tree_marginal.domain[0])
            self.assertAlmostEqual(junction_tree_marginal[0], marginal[0])
            self.assertAlmostEqual(junction_tree_marginal[1], marginal[1])

        # Computing the marginals using the junction tree should be faster.
        t0 = time.time()
        junction_tree.marginals()
        t1 = time.time()
        junction_tree_time = t1 - t0

        t0 = time.time()
        for junction_tree_marginal in junction_tree_marginals:
            network.marginal(junction_tree_marginal.domain[0])
        t1 = time.time()
        one_by_one_time = t1 - t0

        self.assertTrue(junction_tree_time < one_by_one_time)

if __name__ == '__main__':
    unittest.main()
