import unittest
import bayesian

class TestBayesianNetwork(unittest.TestCase):
    """Tests for the bayesian.Network class"""

    def test_joint_domain(self):
        """Test the joint_domain method"""

        # Create the variables of the table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')
        c = bayesian.Variable('c')

        # The test tables.
        table_a = bayesian.Table([a], [15, 85])
        table_ab = bayesian.Table([a, b], [[0.2, 0.8],[0.7, 0.3]])
        table_abc = bayesian.Table([b, c],[[0.5, 0.5], [0.3, 0.7]])

        # Create the network.
        network = bayesian.Network()
        network.add_table(table_a)
        network.add_table(table_ab)
        network.add_table(table_abc)

        # The joint for a should contain a and b.
        a_domain = network.joint_domain(a)
        self.assertEqual(len(a_domain), 2)
        self.assertEqual(a in a_domain, True)
        self.assertEqual(b in a_domain, True)
        
        # The joint for b should contain a, b, and c.
        b_domain = network.joint_domain(b)
        self.assertEqual(len(b_domain), 3)
        self.assertEqual(a in b_domain, True)
        self.assertEqual(b in b_domain, True)
        self.assertEqual(c in b_domain, True)

        # The joint for c should contain b and c.
        c_domain = network.joint_domain(c)
        self.assertEqual(len(c_domain), 2)
        self.assertEqual(b in c_domain, True)
        self.assertEqual(c in c_domain, True)

    def test_marginals(self):
        """Test the marginals method"""

        # Create the variables of the table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')
        c = bayesian.Variable('c')

        # The test tables.
        table_a = bayesian.Table([a], [15, 85])
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

    def test_domain_graph(self):
        """Test the domain_graph property"""

        # Create the variables of the table.
        a = bayesian.Variable('a')
        b = bayesian.Variable('b')
        c = bayesian.Variable('c')

        # The test tables.
        table_a = bayesian.Table([a], [15, 85])
        table_ab = bayesian.Table([a, b], [[0.2, 0.8], [0.7, 0.3]])
        table_ac = bayesian.Table([a, c], [[0.5, 0.5], [0.3, 0.7]])

        # Create the network.
        network = bayesian.Network()
        network.add_table(table_a)
        network.add_table(table_ab)
        network.add_table(table_ac)

        # Get the domain graph of the network.
        domain_graph = network.domain_graph

        # There should be a link between a and b, a and c, but not between
        # b and c.
        node_a = domain_graph.get_node(a)
        node_b = domain_graph.get_node(b)
        node_c = domain_graph.get_node(c)
        self.assertTrue(node_b in node_a.links)
        self.assertTrue(node_c in node_a.links)
        self.assertFalse(node_b in node_c.links)
        
if __name__ == '__main__':
    unittest.main()
