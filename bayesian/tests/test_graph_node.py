import unittest
import bayesian._graph

class TestNode(unittest.TestCase):

    def test_constructor(self):
        """Test the Node constructor"""

        node1 = bayesian._graph.Node('A')

    def test_add_link(self):
        """Test the add_link method"""

        node1 = bayesian._graph.Node('A')
        node2 = bayesian._graph.Node('B')

        # Adding a link between node1 and node2 should also create a link 
        # between node2 and node1.
        node1.add_link(node2)
        self.assertTrue(node2 in node1.links)
        self.assertTrue(node1 in node2.links)

    def test_is_simplical(self):
        """Test the is_simplicial property"""

        node1 = bayesian._graph.Node('A')
        node2 = bayesian._graph.Node('B')
        node3 = bayesian._graph.Node('C')
        node4 = bayesian._graph.Node('D')

        # Add the links. There are no links between nodes 1 and 3.
        node1.add_link(node2)
        node2.add_link(node3)
        node1.add_link(node4)
        node2.add_link(node4)
        node3.add_link(node4)

        # Node 4 is not simplicial.
        self.assertFalse(node4.is_simplicial)

        # If we add the missing link between nodes 1 and 3, node 4 is now
        # simplicial.
        node1.add_link(node3)
        self.assertTrue(node4.is_simplicial)
