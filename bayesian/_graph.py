from itertools import combinations

import numpy as np

class UndirectedGraph(object):
    def __init__(self):
        """Undirected graph

        The undirected graph represents a set of nodes interconnected by links
        which do not have a direction. The domain graph of a Bayesian network
        is an example of an undirected graph.

        """

        self._nodes = set()

    def add_node(self, node):
        """Adds a node to the undirected graph"""
        self._nodes.add(node)

    def get_node(self, data):
        """Get the node with the specified data"""
        nodes = list(self._nodes)
        data_list = [node.data for node in nodes]
        return nodes[data_list.index(data)]


class Node(object):
    def __init__(self, data):
        """Node in a graph

        The Node class represents a node in an undirected graph.

        Args:
            data (object) : The data associated with the node.

        """

        self._data = data
        self._links = set() 

    @property
    def data(self):
        """Get the data of the node"""
        return self._data
    
    @property
    def is_simplicial(self):
        """Indicates if a node is simplicial

        A node is simplicial if all the nodes it is linked to are pairwise
        linked.

        Returns:
            (bool) : True if the node is simplicial, False otherwise.

        """
        
        # Verify if all the nodes are pairwise linked.
        for node, other_node in combinations(self._links, 2):
            if node not in other_node.links:
                return False

        return True

    @property
    def links(self):
        """Get the links of the node"""
        return self._links

    def add_link(self, node):
        """Add a link to a node"""
        
        # Because the links are undirected, it is added to both nodes.
        self._links.add(node)
        node._links.add(self)
