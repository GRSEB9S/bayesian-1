from itertools import combinations
from copy import copy

import numpy as np

import bayesian


class DomainGraph(object):
    def __init__(self, network=None):
        """Domain graph of a bayesian network

        The DomainGraph class represents the domain graph of a Bayesian
        network. It is an undirected graph where nodes represent variables.
        There is a link between two nodes if the are part ot the domain of the
        same table.

        Args:
            network (optional, bayesian.Network) : The Bayesian network used
                to compute the domain graph.
        """

        self._nodes = set()

        # If the network is not provided, the domain graph is empty.
        if network is None:
            network = bayesian.Network()
        self._network = network

        # Initialize the domain graph using the provided network.
        self._build_from_network(network)

    def __copy__(self):
        """Shallow copy of a domain graph"""

        # The deep copy does not duplicated the Bayesian network.
        shallow_copy = DomainGraph(self._network)
        shallow_copy._nodes = copy(self._nodes)

        return shallow_copy

    @property
    def network(self):
        """Get the Bayesian network of the domain graph"""
        return self._network

    @property
    def simplicial_node(self):
        """Get a simplicial node"""
        for node in self._nodes:
            if node.is_simplicial:
                return node
        return None

    @property
    def isolated_node(self):
        """Get an isolated node"""
        return next((n for n in self._nodes if n.is_isolated), None)


    def add_node(self, node):
        """Adds a node to the undirected graph"""
        self._nodes.add(node)

    def get_node(self, data):
        """Get the node with the specified data"""
        nodes = list(self._nodes)
        data_list = [node.data for node in nodes]
        return nodes[data_list.index(data)]

    def get_almost_simplicial(self):
        """Get the node that is closest to being simplicial"""

        selected = next(iter(self._nodes))
        for node in self._nodes:
            if node.missing_links() < selected.missing_links():
                selected = node

        return selected

    def get_minimal_family(self):
        """Get the node that has the smallest family in the graph"""

        selected = next(iter(self._nodes), None)
        for node in self._nodes:
            if len(node.family) < len(selected.family):
                selected = node

        return selected

    def remove_node(self, node_to_remove):
        """Removes a node from the graph"""
        self._nodes.discard(node_to_remove)
        for node in self._nodes:
            node.remove_link(node_to_remove)

    def _build_from_network(self, network):
        """Builds the domain graph of a Bayesian network

        Args:
            network (bayesian.Network) : The network used to build the domain
                graph.

        """

        # Create a node for every variable in the network.
        domain = network.domain
        for variable in domain:
            self.add_node(Node(variable))

        # Add the links between variables that are in the domain of the
        # same table.
        tables = network.get_tables()
        for table in tables:
            for v1, v2 in combinations(table.domain, 2):
                node = self.get_node(v1)
                node.add_link(self.get_node(v2))


class Node(object):
    def __init__(self, data):
        """Node in a domain graph

        The Node class represents a node in the domain graph of a Bayesian
        network.

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
    def family(self):
        """Get the family of a node"""
        nodes = [n for n in self.links]
        nodes.append(self)
        return set(nodes)

    @property
    def is_isolated(self):
        """Indicates if a node is isolated"""
        return len(self._links) == 0

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

    def make_simplicial(self):
        """Makes a node simplicial

        Makes the node simplicial by adding links between its neighbors.

        """

        # Add the missing links between neightbors.
        for node, other_node in combinations(self._links, 2):
            if node not in other_node.links:
                node.add_link(other_node)

    def missing_links(self):
        """Returns the number of missing links to make the node simplicial"""

        # Count the number of missing links.
        missing = 0
        for node, other_node in combinations(self._links, 2):
            if node not in other_node.links:
                missing = missing + 1

        return missing

    def remove_link(self, node):
        """Removes a link between nodes"""
        self._links.discard(node)
        node._links.discard(self)
