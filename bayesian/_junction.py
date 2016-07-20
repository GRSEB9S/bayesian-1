import numpy as np
import operator
from copy import copy
from functools import reduce

import bayesian

class Bucket(object):
    def __init__(self, variables):
        """Bucket in a junction tree

        Args:
            variables (set of bayesian.Variables) : The set of variables
                contained in the bucket.

        """

        # The variables of the bucket.
        self.variables = variables

        # The separator above the bucket.
        self.out = None

        # The separators below the bucket.
        self.separators = []

        # The probability tables of the bucket.
        self.tables = []

class Separator(object):
    def __init__(self, variables):
        """Separator in a junction tree

        Args:
            variables (set of bayesian.Variables) : The set of variables
                contained in the bucket.

        """

        # The variables of the separator.
        self.variables = variables

        # The up/downbound messages.
        self.upbound = None
        self.downbound = None

class JunctionTree(object):
    def __init__(self, network, normalize=True):
        """Junction tree of a bayesian network

        Args:
            network (bayesian.Network) : The Bayesian network used to compute
                the domain graph.
            normalize (optional, bool): Indicates if the resulting junction
                tree should be normalized. Default is True.
        """

        # Indicates if the junction tree should be normalized.
        self._normalize = normalize

        # The separators and buckets of the junction tree.
        self.separators = []
        self.buckets = []

        # Keep the network.
        self._network = network

        # Compute the domain graph and use it to initialize the junction
        # tree.
        domain_graph = bayesian.DomainGraph(network)
        self._build_from_graph(domain_graph)

        # Prepare to compute marginals.
        self.fill()

    @property
    def network(self):
        """Get the Bayesian network of the junction tree"""
        return self._network

    def fill(self):
        """Fills the junction tree

        Fills the junction tree by collecting and distributing evidence along
        buckets and separators.

        """
        self._collect_evidence()
        self._distribute_evidence()

    def _collect_evidence(self):
        """Collect the evidence to the root node"""

        for bucket in self.buckets:

            if bucket.out is not None:
                tables = list(bucket.tables)

                # Collect the messages of the separators that are below the
                # the bucket.
                for separator in bucket.separators:
                    tables.append(separator.upbound)

                # Compute the product of all tables and eliminate the
                # variables that are not in the destination separator.
                new_table = reduce(operator.mul, tables)

                separator = bucket.out
                to_remove = []
                for variable in new_table.domain:
                    if variable not in separator.variables:
                        new_table = new_table.marginalize(
                            variable,
                            self._normalize)

                separator.upbound = new_table

    def _distribute_evidence(self):
        """Distribute the evidence from the root node"""

        for bucket in reversed(self.buckets):
            bucket_tables = list(bucket.tables)

            # Collect the messages of the separator that is above
            # the bucket.
            if bucket.out is not None:
                bucket_tables.append(bucket.out.downbound)

            # Distribute the messages to the separators below the bucket.
            for separator in bucket.separators:
                tables = list(bucket_tables)

                # Add the message from all other separators.
                for other_separator in bucket.separators:
                    if other_separator is not separator:
                        tables.append(other_separator.upbound)

                # Compute the product of all tables.
                #new_table = reduce(operator.mul, tables)
                new_table_domain = []
                for t in tables:
                    new_table_domain.extend(t.domain)
                new_table_domain = np.unique(new_table_domain)

                for variable in new_table_domain:
                    if variable not in separator.variables:
                        variable_tables = [t for t in tables if variable in t.domain]
                        new_table = reduce(operator.mul, variable_tables)
                        new_table = new_table.marginalize(
                            variable,
                            self._normalize)
                        tables = [t for t in tables if variable not in t.domain]
                        tables.append(new_table)

                new_table = reduce(operator.mul, tables)

                separator.downbound = new_table

    def marginals(self):
        """Compute all marginals

        Computes the marginal of every variable in the graph. The order of the
        marginals is arbitrary.

        Returns:
            (list of bayesian.Table) : A list of marignal probability tables.
                One for every variable in the graph.

        """

        processed_variables = set()

        marginals = []
        for bucket in self.buckets:

            # Get all the messages from the separators.
            tables = list(bucket.tables)
            if bucket.out is not None:
                tables.append(bucket.out.downbound)
            for separator in bucket.separators:
                tables.append(separator.upbound)

            # Compute the product of all tables.
            new_table = reduce(operator.mul, tables)

            for variable in bucket.variables:
                if variable not in processed_variables:
                    processed_variables.add(variable)

                    # Marginalize all variables except the current objective.
                    current_table = new_table
                    for bucket_variable in bucket.variables:
                        if bucket_variable is not variable:
                            current_table = current_table.marginalize(
                                bucket_variable,
                                self._normalize)

                    if self._normalize:
                        current_table.normalize()

                    marginals.append(current_table)

        return marginals

    def _build_from_graph(self, graph):
        """Get the junction tree of the graph"""

        # Create a copy to the domain graph to be able to modify it.
        graph_copy = copy(graph)

        # The number of nodes removed is used as a bucket and separator
        # index.
        nb_nodes_removed = 0

        # The list of used tables.
        used_tables = set()

        done = False
        while not done:

            # Find a simplicial node if there is one. If not, make one
            # by adding fillins.
            simplicial_node = graph_copy.simplicial_node
            if simplicial_node is None:
                simplicial_node = graph_copy.get_minimal_family()
                simplicial_node.make_simplicial()

            family = simplicial_node.family
            bucket = Bucket(set([node.data for node in family]))
            self.buckets.append(bucket)

            if len(family) < len(graph_copy._nodes):

                # Find all nodes who have heighbors only in the family.
                nodes_to_remove = set()
                nodes_to_keep = set()
                for node in family:
                    if node.family <= family:
                        nodes_to_remove.add(node)
                        nb_nodes_removed = nb_nodes_removed + 1
                    else:
                        nodes_to_keep.add(node)

                # Remove them from the graph.
                for node in nodes_to_remove:
                    graph_copy.remove_node(node)

                # The remaining nodes are part of a separator.
                if len(nodes_to_keep) > 0:
                    separator = Separator(set([n.data for n in nodes_to_keep]))
                    separator.index = nb_nodes_removed
                    separator.downward = bucket
                    bucket.out = separator
                    self.separators.append(separator)

                # Update the index of the bucket.
                bucket.index = nb_nodes_removed
            else:
                nodes_to_remove = family
                bucket.index = len(graph._nodes)
                done = True

            # Find the tables that contains the variables removed from the
            # graph and add them to the bucket.
            variables = [node.data for node in nodes_to_remove]
            possible_tables = graph_copy.network.get_tables(variables)
            bucket.tables = list(set(possible_tables) - used_tables)
            used_tables = used_tables | set(bucket.tables)

        # Connect the separators to their upward bucket.
        for separator in self.separators:
            separator.upward = None
            for bucket in self.buckets:
                if bucket.index > separator.index and \
                        separator.variables <= bucket.variables:
                    separator.upward = bucket
                    bucket.separators.append(separator)
                    break
