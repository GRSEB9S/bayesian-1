# cython: profile=True
import operator
from functools import reduce
from itertools import combinations

import numpy as np

import bayesian
from bayesian._optimized import table_mul


class Network(object):
    def __init__(self, network=None):
        """Bayesian network

        The bayesian.Network class implement a Bayesian network. A valid
        Bayesian network is defined by:

            - A set of variables each having a finite number of mutually
                exclusive states.
            - A set of directed edges between variables.
            - The variables and the edges must form an acyclic directed graph.

        In this package, edges and variables in the network are not defined
        explicitly. They are instead described using probability tables which
        capture de same information. For the purpose of this package, a
        Bayesian network is therefore a collection of tables.

        Args:
            network (optional, bayesian.Network) : When present, the new
                network is a copy of the supplied network.
        """

        if network is None:
            self._tables = []
        else:
            self._tables = list(network._tables)

    @property
    def domain(self):
        """Get the domain of the Bayesian network

        The domain of a Bayesian network is the union of the domain of all the
        tables in contains. This is equivalent to the set of all variables in
        the graph of the network.

        Returns:
            (list of bayesian.Variable) : All the variables in the graph of the
                network. Returns an empty list if there are no variables in the
                graph.

        """

        # The domain of the network is the union of the domain of the tables
        # of the network.
        domain = []
        for table in self._tables:
            domain.extend(table.domain)

        return list(set(domain))

    def __str__(self):
        """String representation of the bayesian.Network"""

        # Print every table of the network.
        output = ''
        for table in self._tables:
            output += str(table)

        return output

    def add_table(self, table):
        """Adds a probability table to the network"""
        self._tables.append(table)

    def get_tables(self, variables=None):
        """Get tables with variables in their domain

        Returns a list of tables that have the specified variables in their
        domain. If the variables are not supplied, all tables are returned.

        Args:
            variables (optional, bayesian.Variable or list) : The variable or
                list of variables that are in the domain of the returned
                tables.

        Returns:
            (list of bayesian.Table) : The tables that have the variables in
            their domain.

        """

        if variables is None:
            return self._tables

        if not hasattr(variables, '__iter__'):
            variables = [variables]

        # Find all tables with the variable in their domain.
        tables = []
        for variable in variables:
            for table in self._tables:
                if variable in table.domain and table not in tables:
                    tables.append(table)

        return tables

    def joint_domain(self, variable):
        """Returns the domain of the joint table for a variable

        The domain of the joint table is given by the union of the domains of
        the tables where the variable appears.

        Args:
            variable (bayesian.Variable) : The variable used to compute the
                joint table.

        Returns:
            (list of bayesian.Variables) : The domain of the joint table.

        """

        # Find all tables with the variable in their domain.
        tables = self.get_tables(variable)

        # Create a new network with the tables and get its domain.
        network = Network()
        network._tables = tables

        return network.domain

    def marginalize(self, variable):
        """Marginalizes a variable out of the Bayesian network

        Removes a variable from the domain of the Bayesian network by summing
        it out. The result is a new Bayesian network.

        Args:
            variable (bayesian.Variable) : The variable to be removed.

        Returns:
            (bayesian.Network) : A new Bayesian network without the variable in
                its domain.

        """

        # Find all tables with the variable in their domain.
        tables = []
        for table in self._tables:
            if variable in table.domain:
                tables.append(table)

        # Compute the product of all tables and marginalize the variable out
        # of the result.
        new_table = reduce(Table.__mul__, tables)
        new_table = new_table.marginalize(variable)

        # Create a new BN.
        new_network = Network()
        new_network._tables = list(self._tables)

        # Remove all table with the variable in their domain and add the
        # new one.
        for table in tables:
            new_network._tables.remove(table)
        new_network._tables.append(new_table)

        return new_network

    def marginal(self, variable):
        """Computes the marginal probability table of a variable

        Computes the marginal probability table of a variable by marginalizing
        all other variables in the network. If you wish to compute the
        marginal for all variables in the network, use the marginals method
        instead.

        Args:
            variable (bayesian.Variable) : The variable for which the marginal
                probability is computed.

        Returns:
            (bayesian.Table) : The probability table of the variable.

        """

        # Marginalize all other variables of the network.
        network = self
        domain = self.domain
        for domain_variable in domain:
            if variable != domain_variable:
                network = network.marginalize(domain_variable)

        # Compute the product of the remaining tables.
        new_table = reduce(Table.__mul__, network._tables)

        return new_table

    def marginals(self):
        """ Computes the marginal probability for all variables

        Computes the marginal probability tables for all variables in the
        domain of the network. Using this method is significantly faster
        than using the marginal method repeatedly.

        Returns:
            (list of bayesian.Table) : The marginal probability tables for all
                variables in the network.

        """

        # If the domain more than one variable, recursively marginalize them.
        domain = self.domain
        nb_variables = len(domain)
        if nb_variables > 1:

            marginals = []

            # Marginalize the two halves of the domain separately.
            halves = [domain[nb_variables//2:], domain[:nb_variables//2]]
            for to_marginalize in halves:
                new_network = Network(self)
                for variable in to_marginalize:
                    new_network = new_network.marginalize(variable)
                marginals.extend(new_network.marginals())

        else:
            marginals = [self.marginal(domain[0])]

        return marginals


class Table(object):
    def __init__(self, domain, values=None, normalization=1.0):
        """Probability table in a Bayesian network

        A probability table assigns a probability to every possible combination
        of variable states in its domain.

        Args:
            domain (list of bayesian.Variable) : The variables in the domain
                of the table.
            values (optional, np.array) : An array with a number of dimensions
                equal to the number of variables in the domain of the table.
                Each dimension has a size equal to the number of states of the
                corresponding variable. Each element of the array gives the
                probability of a specific combination of variable states.

                For example, values[0, 2] gives the probability that the first
                variable is in the state 0 and the second variables is in the
                state 2.

                If the values are not provided, all states are given an equal
                probability.

        """
        self._domain = bayesian.Domain(domain)
        self._normalization = normalization

        # If the probabilities are not provided, all events are equiprobable.
        if values is None:
            self._values = np.ones(domain.nb_states)
            self.normalize()
        else:
            # Normalize the values.
            coefficient = np.sum(values)
            if coefficient != 0:
                self._normalization *= coefficient
                self._values = np.array(values) / coefficient
            else:
                self._values = np.array(values)

    @property
    def domain(self):
        """Get the domain of the bayesian.Table"""
        return self._domain

    def __getitem__(self, index):
        """Get probability values of the bayesian.Table"""
        return self._values.__getitem__(index)

    def __mul__(left, right):
        """Multiplies two bayesian.Table

        Multiplies two tables by first extending their domain to be the union
        of the domain of the two tables. The result is a new probability table
        with every possible combination of the states of both tables.

        Args:
            left (bayesian.Table) : The left hand table in the product.
            right (bayesian.Table) : The right hand table in the product.

        Returns:
            (bayesian.Table) : The result of the product.

        """

        # Order the variables.
        left_set = set(left._domain)
        right_set = set(right._domain)
        intersection = list(left_set & right_set)
        left_only = list(left_set - right_set)
        right_only = list(right_set - left_set)
        left_domain = left_only + intersection
        right_domain = intersection + right_only
        new_domain = list(left_only + intersection + right_only)

        # Order the values.
        permutation_order = np.zeros((len(left._domain)), dtype=int)
        for j, variable in enumerate(left._domain):
            permutation_order[left_domain.index(variable)] = j
        left_values = left._values.transpose(permutation_order)

        permutation_order = np.zeros((len(right._domain)), dtype=int)
        for j, variable in enumerate(right._domain):
            permutation_order[right_domain.index(variable)] = j
        right_values = right._values.transpose(permutation_order)

        left_divide = 1
        for v in right_only:
            left_divide *= v.nb_states
        right_mod = 1
        for v in right_domain:
            right_mod *= v.nb_states

        # Multiply the table values.
        shape = np.array([v.nb_states for v in new_domain], dtype=np.int)
        new_values = np.zeros(shape, dtype=np.float)
        new_values_inline = new_values.ravel()
        left_ravel = left_values.ravel()
        right_ravel = right_values.ravel()
        table_mul(left_ravel, right_ravel, new_values_inline, left_divide, right_mod, new_values_inline.size)
        #for i in range(new_values.size):
        #    new_values_inline[i] = \
        #        left_ravel[i//left_divide] * right_ravel[i % right_mod]

        # The new normalization factor is the product of the normalization of
        # the tables.
        table = Table(
            new_domain, new_values,
            left._normalization * right._normalization)
        return table

    def __str__(self):
        """String representation of a bayesian.Table"""

        # Add all variable names.
        output = '\n'
        for variable in self._domain:
            output += '{:<10}'.format(variable.symbol)
        output += 'Prob\n'

        # Add a separator just to make it pretty.
        output += '-' * (len(self._domain) * 10 + 14) + '\n'

        # For every entry of the table.
        for i in range(self._values.size):

            # Print the indices.
            indices = np.unravel_index(i, self._values.shape)
            for index in indices:
                output += '{:<10}'.format(index)

            # Add the value.
            output += '{:<10}'.format(self._values[indices])
            output += '\n'

        return output

    def marginalize(self, variable):
        """Marginalizes a variable out of the bayesian.Table

        Removes a variable from the domain of the table by summing it out. The
        result is a new table without the variable in its domain.

        Args:
            variable (bayesian.Variable) : The variable to be marginalized.

        Returns:
            (bayesian.Table) : A new table without the variable in its domain.

        """

        # Find the position of the variable in the domain of the table.
        index = self._domain.index(variable)

        # Sum over the variable and remove it from the domain.
        new_values = np.sum(self._values, index)
        new_domain = list(self._domain)
        new_domain.pop(index)

        # Create the new table.
        new_table = Table(new_domain, new_values, self._normalization)
        return new_table

    def normalize(self):
        """Normalizes a bayesian.Table

        Normalizes a table so that the sum of all values is 1.0.

        """
        self._normalization = np.sum(self._values)
        self._values /= self._normalization

    def unnormalize(self):
        """Unnormalize a bayesian.Table"""
        self._values *= self._normalization
        self._normalization = 1.0

    def _expand_domain(self, new_domain):
        """Expands the domain of a bayesian.Table

        Expands the domain of a table by adding a new variable. Because the
        events of the table are independent of the new variable, this simply
        duplicates the existing values of the table.

        Args:
            new_domain (list of bayesian.Variable) : The new domain of the
                table.

        Returns:
            (bayesian.Table) : A new table whose domain is new_domain.

        """

        # Find variables of the new domain which are not in the left table and
        # add them to the front of the table.
        expanded_domain = list(self._domain)
        tile_shape = [1] * len(expanded_domain)
        for variable in new_domain:
            if variable not in self._domain:
                expanded_domain.insert(0, variable)
                tile_shape.insert(0, variable.nb_states)

        # Repeat the table values so they span the new domain.
        expanded_values = np.tile(self._values, tile_shape)

        # Permute the table values to get the same order as the new domain.
        permutation_order = np.zeros((len(expanded_domain)))
        for i, variable in enumerate(expanded_domain):
            permutation_order[new_domain.index(variable)] = i
        expanded_values = expanded_values.transpose(permutation_order)

        return Table(list(new_domain), expanded_values)


class Variable(object):
    def __init__(self, symbol, nb_states=2):
        """Variable in a Bayesian network

        The bayesian.Variable represents a variable in a Bayesian network with
        a finite number of mutually exclusive states.

        Args:
            symbol (str) : A string used to print the variable. Has no impact
                on the identity of the variable i.e. two varibles with the
                same symbol are still distinct variable in the Bayesian
                network.
            nb_states (optional, int) : The number of states of the variable.
                The default is 2.

        """

        self._symbol = symbol
        self._nb_states = nb_states

    def __str__(self):
        """String representation of a baeysian.Variable"""
        return self._symbol

    @property
    def nb_states(self):
        """Get the number of states of the bayesian.Variable"""
        return self._nb_states

    @property
    def symbol(self):
        """Get the symbol of the bayesian.Variable"""
        return self._symbol
