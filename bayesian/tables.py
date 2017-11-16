import numpy as np

from recur.abc import postorder
from recur.graphs import DirectedAcyclic

from bayesian import Domain, map


class Probability(DirectedAcyclic):

    def __init__(self, domain, values=None):

        super().__init__()

        self.domain = domain
        self.updated = False # Should never be True 
        self.normalization = 1.0

        if values is None:
            self.values = np.ones(domain.nb_states)
        else:
            self.values = values

    def eq(self, other):
        """Equality for probability tables

        Two tables are equal if they have the same domain (irrespective of 
        variable order) and the same probabilities associated with 
        each state.

        """

        # Verify that the tables have the same domain.
        if set(self.domain) != set(other.domain):
            return False

        if np.abs(self.normalization - other.normalization) > 1e-5:
            return False

        # Create a mapping from one table to the other and compare the values.
        new_values = self.values.flat[map(self.domain, other.domain)]
        return np.allclose(new_values, other.values.flat, atol=1e-5)

    def le(self, other):
        """Less than equal for probability tables

        A table is less than or equal to another table if its domain is a
        subset of the other and its values are equal when the extra
        variables are marginalized out.

        """

        if self.domain <= other.domain:
            marginal = other
            for variable in other.domain:
                if variable not in self.domain:
                    marginal = marginal.marginalize(variable)
            return self.eq(marginal)
        else:
            return False

    def __getitem__(self, index):
        """Get probability values of the table"""
        return self._values.flat.__getitem__(index)

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
        left_set = set(left.domain)
        right_set = set(right.domain)
        intersection = list(left_set & right_set)
        left_only = list(left_set - right_set)
        right_only = list(right_set - left_set)
        left_domain = left_only + intersection
        right_domain = intersection + right_only
        new_domain = list(left_only + intersection + right_only)

        # Order the values.
        permutation_order = np.zeros((len(left.domain)), dtype=int)
        for j, variable in enumerate(left.domain):
            permutation_order[left_domain.index(variable)] = j
        left_values = left._values.transpose(permutation_order)

        permutation_order = np.zeros((len(right.domain)), dtype=int)
        for j, variable in enumerate(right.domain):
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
        #table_mul(left_ravel, right_ravel, new_values_inline, left_divide, right_mod, new_values_inline.size)
        for i in range(new_values.size):
            new_values_inline[i] = \
                left_ravel[i//left_divide] * right_ravel[i % right_mod]

        # The new normalization factor is the product of the normalization of
        # the tables.
        table = Probability(Domain(new_domain), new_values)
        table.normalization *= left.normalization * right.normalization
        return table

    def __str__(self):
        """String representation of a bayesian.Table"""

        # Add all variable names.
        output = '\n'
        output = 'N: {}\n'.format(self.normalization)
        for variable in self.domain:
            output += '{:<10}'.format(variable.symbol)
        output += 'Prob\n'

        # Add a separator just to make it pretty.
        output += '-' * (len(self.domain) * 10 + 14) + '\n'

        # For every entry of the table.
        for i in range(self.values.size):

            # Print the indices.
            indices = np.unravel_index(i, self.values.shape)
            for index in indices:
                output += '{:<10}'.format(index)

            # Add the value.
            output += '{:<10}'.format(self.values[indices])
            output += '\n'

        return output

    @property
    def unnormalized(self):
        return self._values * self.normalization

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
    
        new_values = np.array(new_values)
        if new_values.size != self.domain.size:
            raise ValueError('The number of values does not correspond to the domain size.')

        normalization = new_values.sum()
        if normalization == 0:
            raise ValueError('A probability table cannot contain all zeros.')

        self._values = new_values / normalization
        self.normalization = normalization

        def invalidate_parents(node):
            return node is self or node.updated == True

        for ancestor in self.ancestors(invalidate_parents):
            ancestor.updated = False

    def marginalize(self, variable):
        """Marginalizes a variable out of the probability table

        Removes a variable from the domain of the table by summing it out. The
        result is a new table without the variable in its domain.

        """

        # Find the position of the variable in the domain of the table.
        index = self.domain.index(variable)

        # Sum over the variable and remove it from the domain.
        new_values = np.sum(self.values, index)
        new_domain = Domain([v for v in self.domain if v is not variable])

        # Create the new table.
        new_table = Probability(new_domain, new_values)
        new_table.normalization = self.normalization
        return new_table

    @postorder('method')
    def update(self):
        pass

class Marginal(Probability):

    def __init__(self, table, variable):
        """Marginalize a variable out of a probability table"""

        self.table = table
        self.variable = variable
        self.index = table.domain.index(variable)

        # Link the marginal table with the source data.
        super().__init__(table.domain - variable)
        self.add(table)
    
    @postorder('method')
    def update(self):
        """Updates the result table of the marginal"""

        if not self.updated:

            self._values = np.sum(self.table._values, self.index)
            self.normalization = self.table.normalization

            self.updated = True

class Product(Probability):

    def __init__(self, left, right):
        """A table that is the result of the product of two tables"""

        self.left = left
        self.right = right
        domain = left.domain * right.domain

        self.left_map = map(left.domain, domain)
        self.right_map = map(right.domain, domain)

        super().__init__(domain)
        self.add(left)
        self.add(right)

    @postorder('method')
    def update(self):
        """Updates the result table of the product"""

        if not self.updated:

            self._values.flat = self.left._values.flat[self.left_map] * \
                                self.right._values.flat[self.right_map]

            normalization = np.sum(self._values) 
            self._values /= normalization
            self.normalization = self.left.normalization * \
                                 self.right.normalization * \
                                 normalization

            self.updated = True
