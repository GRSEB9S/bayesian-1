import numpy as np

from recur.abc import postorder
from recur.graphs import DirectedAcyclic

from bayesian import map


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
        self.nomalization = normalization

        for ancestor in self.ancestors():
            ancestor.updated = False

    @postorder('method')
    def update(self):
        pass

class Marginal(Probability):

    def __init__(self, table, variable):
        """Marginalize a variable out of a probability table"""

        self.table = table
        self.variable = variable
        self.index = table.domain.index(variable)

        domain = table.domain - variable
        self.map = map(domain, table.domain)

        # Link the marginal table with the source data.
        super().__init__(domain)
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
