from functools import reduce
import operator

import numpy as np

# Import the core classes into the main package.
from bayesian._core import Network, Table, Variable

# Import network analysis classes.
from bayesian._domain import DomainGraph
from bayesian._junction import JunctionTree


def map(subdomain, domain):
    """Returns a map between two domains"""

    subdomain_reordered = tuple(v for v in domain if v in subdomain)
    suborder = tuple(subdomain_reordered.index(v) for v in subdomain)
    subindices = np.arange(subdomain.size).reshape(subdomain.nb_states)
    reordered_subindices = subindices.transpose(suborder).ravel()
    new_shape = tuple(s if v in subdomain else 1
                      for s, v in zip(domain.nb_states, domain))
    reps = tuple(1 if v in subdomain else s
                 for s, v in zip(domain.nb_states, domain))

    return np.tile(reordered_subindices.reshape(new_shape), reps).ravel()


class Domain(tuple):

    def __new__(cls, iterable):
        """Domain of a probability table

        The domain of a probability table is an immutable ordered set of
        variables.

        """

        # The domain cannot be empty.
        if len(iterable) == 0:
            raise ValueError(
                "Empty domains are not permitted. Cannot create a domain "
                "because len(iterable) == 0.")

        # The iterable must not contain duplicates.
        if len(set(iterable)) != len(iterable):
            raise ValueError(
                "The domain of a table must not contain duplicates of "
                "the same variable.")

        self = tuple.__new__(cls, iterable)

        # The size of the domain is the number of unique values of its
        # table.
        self.nb_states = tuple(v.nb_states for v in self)
        self.size = reduce(operator.mul, self.nb_states)

        return self

    @property
    def states(self):
        return (np.array(np.unravel_index(i, self.nb_states))
                for i in range(self.size))

    def __mul__(left, right):
        """Product of table domains

        The result of the product two tables domain is the domain that would
        be obtained if the tables were multiplied.

        """

        # The product of two domains if the concatenation of the domains
        # with the duplicates removed.
        iterable = left + right
        unique = sorted(set(iterable), key=iterable.index)

        return Domain(unique)

    def __sub__(self, variable):
        """Substraction of a variable in the domain

        The result of the subtraction of a variable from a domain is the
        domain that would be obtained if the variable was marginalized out
        of the table.

        """

        # If the variable is not in the domain, it cannot be removed.
        if variable not in self:
            raise ValueError(
                "The variable {} is not part of the domain {}."
                .format(variable, self))

        return Domain([v for v in self if v != variable])
