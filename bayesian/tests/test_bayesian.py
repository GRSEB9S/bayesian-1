import unittest

import numpy as np

import bayesian


class Function(unittest.TestCase):
    """Test the functions of the bayesian.__init__.py module"""

    def test_map(self):

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)
        d = bayesian.Variable('d', 2)

        domain = bayesian.Domain((a, b))
        subdomain = bayesian.Domain((a, b))
        map = bayesian.map(subdomain, domain)
        np.testing.assert_array_almost_equal(map, [0, 1, 2, 3])

        domain = bayesian.Domain((a, b))
        subdomain = bayesian.Domain((a,))
        map = bayesian.map(subdomain, domain)
        np.testing.assert_array_almost_equal(map, [0, 0, 1, 1])

        domain = bayesian.Domain((a, b, c, d))
        subdomain = bayesian.Domain((a, d))
        map = bayesian.map(subdomain, domain)
        np.testing.assert_array_almost_equal(
            map, 
            [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3])

        domain = bayesian.Domain((a, b, c, d))
        subdomain = bayesian.Domain((d, a))
        map = bayesian.map(subdomain, domain)
        np.testing.assert_array_almost_equal(
            map, 
            [0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3])


class Domain(unittest.TestCase):

    def test_new(self):
        """Test the bayesian.Domain.__new__ method"""

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        # Create a simple domain.
        domain = bayesian.Domain((a, b, c))

        # Duplicates should raise an error.
        self.assertRaises(ValueError, bayesian.Domain, (a, a, b))

        # Empty domains are not permitted.
        self.assertRaises(ValueError, bayesian.Domain, [])

    def test_mul(self):
        """Test the bayesian.Domain.__mul__ method"""

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)
        d = bayesian.Variable('d', 2)
        e = bayesian.Variable('e', 2)

        # The new domain is the union of the inputs.
        left = bayesian.Domain((a, b, c))
        right = bayesian.Domain((d, e))
        self.assertEqual(left * right, bayesian.Domain((a, b, c, d, e)))

        # Duplicates should be removed.
        left = bayesian.Domain((a, b, c))
        right = bayesian.Domain((c, d, e))
        self.assertEqual(left * right, bayesian.Domain((a, b, c, d, e)))

    def test_sub(self):
        """Test the bayesian.Domain.__sub__ method"""

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        # If the variable is present, it is removed.
        domain = bayesian.Domain((a, b, c))
        subdomain = domain - b
        self.assertEqual(subdomain, bayesian.Domain((a, c)))

        # Cannot remove what is not present.
        domain = bayesian.Domain((a, c))

        def remove_2():
            subdomain = domain - b

        self.assertRaises(ValueError, remove_2)
