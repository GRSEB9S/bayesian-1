import unittest

import numpy as np

import bayesian


class Function(unittest.TestCase):
    """Test the functions of the bayesian.__init__.py module"""

    def test_map_BCA_ABC(self):

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        domain = bayesian.Domain([a, b, c])
        subdomain = bayesian.Domain([b, c, a])
        
        map = bayesian.map(subdomain, domain)

        np.testing.assert_array_equal(
            map, 
            [0, 2, 4, 6, 1, 3, 5, 7])

#    def test_map_simple(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#        d = bayesian.Variable('d', 2)
#
#        domain = bayesian.Domain((a, b))
#        subdomain = bayesian.Domain((a, b))
#        map = bayesian.map(subdomain, domain)
#        np.testing.assert_array_almost_equal(map, [0, 1, 2, 3])
#
#        domain = bayesian.Domain((a, b))
#        subdomain = bayesian.Domain((a,))
#        map = bayesian.map(subdomain, domain)
#        np.testing.assert_array_almost_equal(map, [0, 0, 1, 1])
#
#        domain = bayesian.Domain((a, b, c, d))
#        subdomain = bayesian.Domain((a, d))
#        map = bayesian.map(subdomain, domain)
#        np.testing.assert_array_almost_equal(
#            map, 
#            [0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3])
#
#        domain = bayesian.Domain((a, b, c, d))
#        subdomain = bayesian.Domain((d, a))
#        map = bayesian.map(subdomain, domain)
#        np.testing.assert_array_almost_equal(
#            map, 
#            [0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3])

    def test_map_AB_ABC(self):

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        domain = bayesian.Domain([a, b, c])
        subdomain = bayesian.Domain([a, b])
        
        map = bayesian.map(subdomain, domain)

        np.testing.assert_array_equal(
            map, 
            [0, 0, 1, 1, 2, 2, 3, 3])

    def test_map_BA_ABC(self):

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        domain = bayesian.Domain([a, b, c])
        subdomain = bayesian.Domain([b, a])
        
        map = bayesian.map(subdomain, domain)

        np.testing.assert_array_equal(
            map, 
            [0, 0, 2, 2, 1, 1, 3, 3])

    def test_map_AB_ACB(self):

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        domain = bayesian.Domain([a, c, b])
        subdomain = bayesian.Domain([a, b])
        
        map = bayesian.map(subdomain, domain)

        np.testing.assert_array_equal(
            map, 
            [0, 1, 0, 1, 2, 3, 2, 3])

    def test_map_AB_BAC(self):

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        domain = bayesian.Domain([b, a, c])
        subdomain = bayesian.Domain([a, b])
        
        map = bayesian.map(subdomain, domain)

        np.testing.assert_array_equal(
            map, 
            [0, 0, 2, 2, 1, 1, 3, 3])


#    def test_map_AB_BCA(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#
#        domain = bayesian.Domain([b, c, a])
#        subdomain = bayesian.Domain([a, b])
#        
#        map = bayesian.map(subdomain, domain)
#
#        np.testing.assert_array_equal(
#            map, 
#            [0, 2, 0, 2, 1, 3, 1, 3])
#
#    def test_map_AB_CAB(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#
#        domain = bayesian.Domain([c, a, b])
#        subdomain = bayesian.Domain([a, b])
#        
#        map = bayesian.map(subdomain, domain)
#
#        np.testing.assert_array_equal(
#            map, 
#            [0, 1, 2, 3, 0, 1, 2, 3])
#
#    def test_map_AB_CBA(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#
#        domain = bayesian.Domain([c, b, a])
#        subdomain = bayesian.Domain([a, b])
#        
#        map = bayesian.map(subdomain, domain)
#
#        np.testing.assert_array_equal(
#            map, 
#            [0, 2, 1, 3, 0, 2, 1, 3])
#
#    def test_map_ABC_CBA(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#
#        domain = bayesian.Domain([c, b, a])
#        subdomain = bayesian.Domain([a, b, c])
#        
#        map = bayesian.map(subdomain, domain)
#
#        np.testing.assert_array_equal(
#            map, 
#            [0, 4, 2, 6, 1, 5, 3, 7])
#
#    def test_map_ACB_ABC(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#
#        domain = bayesian.Domain([a, b, c])
#        subdomain = bayesian.Domain([a, c, b])
#        
#        map = bayesian.map(subdomain, domain)
#
#        np.testing.assert_array_equal(
#            map, 
#            [0, 2, 1, 3, 4, 6, 5, 7])
#
#    def test_map_BAC_ABC(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#
#        domain = bayesian.Domain([a, b, c])
#        subdomain = bayesian.Domain([b, a, c])
#        
#        map = bayesian.map(subdomain, domain)
#
#        np.testing.assert_array_equal(
#            map, 
#            [0, 1, 4, 5, 2, 3, 6, 7])
#
#    def test_map_C_ABC(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#
#        domain = bayesian.Domain([a, b, c])
#        subdomain = bayesian.Domain([c])
#        
#        map = bayesian.map(subdomain, domain)
#
#        np.testing.assert_array_equal(
#            map, 
#            [0, 1, 0, 1, 0, 1, 0, 1])
#
#    def test_map_DACBEF_ABCDEF(self):
#
#        a = bayesian.Variable('a', 2)
#        b = bayesian.Variable('b', 2)
#        c = bayesian.Variable('c', 2)
#        d = bayesian.Variable('d', 2)
#        e = bayesian.Variable('e', 2)
#        f = bayesian.Variable('f', 2)
#
#        domain = bayesian.Domain([a, b, c, d, e, f])
#        subdomain = bayesian.Domain([d, a, c, b, e, f])
#
#        map = bayesian.map(subdomain, domain)
#
#        np.testing.assert_array_equal(
#            map[:8], 
#            [0, 1, 2, 3, 16, 17, 18, 19])


class Domain(unittest.TestCase):
    """Test the bayesian.Domain class"""

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

    def test_eq(self):
        """Test the __eq__ method"""

        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        self.assertTrue(bayesian.Domain((a, b)) == bayesian.Domain((a, b)))
        self.assertTrue(bayesian.Domain((a, b)) == bayesian.Domain((b, a)))

        self.assertFalse(bayesian.Domain((a,)) == bayesian.Domain((a, b)))
        self.assertFalse(bayesian.Domain((a, b)) == bayesian.Domain((c, a)))
        self.assertFalse(bayesian.Domain((a, b, c)) == bayesian.Domain((a, b)))

    def test_ge(self):
        """Test the __ge__ method"""

        # A domain is greater than or equal to another if its variables
        # are a superset of the other.
        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        self.assertTrue(bayesian.Domain((a, b, c)) >= bayesian.Domain((a, b)))
        self.assertTrue(bayesian.Domain((a, b, c)) >= bayesian.Domain((c, a)))
        self.assertTrue(bayesian.Domain((a, b, c)) >= bayesian.Domain((b,)))

        self.assertFalse(bayesian.Domain((a, b)) >= bayesian.Domain((c,)))
        self.assertFalse(bayesian.Domain((a, b)) >= bayesian.Domain((a, c)))

    def test_le(self):
        """Test the __le__ method"""

        # A domain is less than or equal to another if its variables
        # are a subset of the other.
        a = bayesian.Variable('a', 2)
        b = bayesian.Variable('b', 2)
        c = bayesian.Variable('c', 2)

        self.assertTrue(bayesian.Domain((a, b)) <= bayesian.Domain((a, b, c)))
        self.assertTrue(bayesian.Domain((c, a)) <= bayesian.Domain((a, b, c)))
        self.assertTrue(bayesian.Domain((b,)) <= bayesian.Domain((a, b, c)))

        self.assertFalse(bayesian.Domain((c,)) <= bayesian.Domain((a, b)))
        self.assertFalse(bayesian.Domain((a, c)) <= bayesian.Domain((a, b)))

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
