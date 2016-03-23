import unittest
import bayesian

class TestUsingProblems(unittest.TestCase):
    
    def test_car_start_problem(self):
        """Test the bayesian module using the car start problem

        This test is taken from page 35 of:

        Jensen, Finn V. and Nielsen, Thomas D., Bayesian Network and Decision 
        Graph, Springer New York, 2007.

        """

        # Fu : Does the car have fuel?
        # FM : What does the fuel meter say?
        # St : Does the car start?
        # Sp : Are the spark plugs clean?
        Fu = bayesian.Variable('Fu')
        Fm = bayesian.Variable('Fm', 3)
        St = bayesian.Variable('St')
        Sp = bayesian.Variable('Sp')

        PFu = bayesian.Table([Fu], [0.02, 0.98])
        PSp = bayesian.Table([Sp], [0.04, 0.96])
        PFm = bayesian.Table([Fu, Fm], [
            [0.998, 0.001, 0.001], 
            [0.01, 0.60, 0.39]
        ])
        PSt = bayesian.Table([St, Fu, Sp], [
            [[1.0, 1.0], [0.99, 0.01]],
            [[0.0, 0.0], [0.01, 0.99]]
        ])

        bn = bayesian.Network()
        bn.add_table(PFu)
        bn.add_table(PSp)
        bn.add_table(PFm)
        bn.add_table(PSt)
       
        # Add evidence for the St variable.
        ESt = bayesian.Table([St], [1.0, 0.0])
        bn.add_table(ESt)

        # Verify the result agree with the reference.
        self.assertAlmostEqual(bn.marginal(Fu)[0], 0.29, 2)
        self.assertAlmostEqual(bn.marginal(Fu)[1], 0.71, 2)
        self.assertAlmostEqual(bn.marginal(Sp)[0], 0.58, 2)
        self.assertAlmostEqual(bn.marginal(Sp)[1], 0.42, 2)

        # Add evidence for the Fm variable.
        EFm = bayesian.Table([Fm], [0.0, 1.0, 0.0]) 
        bn.add_table(EFm)

        # Verify the result agree with the reference.
        self.assertAlmostEqual(bn.marginal(Sp)[0], 0.804, 2)
        self.assertAlmostEqual(bn.marginal(Sp)[1], 0.196, 2)
        self.assertAlmostEqual(bn.marginal(Fu)[0], 0.001, 2)
        self.assertAlmostEqual(bn.marginal(Fu)[1], 0.999, 2)

if __name__ == '__main__':
    unittest.main()
