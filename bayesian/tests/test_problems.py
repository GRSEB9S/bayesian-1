import unittest
import bayesian
import bayesian.tests.networks


class TestUsingProblems(unittest.TestCase):

    def test_car_start_problem(self):
        """Test the bayesian module using the car start problem"""

        # Generate the network for the car start problem.
        network = bayesian.tests.networks.CarStartProblem()

        # Get the start and fuel meter variables.
        variables = network.domain
        symbols = [variable.symbol for variable in variables]
        start = variables[symbols.index('St')]
        fuel = variables[symbols.index('Fu')]
        spark = variables[symbols.index('Sp')]
        meter = variables[symbols.index('Fm')]

        # Add evidence for the St variable.
        evidence_start = bayesian.Table([start], [1.0, 0.0])
        network.add_table(evidence_start)

        # Verify the result agree with the reference.
        fuel_marginal = network.marginal(fuel)
        fuel_marginal.normalize()
        self.assertAlmostEqual(fuel_marginal[0], 0.29, 2)
        self.assertAlmostEqual(fuel_marginal[1], 0.71, 2)
        spark_marginal = network.marginal(spark)
        spark_marginal.normalize()
        self.assertAlmostEqual(spark_marginal[0], 0.58, 2)
        self.assertAlmostEqual(spark_marginal[1], 0.42, 2)

        # Add evidence for the Fm variable.
        evidence_meter = bayesian.Table([meter], [0.0, 1.0, 0.0])
        network.add_table(evidence_meter)

        # Verify the result agree with the reference.
        spark_marginal = network.marginal(spark)
        spark_marginal.normalize()
        self.assertAlmostEqual(spark_marginal[0], 0.804, 2)
        self.assertAlmostEqual(spark_marginal[1], 0.196, 2)
        fuel_marginal = network.marginal(fuel)
        fuel_marginal.normalize()
        self.assertAlmostEqual(fuel_marginal[0], 0.001, 2)
        self.assertAlmostEqual(fuel_marginal[1], 0.999, 2)

if __name__ == '__main__':
    unittest.main()
