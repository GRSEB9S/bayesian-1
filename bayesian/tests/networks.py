import numpy as np

import bayesian


class CarStartProblem(bayesian.Network):
    def __init__(self):
        """Bayesian network implementing the car start problem

        This test is taken from page 35 of:

        Jensen, Finn V. and Nielsen, Thomas D., Bayesian Network and Decision
        Graph, Springer New York, 2007.

        """

        bayesian.Network.__init__(self)

        # fuel  : Does the car have fuel?
        # meter : What does the fuel meter say?
        # start : Does the car start?
        # spark : Are the spark plugs clean?
        fuel = bayesian.Variable('Fu')
        meter = bayesian.Variable('Fm', 3)
        start = bayesian.Variable('St')
        spark = bayesian.Variable('Sp')

        # It is very likely the car has fuel.
        fuel_table = bayesian.Table([fuel], [0.02, 0.98])

        # It is very likely the spark plug is clean.
        spark_table = bayesian.Table([spark], [0.04, 0.96])

        # The meter depends on the fuel.
        meter_table = bayesian.Table([fuel, meter], [
            [0.998, 0.001, 0.001],
            [0.01, 0.60, 0.39]
        ])

        # The car will start if there is fuel and the spark plug is
        # clean.
        start_table = bayesian.Table([start, fuel, spark], [
            [[1.0, 1.0], [0.99, 0.01]],
            [[0.0, 0.0], [0.01, 0.99]]
        ])

        self.add_table(fuel_table)
        self.add_table(spark_table)
        self.add_table(meter_table)
        self.add_table(start_table)


class Pyramid(bayesian.Network):
    def __init__(self):
        """Pyramid Bayesian network used for tests

        This class represents a pyramid network with the following shape.

              A
             / \
            /   \
           B     C
          / \   / \
         /   \ /   \
        D     E     F

        Evidence is added to the node F.

        """

        # Generate the network with no evidence.
        bayesian.Network.__init__(self)

        A = bayesian.Variable('A')
        B = bayesian.Variable('B')
        C = bayesian.Variable('C')
        D = bayesian.Variable('D')
        E = bayesian.Variable('E')
        F = bayesian.Variable('F')

        table_A = bayesian.Table([A], [0.55, 0.45])
        table_BA = bayesian.Table([A, B], [[0.1, 0.9], [0.9, 0.1]])
        table_CA = bayesian.Table([A, C], [[0.25, 0.75], [0.35, 0.65]])
        table_DB = bayesian.Table([B, D], [[0.40, 0.60], [0.65, 0.35]])
        table_FC = bayesian.Table([C, F], [[0.99, 0.01], [0.03, 0.97]])
        table_EBC = bayesian.Table([B, C, E], [
            [[0.54, 0.46], [0.37, 0.63]],
            [[0.11, 0.89], [0.27, 0.73]]
        ])

        self.add_table(table_A)
        self.add_table(table_BA)
        self.add_table(table_CA)
        self.add_table(table_DB)
        self.add_table(table_FC)
        self.add_table(table_EBC)

        # Add the evidence.
        evidence_F = bayesian.Table([F], [100.0, 15.0])
        self.add_table(evidence_F)


class Dynamic(bayesian.Network):
    def __init__(self, nb_steps=2):
        """Dynamic Bayesian network with varying length

        This class represents a dynamic Bayesian network with a number of
        steps which can be selected on construction.

           A0---------A1---- ...
          / \        / \
         /   \      /   \
        B0   C0    B1   C1
         \_________/\________ ...

        """

        bayesian.Network.__init__(self)

        # Generate variables.
        a = [bayesian.Variable('A{}'.format(i)) for i in range(nb_steps)]
        b = [bayesian.Variable('B{}'.format(i)) for i in range(nb_steps)]
        c = [bayesian.Variable('C{}'.format(i)) for i in range(nb_steps)]

        # Generate the tables and add them to the network.
        for i in range(nb_steps - 1):
            table = bayesian.Table(a[i:i+2], np.random.rand(2,2))
            self.add_table(table)
            table = bayesian.Table(b[i:i+2], np.random.rand(2,2))
            self.add_table(table)
