import bayesian

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
