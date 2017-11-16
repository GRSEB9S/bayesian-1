from functools import reduce
from itertools import chain, combinations

from recur import preorder, postorder
from recur.tree import Tree
from recur.graphs import Undirected as Node

import bayesian.tables


class DomainGraph(Node):

    def __init__(self, variable):
        """Node of a domain graph which represents a variable"""

        super().__init__()
        self.variable = variable

    @staticmethod
    def from_tables(tables):
        """Builds domain graphs from a list of tables"""

        # Get the unique variables of the graph and create a node for
        # each of them.
        variables = list(set(v for t in tables for v in t.domain))
        nodes = [DomainGraph(v) for v in variables]

        # Add the link. Because the nodes are Undirected, every add creates
        # a symmetric link.
        for table in tables:
            for n1, n2 in combinations(table.domain, 2):
                nodes[variables.index(n1)].add(nodes[variables.index(n2)])

        # It is possible for tables to generate disconnected graphs. Here we
        # select one node for every one.
        graphs = []
        while len(nodes) > 0:
            graphs.append(nodes[0])
            for node in graphs[-1]:
                nodes.remove(node)

        return graphs


class JunctionTree(Tree):

    def __init__(self, variables, boundary, tables):

        super().__init__()
        self.variables = variables
        self.boundary = boundary
        self.tables = tables
        self.collect = None
        self.distribute = None

    @staticmethod
    def from_tables(tables):
        """Build a junction tree from an undirected connected graph"""

        remaining_tables = list(tables)

        junction_trees = []
        for graph in DomainGraph.from_tables(tables):
            nodes = []
            while graph is not None:

                simplicial = graph.simplicial
                if simplicial is None:

                    # Find the node with the smallest number of missing links and
                    # add them.
                    minimal = graph.minimize(lambda n: n.nb_missing_links)
                    for n1, n2 in combinations(minimal.neighbors, 2):
                        n1.add(n2)
                    simplicial = minimal

                family = simplicial.family
                boundary = simplicial.boundary
                to_remove = family - boundary

                # Get the tables that contain the nodes to remove.
                vars_to_remove = set(n.variable for n in to_remove)
                node_tables = [t for t in remaining_tables
                               if len(vars_to_remove & set(t.domain)) != 0] 
                remaining_tables = [t for t in remaining_tables
                                    if t not in node_tables]

                # Remove the nodes that are not on the boundary.
                for node in to_remove:
                    node.remove()

                nodes.append(JunctionTree(set(n.variable for n in family),
                                          set(n.variable for n in boundary),
                                          node_tables))
                graph = next((n for n in boundary), None)

            for child, parent in combinations(nodes, 2):
                if child.boundary <= parent.variables:
                    parent.add(child)

            junction_trees.append(nodes[-1])

        return junction_trees

    @postorder('method')
    def _collect(self):

        if self.parent is not None:

            # Get the tables of the current node and the collect ones
            # from the children.
            tables = self.tables + [n.collect for n in self.children]

            table = reduce(bayesian.tables.Product, tables)

            # Marginalize all variables that are not in the boundary.
            for variable in self.variables - self.boundary:
                table = bayesian.tables.Marginal(table, variable)

            self.collect = table

    @preorder
    def _distribute(self):

        for child in self.children:

            # Get the table from the node, it distribute table from the
            # parent and the collect tables from the children.
            children = (c for c in self.children if c is not child)
            tables = self.tables + [c.collect for c in children]

            if self.distribute is not None:
                tables.append(self.distribute)

            table = reduce(bayesian.tables.Product, tables)

            # Marginalize all variables that are not in the boundary of
            # the child.
            for variable in self.variables - child.boundary:
                table = bayesian.tables.Marginal(table, variable)

            child.distribute = table

    @postorder('method')
    def _marginals(self, marginals, visited):

        new_variables = self.variables - self.boundary - visited
        if len(new_variables) > 0:

            visited |= new_variables

            tables = self.tables + [c.collect for c in self.children]
            if self.distribute is not None:
                tables.append(self.distribute)

            table = reduce(bayesian.tables.Product, tables)

            for variable in new_variables:

                marginal = table
                for to_marginalize in self.variables - {variable}:
                    marginal = bayesian.tables.Marginal(marginal, to_marginalize)
                
                marginals.append(marginal)

    @postorder('method')
    def _simplify(self):
        self.tables = [reduce(bayesian.tables.Product, self.tables)]

def super_product(marginal):

    tables = []
    for table in marginal:
        if type(table) is bayesian.tables.Probability:
            tables.append(table)

    result = reduce(bayesian.tables.Product, tables)
    result.update()
    return result

def source_tables(marginal):
    
    tables = []
    for table in marginal:
        if type(table) is bayesian.tables.Probability:
            tables.append(table)

    return tables

def marginalize(tables):
    """Returns the marginal of all variables"""

    junction_trees = JunctionTree.from_tables(tables)

    all_marginals = [[] for _ in junction_trees] 
    for junction_tree, marginals in zip(junction_trees, all_marginals):
        junction_tree._simplify()
        junction_tree._collect()
        junction_tree._distribute()
    
        visited = set()
        junction_tree._marginals(marginals, visited)

        for marginal in marginals:
            marginal.update()

    return all_marginals
