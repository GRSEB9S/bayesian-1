import cProfile as profile
import pstats
import StringIO

import bayesian
import bayesian.tests.networks


def junction_marginals():
    """Profile the marginals method using a dynamic network"""

    # Generate the network.
    network = bayesian.tests.networks.Dynamic(1000)

    # Profile the computation of the marginals.
    junction_tree = bayesian.JunctionTree(network)
    profiler = profile.Profile()
    profiler.enable()
    marginals = junction_tree.marginals()
    profiler.disable()

    # Print the stats.
    stream = StringIO.StringIO()
    profile_stats = pstats.Stats(profiler, stream=stream)
    profile_stats.strip_dirs().sort_stats('cumulative')
    profile_stats.print_stats()
    print(stream.getvalue())

if __name__ == "__main__":
    junction_marginals()
