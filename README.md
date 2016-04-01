# bayesian
Python package to compute marginal distributions in Bayesian networks.

## Quickstart

Here is a simple example take from [wikipedia](https://en.wikipedia.org/wiki/Bayesian_network) to get you started.
First, we define three variables. The first variable indicates if the grass is wet, the second indicates if the sprinkler is on, and the third indicates if it is raining.

```python
import bayesian

grass = bayesian.Variable('grass')
sprinkler = bayesian.Variable('sprinkler')
rain = bayesian.Variable('rain')
```

Next, we define the conditional probability tables of the problem.

```python
rain_table = bayesian.Table([rain], [0.8, 0.2])
sprinkler_table = bayesian.Table([rain, sprinkler], [[0.6, 0.4], [0.99, 0.01]])
grass_table = bayesian.Table([sprinkler, rain, grass], [
    [[1.0, 0.0], [0.2, 0.8]],
    [[0.1, 0.9], [0.01, 0.99]]
])
```

Finally, we generate the Bayesian network and evaluate the marginal probability that the grass is wet.

```python
network = bayesian.Network()
network.add_table(rain_table)
network.add_table(sprinkler_table)
network.add_table(grass_table)

grass_marginal = network.marginal(grass)
print(grass_marginal)
```




