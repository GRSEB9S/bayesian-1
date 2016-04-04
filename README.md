# bayesian
Python package to compute marginal distributions in Bayesian networks.

## Quickstart

Here is a simple example taken from [wikipedia](https://en.wikipedia.org/wiki/Bayesian_network) to get you started.
We are interested in the probability that it rained, given that the grass is wet.
First, we define the three variables of the model. The first variable indicates if the grass is wet, the second indicates if the sprinkler is on, and the third indicates if it is raining.

```python
import bayesian

# For all variables, we define state 0 as no and state 1 as yes.
# Is the grass wet? 
grass = bayesian.Variable('grass')

# Are the sprinklers on?
sprinkler = bayesian.Variable('sprinkler')

# Is it raining?
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

The next step is to generate the Bayesian network.
To test it, we evaluate the marginal probability that the grass is wet. 

```python
network = bayesian.Network()
network.add_table(rain_table)
network.add_table(sprinkler_table)
network.add_table(grass_table)

grass_marginal = network.marginal(grass)
print(grass_marginal)
```

Finally, we add the evidence (we know that the grass is wet) which is simply another probability table.
We then evaluate the probability that it rained, given that the grass is wet.

```python
evidence = bayesian.Table([grass], [0.0, 1.0])
network.add_table(evidence)
rain_marginal = network.marginal(rain)
print(rain_marginal)
```




