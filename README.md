# skelo

An implementation of the [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system) with a [scikit-learn](https://scikit-learn.org/stable/)-compatible interface.

The `skelo` package is a simple implementation suitable for small-scale ratings systems that fit into memory on a single machine.
It's intended to provide a simple API for creating elo ratings in a small games on the scale thousands of players and millions of matches, primarily as a means of feature transformation for inclusion in other `sklearn` pipelines.

## Installation

- Installation via the PyPI package using pip:
```python
pip3 install skelo
```

## Example Usage

### Synthetic Data

We can use a convenience utility for generating match data from players with normally distributed ratings.
The below example shows a simple usage of this utility:
```python
import pandas as pd
import matplotlib.pyplot as plt

import skelo.utils.elo_data as data_utils
from skelo.model.elo import EloEstimator

# Generate some sample constant ratings & match data
ratings = data_utils.generate_constant_ratings(num_players=10, sigma=500)
matches = pd.DataFrame(data_utils.generate_constant_game_outcomes(ratings, num_timesteps=100))

# Fit the model using numpy arrays
X = matches.values[:, :3] # player1, player2, match timestamp
y = matches.values[:, -1] # match outcome
model = EloEstimator().fit(X, y)

# Get a dataframe of the estimated ratings over time from the fitted model
ratings_est = model.elo.to_frame()

# Compare the ratings estimate over time
ts_est = ratings_est.pivot_table(index='valid_from', columns='key', values='rating')
ts_est.plot()
```
[Synthetic Ratings Convergence](examples/ratings_convergence.png)


The estimated ratings will exhibit convergence profiles (players with extremal low or high ratings take longer to converge).
Please note that while the actual original ratings are unlikely to be determined by the fitting procedure, the *relative* difference between the ratings should be preserved, within the noise band of the chosen value of `k` (by default: 20)


## Developer Guide

### `dev` scripts

The `dev` script (and other scripts in `bin`) contain convenience wrappers for some common development tasks:

- **Installing Dependencies**. To set up a local development environment with dependencies, use:
  ```bash
  ./dev up
  ```
  This will create a `.venv` directory in the project root and install the required python packages.

- **Running Tests**. To run the project tests with the virtual environment's `pytest`, use:

  ```python
  ./dev test
  ```
  Arguments to `pytest` can be supplied to this script as well, to e.g. limit the tests to a particular subset.

- **Packaging for PyPI**. The package-building and upload process is wrapped with the following commands:

  ```python
  ./dev package
  ./dev upload --test 
  ./dev upload
  ```
