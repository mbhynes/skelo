# skelo

An implementation of the [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system) with a [scikit-learn](https://scikit-learn.org/stable/)-compatible interface.

The `skelo` package is a simple implementation suitable for small-scale ratings systems that fit into memory on a single machine.
It's intended to provide a simple API for creating elo ratings in a small games on the scale thousands of players and millions of matches, primarily as a means of feature transformation for inclusion in other `sklearn` pipelines.

## Motivation

What problem does this package solve?

Despite there being many ratings systems implementations available (e.g. [elo](https://github.com/sublee/elo/) [Elo](https://github.com/ddm7018/Elo), [elo](https://github.com/rshk/elo), [EloPy](https://github.com/HankSheehan/EloPy), [PythonSkills](https://github.com/McLeopold/PythonSkills)) it's hard to find one that satisfies several criteria for ease of use:
  - A simple and clean API that's convenient for a data-driven model development loop, for which use case the scikit-learn estimator [interface](https://scikit-learn.org/stable/modules/classes.html) is the *de facto* standard
  - Explicit management of intervals of validity for ratings, such that as matches occur a timeseries of ratings is evolved for each players (i.e. type-2 data management as opposed to type-1 fire-and-forget ratings)

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
![Convergence of Synthetic Ratings](https://raw.githubusercontent.com/mbhynes/skelo/main/examples/ratings_convergence.png)

The estimated ratings will exhibit convergence profiles (players with extremal low or high ratings take longer to converge).
Please note that while the actual original ratings are unlikely to be determined by the fitting procedure, the *relative* difference between the ratings should be preserved, within the noise band of the chosen value of `k` (by default: 20)

### Example Tennis Ranking

```python
import numpy as np
import pandas as pd
from skelo.model.elo import EloEstimator
from sklearn.metrics import precision_score

# Download a dataframe of example tennis data from JeffSackmann's ATP match repository (thanks Jeff!)
def load_data():
  df = pd.concat([
    pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_1979.csv"),
    pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_1980.csv"),
  ], axis=0)
  # Do some simple munging to get a date and a match outcome label
  df["tourney_date"] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')

  # For measuring classifier calibration linearity, it's better to have both true and false
  # labels in the dataset, so we relabel the order of (winner, loser) to just be (player1, player2)
  order_mask = (df["winner_name"] < df["loser_name"])
  df["p1"] = ""
  df["p2"] = ""
  df["label"] = 1
  df.loc[order_mask, "p1"] = df.loc[order_mask, "winner_name"]
  df.loc[~order_mask, "p1"] = df.loc[~order_mask, "loser_name"]
  df.loc[order_mask, "p2"] = df.loc[order_mask, "loser_name"]
  df.loc[~order_mask, "p2"] = df.loc[~order_mask, "winner_name"]
  df.loc[~order_mask, "label"] = 0
  return df

df = load_data()
player_counts = pd.concat([df["p1"], df["p2"]], axis=0).value_counts()
players = player_counts[player_counts > 10].index
mask = (df["p1"].isin(players) & df["p2"].isin(players))
X = df.loc[mask]

# Create a model to fit on a dataframe.
# Since our match data has np.datetime64 timestamps, we specify an initial time explicitly
model = EloEstimator(
  key1_field="p1",
  key2_field="p2",
  timestamp_field="tourney_date",
  initial_time=np.datetime64('1979', 'Y'),
).fit(X, X["label"])

#  Retrieve the fitted Elo ratings from the model
ratings_est = model.elo.to_frame()
ts_est = ratings_est.pivot_table(index='valid_from', columns='key', values='rating').ffill()

idx = ts_est.iloc[-1].sort_values().index[-5:]
ts_est.loc[:, idx].plot()
```

This should result in a figure like the one below, showing the 5 highest ranked (within the Elo system) players based on this subset of ATP matches:
![Top ATP Player Ratings, 1979-1980](https://raw.githubusercontent.com/mbhynes/skelo/main/examples/atp_1979.png)

### Example Tennis Ranking - using the `sklearn` API

While the ratings are mildly interesting to visualize, the predictive performance of the rating system's predictions have more practical importance.
For determining the performance of a classifier, the `sklearn` API and model utilities provide simple tools for us.
Below we calculate the classification metrics of the Elo system using only the 1980 data, where each prediction for a match uses only the outcomes of previous matches:

```python
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve

mask = (X["tourney_date"] > "1980-01-01")

print(classification_report(X.loc[mask, "label"], model.predict(X.loc[mask])))

prob_true, prob_pred = calibration_curve(
  X.loc[mask, "label"],
  model.predict_proba(X.loc[mask]).values[:, 0],
  n_bins=10
)
plt.plot(prob_pred, prob_true, label=f"Elo Classifier, k={model.elo.default_k}", marker='s', color='b')
plt.plot([0, 1], [0, 1], label="Perfect Calibration", ls=":", color='k')
plt.xlabel("Predicted Probability")
plt.ylabel("Empirical Probability")
plt.legend()
```

![Elo Calibration for ATP Matches, 1980](https://raw.githubusercontent.com/mbhynes/skelo/main/examples/atp_1979-calibration.png)

We can now also use the `sklearn` utilities for parameter tuning.
The below example trains several instances of the Elo ratings model with different values of `k` to find value that maximizes prediction accuracy during the ATP matches for 1980.
Please note that the `EloModel.predict()` method only uses past information available at each match, such that there is no leakage of information from the future in the model's forecasts.
```python
from sklearn.model_selection import GridSearchCV
import skelo.utils.elo_data as data_utils

clf = GridSearchCV(
  model,
  param_grid={
    'default_k': [
      10, 15, 20, 25, 30, 35, 40, 45, 50,
    ]
  },
  cv=[(X.index, X.index[len(X)//2:])],
).fit(X, X['label'])

results = pd.DataFrame(clf.cv_results_)
```

This should produce a result like the following:
```python
results.sort_values('rank_test_score').head(2).T

                                     7                  8
  mean_fit_time               0.041697            0.09369
  std_fit_time                     0.0                0.0
  mean_score_time             0.050558           0.067095
  std_score_time                   0.0                0.0
  param_default_k                   45                 50
  param_k_fn                       NaN                NaN
  params             {'default_k': 45}  {'default_k': 50}
  split0_test_score           0.679201           0.679201
  mean_test_score             0.679201           0.679201
  std_test_score                   0.0                0.0
  rank_test_score                    1                  1
```

## Development Guide

If you would like to contribute to this repository, please open an [issue](issues/new) first to document the extension or modification you're interested  in contributing.

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
