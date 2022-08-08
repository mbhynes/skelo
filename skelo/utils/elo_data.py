# MIT License
#
# Copyright (c) 2022 Michael B Hynes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

from skelo.model.elo import EloModel

def sigmoid(x):
  return 0.5 * (np.tanh(2*x) + 1)

def sigmoid_k_fn_builder(r1, r2, k1, k2):
  """
  Build an function that returns an Elo k value that is univariate in the rating value.
  This function builder creates a sigmoid-shaped function that decreases approximately
  linearly from value `k1` to `k2` between ratings values `r1` to `r2` (`r2 > r1`).
  
  The function shape is illustrated in the schematic below::

             r1
    k1 _______ 
              \\
               \\
                \\________ k2
                r2

  Such a function shape provides a decreasing value of k in accordance with the
  recommendation by Arpad Elo that better (i.e. higher ranked) players have a lower
  value of `k` and therefore less volatility in their ratings updates after a match.
  """
  assert r2 > r1, f"r2 must be greater than r1; received: r2 ({r2}) <= r1 ({r1})"
  midpoint = (r1 + r2) / 2.0
  slope = midpoint / 2.0 
  return lambda r: k1 - (k1 - k2) * sigmoid((r - midpoint) / slope)

def generate_ratings(num_players, num_timesteps, mu=1500, sigma=1, seed=1):
  """
  Create an array of player skill ratings that follow a random walk at each timestamp.
  The 2-dimensional array returned will have shape (num_players, num_timesteps),
  where each row represents the timeseries of skill for that respective player.

  Args:
    num_players (int): the number of players to simulate
    num_timesteps (int): the number of players to simulate
    mu (float): the

  Returns:
    A (num_player, num_timesteps) array of ratings
  """
  assert type(num_players) is int and num_players > 0
  assert type(num_timesteps) is int and num_timesteps > 0
  np.random.seed(seed)
  drift = sigma * np.random.normal(size=(num_players, num_timesteps)).cumsum(axis=1)
  # Remove the mean of the noise from the first timestamp,
  # such that the initial ratings have a mean of mu.
  centered_drift = drift - drift.mean(axis=0)[0]
  return np.maximum(0, mu + centered_drift)

def generate_game_outcomes(ratings, seed=1):
  """
  Generate a set of games (player x player) interactions and outcomes based on
  the provided skills of each player.
  """
  np.random.seed(seed)
  (num_players, num_timesteps) = ratings.shape
  matches = []
  for ts in range(num_timesteps):
    index = 0
    for p1 in range(num_players):
      for p2 in range(p1 + 1, num_players):
        prob = EloModel.compute_prob(ratings[p1, ts], ratings[p2, ts]) 
        matches.append({
          'p1': p1,
          'p2': p2,
          'match_at': ts,
          'index': index,
          'label': int(np.random.rand() <= prob),
        })
        index += 1
  return matches

def generate_constant_ratings(num_players, mu=1500, **kwargs):
  return generate_ratings(num_players, num_timesteps=1, **kwargs)

def generate_constant_game_outcomes(ratings, num_timesteps, **kwargs):
  ratings = np.tile(ratings[:, 0], (num_timesteps, 1)).T
  return generate_game_outcomes(ratings, **kwargs)
