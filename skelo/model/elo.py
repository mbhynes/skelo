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

import logging

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin

from skelo.model.base import RatingModel, RatingEstimator

logger = logging.getLogger(__name__)


class EloModel(RatingModel):
  """
  Dictionary-based implementation of the `Elo rating system <https://en.wikipedia.org/wiki/Elo_rating_system>`_.

  This class creates a dictionary of Elo ratings for each player inserted into the rating system, such
  that each match update will append new ratings for the respective match players, calculated according
  to the Elo update formula.

  This model may be used directly, but is primarily intended as a utility class for an EloEstimator.
  """

  def __init__(self, default_k=20, k_fn=None, initial_value=1500, initial_time=0, **kwargs):
    """
    Construct an Elo RatingsModel.

    Args:
      default_k (int): default value of `k` to use in the Elo update formula if no `k_fn` is provided
      k_fn (callable): univariate function of a rating that returns a value of `k` for updates
      initial_value (int): initial default rating value to assign to a new player in the system
      initial_time (int or orderable): the earliest "time" value for matches between players.
    """
    super().__init__(initial_value=float(initial_value), initial_time=initial_time)
    self.default_k = default_k
    self.k = k_fn or (lambda _: default_k)

  def evolve_rating(self, r1, r2, label):
    exp = self.compute_prob(r1, r2)
    return r1 + self.k(r1) * (label - exp)

  @staticmethod
  def compute_prob(r1, r2):
    """
    Return the probability of a player with rating r1 beating a player with rating r2.
    """
    diff = (r2 - r1) / 400.0
    prob = 1.0 / (1 + 10**diff)
    return prob


class EloEstimator(RatingEstimator):
  """
  A scikit-learn Classifier implementing the `Elo rating system <https://en.wikipedia.org/wiki/Elo_rating_system>`_.
  """
  RATING_MODEL_CLS = EloModel
  RATING_MODEL_ATTRIBUTES = [
    'default_k',
    'k_fn',
    'initial_value',
    'initial_time',
  ]

  def __init__(self, key1_field=None, key2_field=None, timestamp_field=None, default_k=20, k_fn=None, initial_value=1500, initial_time=0, **kwargs):
    """
    Construct a classifier object, without fitting it.
    
    Args:
      key1_field (string): column name of the player1 key, if fit on a pandas DataFrame
      key2_field (string): column name of the player2 key, if fit on a pandas DataFrame
      timestamp_field (string): column name of the timestamp field, if fit on a pandas DataFrame
    """
    super().__init__(
      key1_field=key1_field,
      key2_field=key2_field,
      timestamp_field=timestamp_field,
      initial_value=initial_value,
      initial_time=initial_time,
      **kwargs
    )
    self.default_k = default_k
    self.k_fn = k_fn
