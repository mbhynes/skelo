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

import pytest

import numpy as np
import pandas as pd

from skelo.model.elo import EloModel, EloEstimator
import skelo.utils.elo_data as data_utils


class TestEloModel(object):
  
  def test_init(self):
    model = EloModel().add('a').add('b')
    assert model.ratings == {
      'a': [{'rating': 1500, 'valid_from': 0, 'valid_to': None}],
      'b': [{'rating': 1500, 'valid_from': 0, 'valid_to': None}],
    }
    assert model.k(0) == model.default_k
    assert model.k(1000) == model.default_k

  def test_get(self):
    model = (
      EloModel().add('a').add('b')
      .update('a', 'b', 2)
    )
    assert model.get('a', 0) == {'rating': 1500.0, 'valid_from': 0, 'valid_to': 2}
    assert model.get('a', 1) == {'rating': 1500.0, 'valid_from': 0, 'valid_to': 2}
    assert model.get('a', 2, strict_past_data=True) == {'rating': 1500.0, 'valid_from': 0, 'valid_to': 2}
    assert model.get('a', 2, strict_past_data=False) == {'rating': 1510.0, 'valid_from': 2, 'valid_to': None}
    assert model.get('a') == {'rating': 1510.0, 'valid_from': 2, 'valid_to': None}

  def test_update(self):
    model = (
      EloModel().add('a').add('b')
      .update('a', 'b', 2)
    )
    expected = {
      'a': [
        {'rating': 1500, 'valid_from': 0, 'valid_to': 2},
        {'rating': 1510.0, 'valid_from': 2, 'valid_to': None},
      ],
      'b': [
        {'rating': 1500, 'valid_from': 0, 'valid_to': 2},
        {'rating': 1490.0, 'valid_from': 2, 'valid_to': None},
      ],
    }
    key = lambda r: r['valid_from']
    assert sorted(model.ratings['a'], key=key) == sorted(expected['a'], key=key)
    assert sorted(model.ratings['b'], key=key) == sorted(expected['b'], key=key)

  def test_transform(self):
    model = (
      EloModel().add('a').add('b')
      .update('a', 'b', 2)
      .update('a', 'b', 3)
      .update('a', 'b', 3)
    )
    assert model.transform('a', 'b', timestamp=-1) == (1500, 1500)
    assert model.transform('a', 'b', timestamp=1) == (1500, 1500)
    assert model.transform('a', 'b', timestamp=2) == (1500, 1500)
    assert model.transform('a', 'b', timestamp=2, strict_past_data=False) == (1510, 1490)
    assert model.transform('a', 'b', timestamp=3) == (1510, 1490)
    assert tuple([round(r) for r in model.transform('a', 'b', timestamp=3, strict_past_data=False)]) == (1528, 1472)
    assert tuple([round(r) for r in model.transform('a', 'b', timestamp=4)]) == (1528, 1472)

  def test_predict_proba(self):
    model = (
      EloModel().add('a').add('b')
    )
    assert model.predict_proba('a', 'b') == 0.5
    assert model.predict_proba(['a', 'b'], ['b', 'a'], timestamp=None) == [0.5, 0.5]

    model.update('a', 'b', 2)
    assert model.predict_proba('a', 'b', timestamp=-1) == 0.5

    expected_future_prob = 1.0 / (1 + 10**((1490 - 1510) / 400.0))
    assert model.predict_proba('a', 'b', timestamp=2, strict_past_data=True) == 0.5
    assert model.predict_proba('a', 'b', timestamp=2, strict_past_data=False) == expected_future_prob
    assert model.predict_proba('a', 'b', timestamp=2.001) == expected_future_prob

    expected = [
      1.0 / (1 + 10**((1490 - 1510) / 400.0)),
      1.0 / (1 + 10**((1510 - 1490) / 400.0)),
    ]
    assert model.predict_proba(['a', 'b'], ['b', 'a'], timestamp=[3, 3]) == expected

  def test_to_frame(self):
    model = (
      EloModel().add('a').add('b').update('a', 'b', 2)
    )
    df = model.to_frame()
    expected = pd.DataFrame({
      'key':        ['a', 'a', 'b', 'b'],
      'rating':     [1500.0, 1510.0, 1500.0, 1490.0],
      'valid_from': [0, 2, 0, 2],
      'valid_to':   [2, None, 2, None],
    })
    sort_key = ['key', 'valid_from']
    is_equal = (df.sort_values(sort_key).fillna(np.inf) == expected.sort_values(sort_key).fillna(np.inf))
    assert is_equal.fillna(True).all(axis=None)


class TestEloEstimator:

  def test_fit_ndarray(self):
    num_players = 4
    ratings = data_utils.generate_constant_ratings(num_players=num_players, sigma=40)
    games = pd.DataFrame(data_utils.generate_constant_game_outcomes(ratings, num_timesteps=100))
    X = games.values[:, :3] # key1, key2, timestamp
    y = games.values[:, -1] # outcome
    estimator = EloEstimator().fit(X, y)
    for p in range(num_players):
      fit_rating = estimator.rating_model.get(p, timestamp=None)['rating']
      assert np.abs(fit_rating - ratings[p]) < estimator.rating_model.default_k

  def test_fit_dataframe(self):
    num_players = 4
    ratings = data_utils.generate_constant_ratings(num_players=num_players, sigma=40)
    games = pd.DataFrame(data_utils.generate_constant_game_outcomes(ratings, num_timesteps=100))
    estimator = EloEstimator('p1', 'p2', 'match_at').fit(games, games['label'])
    for p in range(num_players):
      fit_rating = estimator.rating_model.get(p, timestamp=None)['rating']
      assert np.abs(fit_rating - ratings[p]) < estimator.rating_model.default_k

  def test_transform_dataframe(self):
    num_players = 2
    ratings = data_utils.generate_constant_ratings(num_players=num_players, sigma=40)
    games = pd.DataFrame(data_utils.generate_constant_game_outcomes(ratings, num_timesteps=10))
    estimator = EloEstimator('p1', 'p2', 'match_at').fit(games, games['label'])

    transformed_ratings = estimator.transform(games, output_type='rating')
    for k, (_, g) in enumerate(games.iterrows()):
      r1_expected = estimator.rating_model.get(g['p1'], timestamp=g['match_at'])['rating']
      r2_expected = estimator.rating_model.get(g['p2'], timestamp=g['match_at'])['rating']
      assert r1_expected == transformed_ratings.iloc[k]['r1']
      assert r2_expected == transformed_ratings.iloc[k]['r2']

    transformed_prob = estimator.transform(games, output_type='prob')
    for k, (_, g) in enumerate(games.iterrows()):
      r1_expected = estimator.rating_model.get(g['p1'], timestamp=g['match_at'])['rating']
      r2_expected = estimator.rating_model.get(g['p2'], timestamp=g['match_at'])['rating']
      prob = EloModel.compute_prob(r1_expected, r2_expected)
      assert transformed_prob.iloc[k] == EloModel.compute_prob(r1_expected, r2_expected)

  def test_transform_ndarray(self):
    num_players = 2
    ratings = data_utils.generate_constant_ratings(num_players=num_players, sigma=40)
    games = pd.DataFrame(data_utils.generate_constant_game_outcomes(ratings, num_timesteps=10))
    X = games.values[:, :3] # key1, key2, timestamp
    y = games.values[:, -1] # outcome
    estimator = EloEstimator().fit(X, y)

    transformed_ratings = estimator.transform(X, output_type='rating')
    for k, (_, g) in enumerate(games.iterrows()):
      r1_expected = estimator.rating_model.get(g['p1'], timestamp=g['match_at'])['rating']
      r2_expected = estimator.rating_model.get(g['p2'], timestamp=g['match_at'])['rating']
      assert r1_expected == transformed_ratings[k, 0]
      assert r2_expected == transformed_ratings[k, 1]

    transformed_prob = estimator.transform(X, output_type='prob')
    for k, (_, g) in enumerate(games.iterrows()):
      r1_expected = estimator.rating_model.get(g['p1'], timestamp=g['match_at'])['rating']
      r2_expected = estimator.rating_model.get(g['p2'], timestamp=g['match_at'])['rating']
      prob = EloModel.compute_prob(r1_expected, r2_expected)
      assert transformed_prob[k] == prob

  def test_predict_proba_dataframe(self):
    num_players = 2
    games = pd.DataFrame({
      'p1': [1],
      'p2': [2],
      'match_at': [2],
      'label': [1],
    })
    estimator = EloEstimator('p1', 'p2', 'match_at').fit(games, games['label'])
    y_pred = estimator.predict_proba(np.array([[1, 2, 2], [1, 2, 3], [2, 1, 3]]))

    pr = EloModel.compute_prob(1510, 1490)
    assert np.allclose(y_pred, np.array([[0.5, 0.5], [pr, 1 - pr], [1 - pr, pr]]))

  def test_predict_proba_ndarray(self):
    num_players = 2
    games = pd.DataFrame({
      'p1': [1],
      'p2': [2],
      'match_at': [2],
      'label': [1],
    })
    X = games.values[:, :3] # key1, key2, timestamp
    y = games.values[:, -1] # outcome
    estimator = EloEstimator().fit(X, y)
    y_pred = estimator.predict_proba(np.array([[1, 2, 2], [1, 2, 3], [2, 1, 3]]))

    pr = EloModel.compute_prob(1510, 1490)
    assert np.allclose(y_pred, np.array([[0.5, 0.5], [pr, 1 - pr], [1 - pr, pr]]))
