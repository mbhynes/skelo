.. skelo documentation master file, created by
   sphinx-quickstart on Sun Aug  7 22:20:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

`skelo`
=================================

The `skelo` package is an implementation of the `Elo <https://en.wikipedia.org/wiki/Elo_rating_system>`_ and `Glicko2 <https://en.wikipedia.org/wiki/Glicko_rating_system>`_ rating systems with a `scikit-learn <https://scikit-learn.org/stable>`_ compatible interface.

The `skelo` package is a simple implementation suitable for small-scale rating systems that fit into memory on a single machine.
It's intended to provide a convenient API for creating Elo/Glicko ratings in a data science & analytics workflow for small games on the scale thousands of players and millions of matches, primarily as a means of feature transformation in other `sklearn` pipelines or benchmarking classifier accuracy.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Motivation
~~~~~~~~~~

What problem does this package solve?

Despite there being many opensource rating system implementations available, it's hard to find one that satisfies several criteria:

- A simple and clean API that's convenient for a data-driven model development loop, for which use case the scikit-learn `estimator interface <https://scikit-learn.org/stable/modules/classes.html>`_ is the *de facto* standard
- Explicit management of intervals of validity for ratings, such that as matches occur a timeseries of ratings is evolved for each player (i.e. type-2 data management as opposed to type-1 fire-and-forget ratings)

This package addresses this gap by providing rating system implementations with:

- a simple interface for in-memory data management (i.e. storing the ratings as they evolve)
- time-aware ratings retrieval (i.e. *resolving* a player to their respective rating at an arbitrary point in time)
- scikit-learn classifier methods to interact with the predictions in a typical data science workflow

Installation
~~~~~~~~~~~~

Install via the PyPI package `skelo <https://pypi.org/project/skelo/>`_ using pip:

.. code-block:: bash

  pip3 install skelo

License
~~~~~~~
This project is released under the MIT license. Please see the LICENSE header in the source code for more details.


Quickstart
==========

As a quickstart, we can load and fit an `EloEstimator` (classifier) on some sample tennis data:

.. code-block:: python

  import numpy as np
  import pandas as pd
  from skelo.model.elo import EloEstimator

  df = pd.read_csv("https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_1979.csv")
  labels = len(df) * [1] # the data are ordered as winner/loser

  model = EloEstimator(
    key1_field="winner_name",
    key2_field="loser_name",
    timestamp_field="tourney_date",
    initial_time=19781231,
  ).fit(df, labels)

The ratings data are available as a `pandas DataFrame` if we wish to do any further analysis on it:

.. code-block::python

  >>> model.rating_model.to_frame()

                       key       rating  valid_from    valid_to
  0               Tim Vann  1500.000000    19781231  19790115.0
  1               Tim Vann  1490.350941    19790115         NaN
  2     Alejandro Gattiker  1500.000000    19781231  19790917.0
  3     Alejandro Gattiker  1492.529478    19790917  19791119.0
  4     Alejandro Gattiker  1485.415228    19791119         NaN
  ...                  ...          ...         ...         ...
  8462       Tom Gullikson  1483.914545    19791029  19791105.0
  8463       Tom Gullikson  1478.934755    19791105  19791105.0
  8464       Tom Gullikson  1489.400521    19791105  19791105.0
  8465       Tom Gullikson  1498.757080    19791105  19791113.0
  8466       Tom Gullikson  1490.600009    19791113         NaN
  ```

Once fit, we can transform a `DataFrame` or `ndarray` of player/player match data into the respective ratings for each player immediately *prior* to the match

.. code-block:: python

  >>> model.transform(df, output_type='rating')

                 r1           r2
  0     1598.787906  1530.008777
  1     1548.633423  1585.653196
  2     1548.633423  1598.787906
  3     1445.555739  1489.089241
  4     1439.595891  1502.254666
  ...           ...          ...
  3954  1872.284295  1714.108269
  3955  1872.284295  1698.007094
  3956  1837.623245  1714.108269
  3957  1837.623245  1698.007094
  3958  1698.007094  1714.108269

  [3959 rows x 2 columns]

Alternatively, we could also transform a datafrom into the forecast probabilities of victory for the player `"winner_name"`:

.. code-block:: python

  >>> model.transform(df, output_type='prob')

  0       0.597708
  1       0.446925
  2       0.428319
  3       0.437676
  4       0.410792
            ...
  3954    0.713110
  3955    0.731691
  3956    0.670624
  3957    0.690764
  3958    0.476845
  Length: 3959, dtype: float64

These probabilities are also available using the `predict_proba` or `predict` classifier methods, as shown below. What distinguishes `transform` from `predict_proba` is that `predict_proba` and `predict` return predictions that only use past data (i.e. you cannot cheat by leaking future data into the forecast), while `transform(X, strict_past_data=False)` may be used to compute ratings that "peek" into the future and could return ratings updated using match outcomes pushed (slightly) back in time to the match start timestamp. This is a specific convenience utility for non-forecasting use cases in which the match start time is a more convenient timestamp with which to index and manipulate data.

.. code-block:: python

  >>> model.predict_proba(df)

             pr1       pr2
  0     0.597708  0.402292
  1     0.446925  0.553075
  2     0.428319  0.571681
  3     0.437676  0.562324
  4     0.410792  0.589208
  ...        ...       ...
  3954  0.713110  0.286890
  3955  0.731691  0.268309
  3956  0.670624  0.329376
  3957  0.690764  0.309236
  3958  0.476845  0.523155

  [3959 rows x 2 columns]

  >>> model.predict(df)

  0       1.0
  1       0.0
  2       0.0
  3       0.0
  4       0.0
         ...
  3954    1.0
  3955    1.0
  3956    1.0
  3957    1.0
  3958    0.0
  Name: pr1, Length: 3959, dtype: float64

API Reference
=============

Rating Estimators
~~~~~~~~~~~~~~~~~~

.. autoclass:: skelo.model.RatingEstimator
   :members:

.. autoclass:: skelo.model.elo.EloEstimator
   :members:

.. autoclass:: skelo.model.glicko2.Glicko2Estimator
   :members:

Rating Models
~~~~~~~~~~~~~~~~~~

.. autoclass:: skelo.model.RatingModel
   :members:

.. autoclass:: skelo.model.elo.EloModel
   :members:

.. autoclass:: skelo.model.glicko2.Glicko2Model
   :members:

Utilities
~~~~~~~~~

.. automodule:: skelo.utils.elo_data
   :members:

Examples
========
More usage examples, including using `sklearn` cross validation routines to tune Elo hyperparameters are available in the project repository's `README <https://github.com/mbhynes/skelo#extended-usage-examples>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
