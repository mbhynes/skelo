# skelo

An implementation of the [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system) with a [scikit-learn](https://scikit-learn.org/stable/)-compatible interface.

The `skelo` package is a simple implementation suitable for small-scale ratings that fit into memory on a single machine, intended to provide a simple API for creating elo ratings on small-scale problems, primarily as a means of feature transformation for inclusion in other `sklearn` pipelines.

## Installation

- Installation via the PyPI package using pip:
```python
pip3 install skelo
```

## Example Usage

```python
from skelo.utils.data import generate_data
from skelo.models import EloEstimator

```

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
