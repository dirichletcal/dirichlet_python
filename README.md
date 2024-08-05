[![CI][ci:b]][ci]
[![License BSD3][license:b]][license]
![Python3.12][python:b]
[![pypi][pypi:b]][pypi]
[![codecov][codecov:b]][codecov]
[![DOI](https://zenodo.org/badge/227004259.svg)](https://zenodo.org/badge/latestdoi/227004259)

[ci]: https://github.com/dirichletcal/dirichlet_python/actions/workflows/ci.yml
[ci:b]: https://github.com/dirichletcal/dirichlet_python/workflows/CI/badge.svg
[license]: https://github.com/dirichletcal/dirichlet_python/blob/master/LICENSE.txt
[license:b]: https://img.shields.io/github/license/dirichletcal/dirichlet_python.svg
[python:b]: https://img.shields.io/badge/python-3.12-blue
[pypi]: https://badge.fury.io/py/dirichletcal
[pypi:b]: https://badge.fury.io/py/dirichletcal.svg
[codecov]: https://codecov.io/gh/dirichletcal/dirichlet_python
[codecov:b]: https://codecov.io/gh/dirichletcal/dirichlet_python/branch/master/graph/badge.svg

# Dirichlet Calibration Python implementation

This is a Python implementation of the Dirichlet Calibration presented in
__Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities
with Dirichlet calibration__ at NeurIPS 2019. The original version used Python
3.8 and reached version `0.4.2`. The code started using Python 3.12 from
version `0.5.0`, you can see the other version in the GitHub history, tags, or
in Pypi.

# Installation

```
# Clone the repository
git clone git@github.com:dirichletcal/dirichlet_python.git
# Go into the folder
cd dirichlet_python
# Create a new virtual environment with Python3
python3.12 -m venv venv
# Load the generated virtual environment
source venv/bin/activate
# Upgrade pip
pip install --upgrade pip
# Install all the dependencies
pip install -r requirements.txt
pip install --upgrade jaxlib
```

# Unittest

```
python -m unittest discover dirichletcal
```


# Cite

If you use this code in a publication please cite the following paper


```
@inproceedings{kull2019dircal,
  title={Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration},
  author={Kull, Meelis and Nieto, Miquel Perello and K{\"a}ngsepp, Markus and Silva Filho, Telmo and Song, Hao and Flach, Peter},
  booktitle={Advances in Neural Information Processing Systems},
  pages={12295--12305},
  year={2019}
}
```

# Examples

You can find some examples on how to use this package in the folder
[examples](examples)

# Pypi

To push a new version to Pypi first build the package

```
python3.12 setup.py sdist
```

And then upload to Pypi with twine

```
twine upload dist/*
```

It may require user and password if these are not set in your home directory a
file  __.pypirc__

```
[pypi]
username = __token__
password = pypi-yourtoken
```
