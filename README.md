[![CI][ci:b]][ci]
[![License BSD3][license:b]][license]
![Python3.8][python:b]
[![pypi][pypi:b]][pypi]
[![codecov][codecov:b]][codecov]

[ci]: https://github.com/dirichletcal/dirichlet_python/actions/workflows/ci.yml
[ci:b]: https://github.com/dirichletcal/pycalib/workflows/CI/badge.svg
[license]: https://github.com/dirichletcal/dirichlet_python/blob/master/LICENSE.txt
[license:b]: https://img.shields.io/github/license/dirichletcal/pycalib.svg
[python:b]: https://img.shields.io/badge/python-3.8-blue
[pypi]: https://badge.fury.io/py/pycalib
[pypi:b]: https://badge.fury.io/py/pycalib.svg
[codecov]: https://codecov.io/gh/dirichletcal/dirichlet_python
[codecov:b]: https://codecov.io/gh/dirichletcal/dirichlet_python/branch/master/graph/badge.svg

# Dirichlet Calibration Python implementation

This is a Python implementation of the Dirichlet Calibration presented in
__Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities
with Dirichlet calibration__ at NeurIPS 2019.

# Installation

```
# Clone the repository
git clone git@github.com:dirichletcal/dirichlet_python.git
# Go into the folder
cd dirichlet_python
# Create a new virtual environment with Python3
python3.6 -m venv venv
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
