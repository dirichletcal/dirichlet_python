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
```

# Unittest

```
python -m unittest discover dirichletcal
```


# Cite

If you use this code please cite the following paper


```
@inproceedings{kull2019dircal,
  title={Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with Dirichlet calibration},
  author={Kull, Meelis and Nieto, Miquel Perello and K{\"a}ngsepp, Markus and Silva Filho, Telmo and Song, Hao and Flach, Peter},
  booktitle={Advances in Neural Information Processing Systems},
  pages={12295--12305},
  year={2019}
}
```
