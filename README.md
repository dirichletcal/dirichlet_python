# Development

Please follow this instructions to be sure that we all have the same library
versions (it may take 30 minutes or more to install all packages).

```
# Clone the repository
git clone git@bitbucket.org:dirichlet_cal/dirichlet.git
# Go into the folder
cd dirichlet
# Create a new virtual environment with Python3
python3 -m venv venv
# Load the generated virtual environment
source venv/bin/activate
# Install all the dependencies
pip install -r requirements.txt
```

# Unittest

It is currently necessary to be in the upper folder to run the unittest. We may
think in the future to create the Dirichlet package in a folder.

```
cd ..
python -m unittest discover dirichlet
```
