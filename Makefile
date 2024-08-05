SHELL := /bin/bash

requirements:
	pip install -r requirements.txt

requirements-dev: requirements
	pip install -r requirements-dev.txt

build: requirements-dev
	python setup.py sdist

pypi: build check-readme
	twine upload dist/*

# From Scikit-learn
code-analysis:
	flake8 pycalib | grep -v external
	pylint -E pycalib/ -d E1103,E0611,E1101

clean:
	rm -rf ./dist

# All the following assume the requirmenets-dev are installed, but to make the
# output clean the dependency has been removed
test: requirements-dev
	which pytest
	pytest --version
	pytest dirichletcal

check-readme:
	twine check dist/*
