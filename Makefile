.PHONY: venv

SHELL := /bin/bash

venv:
	python3.8 -m venv venv

pip: load-venv
	pip install --upgrade pip

load-venv:
	source ./venv/bin/activate

requirements: pip
	pip install -r requirements.txt

requirements-dev: requirements pip
	pip install -r requirements-dev.txt

build: requirements-dev
	python3.8 setup.py sdist

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
	pytest --doctest-modules --cov-report=term-missing --cov=dirichletcal dirichletcal

check-readme:
	twine check dist/*
