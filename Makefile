.PHONY: clean clean-test clean-pyc clean-build docs help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"
LATEST_TAG := $(shell git describe --abbrev=0)

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

f8lint: ## check style with flake8
	poetry run flake8 raylab tests

pylint: ## lint code with pylint
	poetry run pylint raylab -d similarities

pylint-test: ## lint test files with pylint
	poetry run pylint --rcfile=tests/pylintrc tests -d similarities

similarities: ## check code duplication with pylint
	poetry run pylint raylab -d all -e similarities

test: ## run tests quickly with the default Python
	poetry run pytest

test-all: ## run tests on every Python version with tox
	poetry run tox

changelog:
	git describe --abbrev=0 | xargs poetry run auto-changelog --tag-prefix v --unreleased --stdout --starting-commit

push-release:
	git push origin master develop --tags

reorder-imports-staged:
	git diff --cached --name-only | xargs grep -rl --include "*.py" 'import' | xargs poetry run reorder-python-imports --separate-relative

bump-patch:
	poetry version patch
	git add pyproject.toml
	git commit -s -m "chore: bump version patch"

bump-minor:
	poetry version minor
	git add pyproject.toml
	git commit -s -m "chore: bump version minor"

bump-major:
	poetry version major
	git add pyproject.toml
	git commit -s -m "chore: bump version major"

coverage: ## check code coverage quickly with the default Python
	coverage run --source raylab -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/raylab.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ raylab
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

black:
	blackd 2> /dev/null &
