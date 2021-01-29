PYTHON_SOURCES = anlearn tests examples

default: check

check:
	tox -p

fmt: isort black

black:
	black $(PYTHON_SOURCES)
isort:
	isort $(PYTHON_SOURCES)

fmt-check:
	tox -e format

flake8:
	tox -e flake8

mypy:
	tox -e mypy

pytest:
	tox -e py36,py37,py38 -p

docs:
	make -C docs html

rtfm: docs
	xdg-open docs/_build/html/index.html

requirements:
	poetry update

.PHONY: default fmt check black fmt-check flake8 mypy pytest docs rtfm requirements
