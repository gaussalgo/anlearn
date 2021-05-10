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
	@echo "# Please seat back and relax, this may take some time. :)"
	poetry update
	poetry export -f requirements.txt -o requirements.txt
	poetry export --dev -f requirements.txt -o requirements-dev.txt

.PHONY: default fmt check black fmt-check flake8 mypy pytest docs rtfm requirements
