PIP_COMPILE_FLAGS = -U --generate-hashes --build-isolation
PYTHON_SOURCES = anlearn tests setup.py
PACKAGE_NAME = anlearn
REQUIREMENTS = requirements/requirements

default: check

check: black-check flake8 mypy pytest isort-check

fmt: black

black:
	black $(PYTHON_SOURCES)

black-check:
	black --check --diff $(PYTHON_SOURCES)

flake8:
	flake8 $(PYTHON_SOURCES)

isort:
	isort -rc $(PYTHON_SOURCES)

isort-check:
	isort --check --diff -rc $(PYTHON_SOURCES)

mypy:
	mypy $(PYTHON_SOURCES)

pytest:
	# run doctests on the installed Python modules instead of the source tree
	MODULE_PATH="$(shell python3 -c 'import $(PACKAGE_NAME); import os; print(os.path.dirname($(PACKAGE_NAME).__file__))')"
	pytest -v --color=yes --durations=20 --doctest-modules --cov "$(PACKAGE_NAME)" "$${MODULE_PATH}" tests

docs:
	mkdocs build --clean

rtfm: docs
	xdg-open docs/_build/html/index.html

requirements: requirements-3.6 requirements-3.7 requirements-3.8

requirements-3.6:
	@echo "# Please seat back and relax, this may take some time. :)"
	python3.6 -m piptools compile $(PIP_COMPILE_FLAGS) -o $(REQUIREMENTS)-3.6.txt setup.py
	python3.6 -m piptools compile $(PIP_COMPILE_FLAGS) -o  $(REQUIREMENTS)-3.6-dev.txt $(REQUIREMENTS)-dev.in

requirements-3.7:
	@echo "# Please seat back and relax, this may take some time. :)"
	python3.7 -m piptools compile $(PIP_COMPILE_FLAGS) -o  $(REQUIREMENTS)-3.7.txt setup.py
	python3.7 -m piptools compile $(PIP_COMPILE_FLAGS) -o  $(REQUIREMENTS)-3.7-dev.txt $(REQUIREMENTS)-dev.in

requirements-3.8:
	@echo "# Please seat back and relax, this may take some time. :)"
	python3.8 -m piptools compile $(PIP_COMPILE_FLAGS) -o  $(REQUIREMENTS)-3.8.txt setup.py
	python3.8 -m piptools compile $(PIP_COMPILE_FLAGS) -o  $(REQUIREMENTS)-3.8-dev.txt $(REQUIREMENTS)-dev.in

requirements-notebook:
	@echo "# Please seat back and relax, this may take some time. :)"
	python3.8 -m piptools compile $(PIP_COMPILE_FLAGS) -o  $(REQUIREMENTS)-notebook.txt $(REQUIREMENTS)-notebook.in

.PHONY: default fmt check black black-check flake8 mypy pytest docs rtfm requirements
