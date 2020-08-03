Developers Guide
================

Tools
-----

As anlearn developers, we're using these tools.

* Code formating: black_, isort_
* Linting: flake8_, mypy_
* Requirements: pip-tools_
* Testing: pytest_
* Documentation: sphinx_
* Other: pre-commit_, nix-shell_, direnv_

Setting-up the developers' environment
--------------------------------------

Nix-shell & direnv
~~~~~~~~~~~~~~~~~~

For easy environment management, we're using nix-shell_ in combination with direnv_.
Using these two tools reduces the time and effort required to create and maintain a deterministic environment.
We highly recommend using them. Nix configuration is in `shell.nix` and for direnv in `.envrc`.

Python tools
~~~~~~~~~~~~

As for python versions currently, support is for python 3.6, 3.7, and 3.8.
To ensure a similar code style choice for formating is black_ and isort_. As a linters we use mypy_ and flake8_.

For easier code check before committing any changes, there is an option to use the pre-commit tool.
As you can see in `.pre-commit-config.yaml` it is using only currently installed versions of black, isort, and flake8.

Installation
~~~~~~~~~~~~

It's highly recommended to use python virtual environment for development.

* All tools with their specified version are in the requirements-dev files.

  .. code-block:: bash

    pip install -r requirements/requirements-3.8-dev.txt

* For pre-commit (after installing tools)

  .. code-block:: bash

    pre-commit install

Configuration files
~~~~~~~~~~~~~~~~~~~

- Configuration for isort_, pytest_, flake8_, and mypy_ is in `setup.cfg` file.
- Configuration for pre-commit_ is in `.pre-commit-config.yaml` file.

Generating requirements - pip-tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For easier testing, we're using pinned requirements for every supported python version. For this purpose, there is the pip-tools package.
Configuration for generating requirement files is in Makefile.
The easiest way to generate them is by using a pre-prepared make file.

.. code-block::: bash

    make requirements

If you want only to generate requirements for one specific python version you could use

.. code-block::: bash

    make requirements-3.8

Tests
-----

All tests are in `test` folder. We're using pytest_ for testing.

Documentation
-------------

We're using sphinx_ in combination with Read the Docs Sphinx Theme.
For generating documentation, you need to have `requirements-dev` and `anlearn` installed.

.. _black: https://github.com/psf/black
.. _isort: https://github.com/timothycrosley/isort
.. _flake8: https://github.com/PyCQA/flake8
.. _mypy: https://github.com/python/mypy
.. _pip-tools: https://github.com/jazzband/pip-tools
.. _pytest: https://github.com/pytest-dev/pytest
.. _sphinx: https://github.com/sphinx-doc/sphinx
.. _pre-commit: https://github.com/pre-commit/pre-commit
.. _nix-shell: https://nixos.org/
.. _direnv: https://direnv.net/