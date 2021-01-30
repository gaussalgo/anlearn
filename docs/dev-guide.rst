Developers Guide
================

Tools
-----

As anlearn developers, we're using these tools.

* Code formating: black_, isort_
* Linting: flake8_, mypy_
* Requirements: poetry_
* Testing: pytest_, tox_
* Documentation: sphinx_
* Other: pre-commit_, nix-shell_, direnv_

Setting-up the developers' environment
--------------------------------------

Nix-shell & direnv
~~~~~~~~~~~~~~~~~~

For easy environment management, we're using nix-shell_ in combination with direnv_.
Using these two tools reduces the time and effort required to create and maintain a deterministic environment.
We highly recommend using them. Nix configuration is in the `shell.nix` files + `nix` folder and the direnv configuration is in the `.envrc` file.

Python tools
~~~~~~~~~~~~

As for python versions currently, support is for python 3.6, 3.7, and 3.8.
To ensure a similar code style choice for formating is black_ and isort_. As a linters we use mypy_ and flake8_.

For easier code check before committing any changes, there is an option to use the pre-commit tool.
As you can see in `.pre-commit-config.yaml` it is using only currently installed versions of black, isort, and flake8.

Installation using poetry
~~~~~~~~~~~~~~~~~~~~~~~~~

We are using the poetry_ for packaging and dependencies managemet. To installing all developement dependencies including ones for generating documentation simply use:

  .. code-block:: bash

    poetry install -E docs

* For pre-commit (after installing tools)

  .. code-block:: bash

    pre-commit install

Tests
-----

All tests are in `test` folder. We're using a combinaiton of tox_ and pytest_ for testing.
You can run tests by using the `tox` command directly or by using `make pytest` or `make check` commands.

Configuration files
~~~~~~~~~~~~~~~~~~~

- Configurations for flake8_ and mypy_ are in the `setup.cfg` file.
- Configurations for isort_, black_, and pytest_ are in the `pyproject.toml` file.
- Configuration for pre-commit_ is in the `.pre-commit-config.yaml` file.
- Configuration for tox_ is in the `tox.ini` file.

Documentation
-------------

We're using sphinx_ in combination with Read the Docs Sphinx Theme.
For generating the documentation, you have to have `anlearn[docs]` installed (`poetry install -E docs`). You can create the documentation by using the `make docs` command.

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
.. _poetry: https://github.com/python-poetry/poetry
.. _tox: https://github.com/tox-dev/tox
