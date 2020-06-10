# Developers Guide

## Tools

As anlearn developers, we're using these tools.

- Code formating:
  - [black]
  - [isort]
- Linting:
  - [flake8]
  - [mypy]
- Requirements:
  - [pip-tools]
- Testing:
  - [pytest]
- Documentation:
  - [MkDocs]
  - [Material for MkDocs]
- Other:
  - [pre-commit]
  - [nix-shell]
  - [direnv]

## Setting-up the developers' environment

### Nix-shell & direnv

For easy environment management, we're using [nix-shell] in combination with [direnv].
Using these two tools reduces the time and effort required to create and maintain a deterministic environment.
We highly recommend using them. Nix configuration is in [shell.nix](../shell.nix) and for direnv in [.envrc](../.envrc).

### Python tools

As for python versions currently, support is for python 3.6, 3.7, and 3.8.
To ensure a similar code style choice for formating is [black] and [isort]. As a linters we use [mypy] and [flake8].

For easier code check before committing any changes, there is an option to use the pre-commit tool.
As you can see in `.pre-commit-config.yaml` it is using only currently installed versions of black, isort, and flake8.

#### Installation

It's highly recommended to use python virtual environment for development.

- All tools with their specified version are in the requirements-dev files.
  - `pip install -r requirements/requirements-3.8-dev.txt`
- For pre-commit `pre-commit install` (after installing tools)

#### Configuration files

- Configuration for [isort], [pytest], [flake8], and [mypy] is in [setup.cfg](../setup.cfg) file.
- Configuration for [pre-commit] is in [.pre-commit-config.yaml](../.pre-commit-config.yaml) file.

### Generating requirements - pip-tools

For easier testing, we're using pinned requirements for every supported python version. For this purpose, there is the pip-tools package.
Configuration for generating requirement files is in Makefile.
The easiest way to generate them is by using a pre-prepared make file.

```
make requirements
```

If you want only to generate requirements for one specific python version you could use

```
make requirements-3.8
```

## Tests

All tests are in [test](../test) folder. We're using [pytest] for testing.

## Documentation

We're using [MkDocs] in combination with [Material for MkDocs].
For generating documentation, you need to have `requirements-dev` and `anlearn` installed.
Then you can simply run `mkdocs serve` or `make docs`.

[black]: https://github.com/psf/black
[isort]: https://github.com/timothycrosley/isort
[flake8]: https://github.com/PyCQA/flake8
[mypy]: https://github.com/python/mypy
[pip-tools]: https://github.com/jazzband/pip-tools
[pytest]: https://github.com/pytest-dev/pytest
[mkdocs]: https://github.com/mkdocs/mkdocs/
[material for mkdocs]: https://github.com/squidfunk/mkdocs-material
[pre-commit]: https://github.com/pre-commit/pre-commit
[nix-shell]: https://nixos.org/
[direnv]: https://direnv.net/
