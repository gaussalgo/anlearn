![anomaly-learn-with-text](https://raw.githubusercontent.com/gaussalgo/anlearn/master/docs/img/anomaly-learn-with-text.png)

[![PyPI version](https://badge.fury.io/py/anlearn.svg)](https://badge.fury.io/py/anlearn) [![Documentation Status](https://readthedocs.org/projects/anlearn/badge/?version=latest)](https://anlearn.readthedocs.io/en/latest/?badge=latest) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)

# anlearn - Anomaly learn

anlearn is a python package that aims to implement multiple state-of-the-art anomaly detection methods in familiar scikit-learn API.

## Installation
anlearn depends on [scikit-learn] and it's dependencies [scipy] and [numpy].

Requirements:
* python >=3.6
* [scikit-learn]
* [scipy]
* [numpy]

Requirements for every supported python version with version and hashes could be found in [requirements](requirements) folder.
We're using [pip-tools](https://github.com/jazzband/pip-tools) for generating requirements files.


### Intallation options
#### PyPI installation
```
pip install anlearn
```

#### Installation from source
```
git clone https://github.com/gaussalgo/anlearn
cd anlearn
```

Installing requirements.

```
# Generated requirements for all supported python versions
ls requirements/requirements-3.*.txt | grep -v dev
requirements/requirements-3.6.txt
requirements/requirements-3.7.txt
requirements/requirements-3.8.txt
pip install -r requirements/requirements-3.8.txt
```
or
```
pip install scikit-learn numpy scipy
```

Install `anlearn`.
```
pip install .
```
or
```
python setup.py install
```

## Documentation
[docs](https://gaussalgo.github.io/anlearn/)

## License
GNU Lesser General Public License v3 or later (LGPLv3+)

anlearn  Copyright (C) 2020  Gauss Algorithmic a.s.

This package is in alpha state and comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to use, redistribute it, and contribute under certain conditions of its license.

[scikit-learn]: https://github.com/scikit-learn/scikit-learn
[numpy]: https://github.com/numpy/numpy
[scipy]: https://github.com/scipy/scipy

# Code of Conduct
[Code of Conduct](https://github.com/gaussalgo/anlearn/blob/master/CODE_OF_CONDUCT.md)
