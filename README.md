![anomaly-learn-with-text](https://raw.githubusercontent.com/gaussalgo/anlearn/master/docs/img/anomaly-learn-with-text.png)

[![PyPI version](https://badge.fury.io/py/anlearn.svg)](https://badge.fury.io/py/anlearn) [![Documentation Status](https://readthedocs.org/projects/anlearn/badge/?version=latest)](https://anlearn.readthedocs.io/en/latest/?badge=latest) [![Gitter](https://badges.gitter.im/gaussalgo-anlearn/community.svg)](https://gitter.im/gaussalgo-anlearn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](code_of_conduct.md)

# anlearn - Anomaly learn

In [Gauss Algorithmic], we're working on many anomaly/fraud detection projects using open-source tools. We decided to put our two cents in and  "tidy up" some of our code snippets, add documentation, examples, and release them as an open-source package. So let me introduce **anlearn**. It aims to offer multiple interesting anomaly detection methods in familiar [scikit-learn] API so you could quickly try some anomaly detection experiments yourself.

So far, this package is an alpha state and ready for your experiments.

Do you have any questions, suggestions, or want to chat? Feel free to contact us via [Github], [Gitter], or email.

## Installation
anlearn depends on [scikit-learn] and it's dependencies [scipy] and [numpy].

Requirements:
* python >=3.6
* [scikit-learn]
* [scipy]
* [numpy]

Requirements for every supported python version with version and hashes could be found in `requirements` folder.
We're using [pip-tools] for generating requirements files.


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

Instalil `anlearn`.
```
pip install .
```
or by using [poetry]
```
poetry install
```


## Documentation
You can find documentation at Read the Docs: [docs](https://anlearn.readthedocs.io/en/latest/).

## Contat us
Do you have any questions, suggestions, or want to chat? Feel free to contact us via [Github], [Gitter], or email.

## License
GNU Lesser General Public License v3 or later (LGPLv3+)

anlearn  Copyright (C) 2020  Gauss Algorithmic a.s.

This package is in alpha state and comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to use, redistribute it, and contribute under certain conditions of its license.

# Code of Conduct
[Code of Conduct](https://github.com/gaussalgo/anlearn/blob/master/CODE_OF_CONDUCT.md)


[scikit-learn]: https://github.com/scikit-learn/scikit-learn
[numpy]: https://github.com/numpy/numpy
[scipy]: https://github.com/scipy/scipy
[pip-tools]: https://github.com/jazzband/pip-tools
[Gitter]: https://gitter.im/gaussalgo-anlearn/community
[Gauss Algorithmic]: https://www.gaussalgo.com/en/
[GitHub]: https://github.com/gaussalgo/anlearn
[poetry]: https://github.com/python-poetry/poetry
