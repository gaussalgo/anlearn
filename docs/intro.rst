Installation
============

anlearn depends on scikit-learn_ and it's dependencies scipy_ and numpy_.

Requirements:

* python >=3.6
* scikit-learn_
* scipy_
* numpy_

Requirements for every supported python version with version and hashes could be found in requirements folder.
We're using pip-tools_ for generating requirements files.

.. _numpy: https://github.com/numpy/numpy
.. _scipy: https://github.com/scipy/scipy
.. _scikit-learn: https://github.com/scikit-learn/scikit-learn
.. _pip-tools: https://github.com/jazzband/pip-tools

Intallation options
-------------------
PyPI installation
~~~~~~~~~~~~~~~~~
.. code-block:: bash

    pip install anlearn


Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

    git clone https://github.com/gaussalgo/anlearn
    cd anlearn


Installing requirements.

.. code-block:: bash

    # Generated requirements for all supported python versions
    ls requirements/requirements-3.*.txt | grep -v dev
    requirements/requirements-3.6.txt
    requirements/requirements-3.7.txt
    requirements/requirements-3.8.txt
    pip install -r requirements/requirements-3.8.txt

or

.. code-block:: bash

    pip install scikit-learn numpy scipy

Install `anlearn`.

.. code-block:: bash

    pip install .

or

.. code-block:: bash

    python setup.py install
