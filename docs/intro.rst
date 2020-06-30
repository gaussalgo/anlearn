Installation & Jupyter notebooks
================================

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


Jupyter notebooks
-----------------

We're trying to provide multiple useful and interesting examples in the form of Jupyter notebooks.
You can find them in the [notebooks](../notebooks/) folder.
In these examples, we'll try to show how to use, the advantages and disadvantages of anomaly detection methods found in anlearn library.
Also, we'll provide some background for them.

Installation
~~~~~~~~~~~~

Running notebooks with kernel from virtual environment (recommended):

1. Activate virtual environment.

    .. code-block:: bash

        source venv/bin/activate

2. Install requirements. We're providing requirements-notebook.txt with version and hashes for Python 3.8.
   If you're using other python or you don't want to use the same versions as we did, you could use requirements-notebook.in.
   Python 3.8

   .. code-block:: bash

        pip install -r requirements/requirements-notebook.txt

   other Python versions (or if you don't want to use specific version of libraries)

   .. code-block:: bash

        pip install -r requirements/requirements-notebook.in

3. Install ipython kernel

    .. code-block:: bash

        python -m ipykernel install --user --name anlearn-env --display-name "Anomaly learn"

4. Now you can run jupyter notebooks with kernel from your virtual environment.

    .. code-block:: bash

        jupyter notebook