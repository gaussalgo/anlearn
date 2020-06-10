# Jupyter notebooks

We're trying to provide multiple useful and interesting examples in the form of Jupyter notebooks.
You can find them in the [notebooks](../notebooks/) folder.
In these examples, we'll try to show how to use, the advantages and disadvantages of anomaly detection methods found in anlearn library.
Also, we'll provide some background for them.

## Installation

Running notebooks with kernel from virtual environment (recommended):

1. Activate virtual environment.
   ```
   source venv/bin/activate
   ```
2. Install requirements. We're providing [requirements-notebook.txt](../requirements/requirements-notebook.txt) with version and hashes for Python 3.8.
   If you're using other python or you don't want to use the same versions as we did, you could use
   [requirements-notebook.in](../requirements/requirements-notebook.in).
   Python 3.8
   ```
   pip install -r requirements/requirements-notebook.txt
   ```
   other Python versions (or if you don't want to use specific version of libraries)
   ```
   pip install -r requirements/requirements-notebook.in
   ```
3. Install ipython kernel
   ```
   python -m ipykernel install --user --name anlearn-env --display-name "Anomaly learn"
   ```
4. Now you can run jupyter notebooks with kernel from your virtual environment.
   ```
   jupyter notebook
   ```
