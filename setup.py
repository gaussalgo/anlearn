#!/usr/bin/env python3

from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="UTF-8") as f:
    long_description = f.read()

setup(
    name="anlearn",
    description="Anomaly learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ondrej Kurák",
    author_email="kurak@gaussalgo.com",
    url="https://github.com/gaussalgo/anlearn",
    packages=["anlearn"],
    use_scm_version={
        "write_to": ".version",
        "write_to_template": "{version}\n",
        # Custom version for test PyPI
        "local_scheme": lambda x: "",
    },
    setup_requires=["setuptools_scm"],
    python_requires=">3.6",
    install_requires=["numpy", "scipy", "sklearn"],
    entry_points={"console_scripts": []},
    package_data={"anlearn": ["py.typed"]},
    include_package_data=True,
    license="LGPLv3+",
    platforms=["platform-independent"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    keywords=[
        "Anomaly detection",
        "Outlier detection",
        "Anomaly learn",
        "anomaly-learn",
        "anlearn",
    ],
)
