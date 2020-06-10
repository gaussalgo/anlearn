#!/bin/env sh
set -e

python3.6 --version
pip3.6 install --upgrade pip setuptools wheel pip-tools

python3.7 --version
pip3.7 install --upgrade pip setuptools wheel pip-tools

python3.8 --version
pip3.8 install --upgrade pip setuptools wheel pip-tools
