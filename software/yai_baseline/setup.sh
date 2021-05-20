#!/bin/bash

MY_DIR="`python -c "import os; print(os.path.realpath('$1'))"`"
cd $MY_DIR

# Run AIX server
cd aix
echo 'Setting up AIX server..'
python3 -m venv .env
source .env/bin/activate
pip install --use-deprecated=legacy-resolver -U pip setuptools wheel twine
pip install --use-deprecated=legacy-resolver -r requirements.txt
cd ..

# Run YAI server
cd yai
echo 'Setting up YAI server..'
python3 -m venv .env
source .env/bin/activate
pip install --use-deprecated=legacy-resolver -U pip setuptools wheel twine
pip install --use-deprecated=legacy-resolver -r requirements.txt
cd ..
