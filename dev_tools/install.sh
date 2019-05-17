#!/bin/bash

if [ "$TRAVIS_OS_NAME" == "osx" ]; then

    # Install some custom requirements on macOS
    # e.g. brew install pyenv-virtualenv
    brew update
    brew install openssl readline pyenv-virtualenv
    brew outdated pyenv || brew upgrade pyenv
    pyenv install $PYTHON
    export PYENV_VERSION=$PYTHON
    export PATH="/Users/travis/.pyenv/shims:${PATH}"
    pyenv virtualenv venv
    source /Users/travis/.pyenv/versions/3.6.0/envs/venv/bin/activate
    python --version
    mkdir ~/.matplotlib && echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc
    python -m pip install -r requirements.txt
    python -m pip install -r dev_tools/conf/pip-list-dev-tools.txt
    check/pytest-and-incremental-coverage master
    fi