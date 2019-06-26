#!/bin/bash
# This script should only be run by Travis
if [ "$TRAVIS_OS_NAME" == "osx" ]; then

    # Install some custom requirements on macOS
    # e.g. brew install pyenv-virtualenv
    PYTHON=3.7.0
    brew update
    brew install openssl readline pyenv-virtualenv
    brew outdated pyenv || brew upgrade pyenv
    pyenv install $PYTHON
    export PYENV_VERSION=$PYTHON
    export PATH="/Users/travis/.pyenv/shims:${PATH}"
    pyenv virtualenv venv
    
    #`import matplotlib` throws a  RuntimeError without this
    mkdir ~/.matplotlib && echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc
    fi

