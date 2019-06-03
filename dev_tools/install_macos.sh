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
    mkdir ~/.matplotlib && echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc
    fi

