#!/bin/bash
# This script should only be run by Travis
if [ "$TRAVIS_OS_NAME" == "osx" ]; then

    # Install some custom requirements on macOS
    # e.g. brew install pyenv-virtualenv
    
    #`import matplotlib` throws a  RuntimeError without this
    mkdir -p ~/.matplotlib && echo "backend: TkAgg" >> ~/.matplotlib/matplotlibrc
else
    echo "This operating system is not osx"
    exit 1
fi

