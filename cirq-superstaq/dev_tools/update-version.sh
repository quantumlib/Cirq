#!/bin/bash

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$(git rev-parse --show-toplevel)"

python setup.py bdist_wheel
twine upload dist/* -u __token__ -p "$PYPI_API_KEY"
