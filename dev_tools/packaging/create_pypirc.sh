#!bin/bash

# Create default pypi config file with token authentication.
# refrence: https://packaging.python.org/en/latest/specifications/pypirc/#common-configurations
echo """
[pypi]
username = __token__
password = ${CIRQ_PYPI_TOKEN}

[testpypi]
username = __token__
password = ${CIRQ_TEST_PYPI_TOKEN}
""" >> $HOME/.pypirc
