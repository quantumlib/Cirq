echo """
[pypi]
username = __token__
password = ${{ secrets.CIRQ_PYPI_TOKEN }}

[testpypi]
username = __token__
password = ${{ secrets.CIRQ_TEST_PYPI_TOKEN }}
""" >> $HOME/.pypirc