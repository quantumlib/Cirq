#!/bin/sh

check/pytest -p no:randomly -m slow \
    dev_tools/notebooks/notebook_test.py::test_notebooks_against_cirq_head \
    -k "\
    docs/build/classical_control.ipynb \
    or docs/google/best_practices.ipynb \
    or docs/hardware/pasqal/getting_started.ipynb \
    or docs/simulate/virtual_engine_interface.ipynb \
    or docs/start/intro.ipynb \
    or docs/transform/routing_transformer.ipynb \
    " "$@"
