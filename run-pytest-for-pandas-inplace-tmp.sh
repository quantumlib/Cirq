#!/bin/sh

if test ${#ERROR_ON_INPLACE} -ne 0; then
    w_option="-Werror:The inplace:DeprecationWarning"
else
    w_option="-Wdefault"
fi

check/pytest -n0 -s "${w_option}" \
    --randomly-seed=0 --randomly-dont-reorganize \
    cirq-core/cirq/contrib/quantum_volume/quantum_volume_test.py \
    cirq-core/cirq/experiments/t1_decay_experiment_test.py \
    examples/advanced/quantum_volume_test.py \
    ;
