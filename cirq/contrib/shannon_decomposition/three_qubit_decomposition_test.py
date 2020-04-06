# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from random import random

import numpy as np

import cirq
from cirq.contrib.shannon_decomposition.three_qubit_decomposition import (
    _to_special, _is_three_cnot_two_qubit_unitary)


def test_is_three_cnot_two_qubit_unitary():
    assert _is_three_cnot_two_qubit_unitary(
        _two_qubit_circuit_with_cnots(3)._unitary_())
    assert not _is_three_cnot_two_qubit_unitary(
        _two_qubit_circuit_with_cnots(2)._unitary_())
    assert not _is_three_cnot_two_qubit_unitary(
        _two_qubit_circuit_with_cnots(1)._unitary_())
    assert not _is_three_cnot_two_qubit_unitary(np.eye(4))


def test_to_special():
    u = cirq.testing.random_unitary(4)
    su = _to_special(u)
    assert not cirq.is_special_unitary(u)
    assert cirq.is_special_unitary(su)


def _two_qubit_circuit_with_cnots(num_cnots=3, a=None, b=None):
    if a is None or b is None:
        a, b = cirq.LineQubit.range(2)

    def random_one_qubit_gate():
        return cirq.PhasedXPowGate(phase_exponent=random(), exponent=random())

    def one_cz():
        return [
            random_one_qubit_gate().on(a),
            random_one_qubit_gate().on(b),
            cirq.CZ.on(a, b)
        ]

    return cirq.Circuit([
        random_one_qubit_gate().on(a),
        random_one_qubit_gate().on(b), [one_cz() for _ in range(num_cnots)]
    ])
