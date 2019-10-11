# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import cirq

from cirq.contrib.paulistring import (
    convert_and_separate_circuit,
    pauli_string_dag_from_circuit,
)


@pytest.mark.parametrize('repetition', range(6))
def test_pauli_string_dag_from_circuit(repetition):
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)
    c_left, _ = convert_and_separate_circuit(c_orig)

    c_left_dag = pauli_string_dag_from_circuit(c_left)
    c_left_reordered = c_left_dag.to_circuit()

    cirq.testing.assert_allclose_up_to_global_phase(c_left.unitary(),
                                                    c_left_reordered.unitary(),
                                                    atol=1e-7)
