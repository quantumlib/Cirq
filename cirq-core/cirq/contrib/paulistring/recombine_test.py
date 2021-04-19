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
    move_pauli_strings_into_circuit,
)


def _assert_no_multi_qubit_pauli_strings(circuit: cirq.Circuit) -> None:
    for op in circuit.all_operations():
        if isinstance(op, cirq.PauliStringGateOperation):
            assert len(op.pauli_string) == 1


def test_move_non_clifford_into_clifford():
    cg = pytest.importorskip("cirq_google")
    q0, q1, q2 = cirq.LineQubit.range(3)
    c_orig = cirq.testing.nonoptimal_toffoli_circuit(q0, q1, q2)

    c_left, c_right = convert_and_separate_circuit(c_orig)

    # Normally, c_left would be optimized here
    c_left_dag = pauli_string_dag_from_circuit(c_left)

    c_recombined1 = move_pauli_strings_into_circuit(c_left, c_right)
    c_recombined2 = move_pauli_strings_into_circuit(c_left_dag, c_right)

    _assert_no_multi_qubit_pauli_strings(c_recombined1)
    _assert_no_multi_qubit_pauli_strings(c_recombined2)

    baseline_len = len(cg.optimized_for_xmon(c_orig))
    opt_len1 = len(cg.optimized_for_xmon(c_recombined1))
    opt_len2 = len(cg.optimized_for_xmon(c_recombined2))
    assert opt_len1 <= baseline_len
    assert opt_len2 <= baseline_len
