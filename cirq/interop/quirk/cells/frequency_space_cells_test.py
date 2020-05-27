# Copyright 2019 The Cirq Developers
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

import sympy

import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns


def test_frequency_space_gates():
    a, b, c = cirq.LineQubit.range(3)

    assert_url_to_circuit_returns('{"cols":[["QFT3"]]}',
                                  cirq.Circuit(cirq.qft(a, b, c),))
    assert_url_to_circuit_returns(
        '{"cols":[["QFT†3"]]}', cirq.Circuit(cirq.inverse(cirq.qft(a, b, c)),))

    assert_url_to_circuit_returns(
        '{"cols":[["PhaseGradient3"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=3, exponent=0.5)(a, b, c),))
    assert_url_to_circuit_returns(
        '{"cols":[["PhaseUngradient3"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=3, exponent=-0.5)(a, b, c),))

    t = sympy.Symbol('t')
    assert_url_to_circuit_returns(
        '{"cols":[["grad^t2"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=2, exponent=2 * t)(a, b),))
    assert_url_to_circuit_returns(
        '{"cols":[["grad^t3"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=3, exponent=4 * t)(a, b, c),))
    assert_url_to_circuit_returns(
        '{"cols":[["grad^-t3"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=3, exponent=-4 * t)(a, b, c),))
