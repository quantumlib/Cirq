# Copyright 2024 The Cirq Developers
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

import numpy as np

import cirq
from cirq.transformers.gauge_compiling import CZGaugeTransformer, CZGaugeTransformerMM
from cirq.transformers.gauge_compiling.gauge_compiling_test_utils import GaugeTester


class TestCZGauge(GaugeTester):
    two_qubit_gate = cirq.CZ
    gauge_transformer = CZGaugeTransformer


def test_multi_layer_pull_through():
    """Test case.
    Input:
              ┌──┐
    0: ───@────@─────H───────@───@───
          │    │             │   │
    1: ───@────┼@────────────@───@───
               ││
    2: ───@────@┼────────@───@───@───
          │     │        │   │   │
    3: ───@─────@────────@───@───@───
              └──┘
        An example output:
                  ┌──┐
    0: ───Z───@────@─────Z───H───X───────@───@───X───
              │    │                     │   │
    1: ───Y───@────┼@────X───────I───────@───@───────
                   ││
    2: ───Y───@────@┼────X───────I───@───@───@───────
              │     │                │   │   │
    3: ───X───@─────@────X───────I───@───@───@───────
                  └──┘
    """
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    input_circuit = cirq.Circuit(
        cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3)),
        cirq.Moment(cirq.CZ(q0, q2), cirq.CZ(q1, q3)),
        cirq.Moment(cirq.H(q0)),
        cirq.Moment(cirq.CZ(q2, q3)),
        cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3)),
        cirq.Moment(cirq.CZ(q0, q1), cirq.CZ(q2, q3)),
    )
    transformer = CZGaugeTransformerMM

    output_circuit = transformer(input_circuit, prng=np.random.default_rng())
    cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
        input_circuit, output_circuit, {q: q for q in input_circuit.all_qubits()}
    )
