# Copyright 2022 The Cirq Developers
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


def test_routed_circuit_with_mapping_simple():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.Moment(cirq.SWAP(q[0], q[1]).with_tags(cirq.RoutingSwapTag()))])
    expected_diagram = """
0: ───q(0)───×[<r>]───q(1)───
      │      │        │
1: ───q(1)───×────────q(0)───"""
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit), expected_diagram)

    expected_diagram_with_initial_mapping = """
0: ───a───×[<r>]───b───
      │   │        │
1: ───b───×────────a───"""
    cirq.testing.assert_has_diagram(
        cirq.routed_circuit_with_mapping(
            circuit, {cirq.NamedQubit("a"): q[0], cirq.NamedQubit("b"): q[1]}
        ),
        expected_diagram_with_initial_mapping,
    )

    # if swap is untagged should not affect the mapping
    circuit = cirq.Circuit([cirq.Moment(cirq.SWAP(q[0], q[1]))])
    expected_diagram = """
0: ───q(0)───×───
      │      │
1: ───q(1)───×───"""
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit), expected_diagram)

    circuit = cirq.Circuit(
        [
            cirq.Moment(cirq.X(q[0]).with_tags(cirq.RoutingSwapTag())),
            cirq.Moment(cirq.SWAP(q[0], q[1])),
        ]
    )
    with pytest.raises(
        ValueError, match="Invalid circuit. A non-SWAP gate cannot be tagged a RoutingSwapTag."
    ):
        cirq.routed_circuit_with_mapping(circuit)


def test_routed_circuit_with_mapping_multi_swaps():
    q = cirq.LineQubit.range(6)
    circuit = cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(q[3], q[4])),
            cirq.Moment(cirq.CNOT(q[5], q[4]), cirq.CNOT(q[2], q[3])),
            cirq.Moment(
                cirq.CNOT(q[2], q[1]), cirq.SWAP(q[4], q[3]).with_tags(cirq.RoutingSwapTag())
            ),
            cirq.Moment(
                cirq.SWAP(q[0], q[1]).with_tags(cirq.RoutingSwapTag()),
                cirq.SWAP(q[3], q[2]).with_tags(cirq.RoutingSwapTag()),
            ),
            cirq.Moment(cirq.CNOT(q[2], q[1])),
            cirq.Moment(cirq.CNOT(q[1], q[0])),
        ]
    )
    expected_diagram = """
0: ───q(0)────────────────────q(0)───×[<r>]───q(1)───────X───
      │                       │      │        │          │
1: ───q(1)───────────X────────q(1)───×────────q(0)───X───@───
      │              │        │               │      │
2: ───q(2)───────@───@────────q(2)───×────────q(4)───@───────
      │          │            │      │        │
3: ───q(3)───@───X───×────────q(4)───×[<r>]───q(2)───────────
      │      │       │        │               │
4: ───q(4)───X───X───×[<r>]───q(3)────────────q(3)───────────
      │          │            │               │
5: ───q(5)───────@────────────q(5)────────────q(5)───────────
"""
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit), expected_diagram)
