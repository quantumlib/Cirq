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

import cirq


def test_routed_circuit_with_mapping_simple():
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.Moment(cirq.SWAP(q[0], q[1]).with_tags(cirq.RoutingSwapTag()))])
    expected_diagram = """
0: ───q(0)───×[cirq.RoutingSwapTag()]───q(1)───
      │      │                          │
1: ───q(1)───×──────────────────────────q(0)───"""
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit), expected_diagram)

    expected_diagram_with_initial_mapping = """
0: ───a───×[cirq.RoutingSwapTag()]───b───
      │   │                          │
1: ───b───×──────────────────────────a───"""
    cirq.testing.assert_has_diagram(
        cirq.routed_circuit_with_mapping(
            circuit, {cirq.NamedQubit("a"): q[0], cirq.NamedQubit("b"): q[1]}
        ),
        expected_diagram_with_initial_mapping,
    )

    # if swap is untagged should not affect the mapping
    circuit = cirq.Circuit([cirq.Moment(cirq.SWAP(q[0], q[1]))])
    expected_diagram = """
0: ───q(0)───×───q(0)───
      │      │   │
1: ───q(1)───×───q(1)───"""
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit), expected_diagram)


def test_routed_circuit_with_mapping_multi_swaps():
    circuit = cirq.Circuit(
        [
            cirq.Moment(cirq.CNOT(cirq.GridQubit(6, 4), cirq.GridQubit(7, 4))),
            cirq.Moment(
                cirq.CNOT(cirq.GridQubit(8, 4), cirq.GridQubit(7, 4)),
                cirq.CNOT(cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)),
            ),
            cirq.Moment(
                cirq.CNOT(cirq.GridQubit(5, 4), cirq.GridQubit(4, 4)),
                cirq.SWAP(cirq.GridQubit(7, 4), cirq.GridQubit(6, 4)).with_tags(
                    cirq.RoutingSwapTag()
                ),
            ),
            cirq.Moment(
                cirq.SWAP(cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)).with_tags(
                    cirq.RoutingSwapTag()
                ),
                cirq.SWAP(cirq.GridQubit(6, 4), cirq.GridQubit(5, 4)).with_tags(
                    cirq.RoutingSwapTag()
                ),
            ),
            cirq.Moment(cirq.CNOT(cirq.GridQubit(5, 4), cirq.GridQubit(4, 4))),
            cirq.Moment(cirq.CNOT(cirq.GridQubit(4, 4), cirq.GridQubit(3, 4))),
        ]
    )
    expected_diagram = """
(3, 4): ───q(3, 4)───────q(3, 4)───────q(3, 4)──────────────────────────────q(3, 4)───×[cirq.RoutingSwapTag()]───q(4, 4)───────q(4, 4)───X───q(4, 4)───
           │             │             │                                    │         │                          │             │         │   │
(4, 4): ───q(4, 4)───────q(4, 4)───────q(4, 4)───X──────────────────────────q(4, 4)───×──────────────────────────q(3, 4)───X───q(3, 4)───@───q(3, 4)───
           │             │             │         │                          │                                    │         │   │             │
(5, 4): ───q(5, 4)───────q(5, 4)───@───q(5, 4)───@──────────────────────────q(5, 4)───×──────────────────────────q(7, 4)───@───q(7, 4)───────q(7, 4)───
           │             │         │   │                                    │         │                          │             │             │
(6, 4): ───q(6, 4)───@───q(6, 4)───X───q(6, 4)───×──────────────────────────q(7, 4)───×[cirq.RoutingSwapTag()]───q(5, 4)───────q(5, 4)───────q(5, 4)───
           │         │   │             │         │                          │                                    │             │             │
(7, 4): ───q(7, 4)───X───q(7, 4)───X───q(7, 4)───×[cirq.RoutingSwapTag()]───q(6, 4)──────────────────────────────q(6, 4)───────q(6, 4)───────q(6, 4)───
           │             │         │   │                                    │                                    │             │             │
(8, 4): ───q(8, 4)───────q(8, 4)───@───q(8, 4)──────────────────────────────q(8, 4)──────────────────────────────q(8, 4)───────q(8, 4)───────q(8, 4)───
"""
    cirq.testing.assert_has_diagram(cirq.routed_circuit_with_mapping(circuit), expected_diagram)
