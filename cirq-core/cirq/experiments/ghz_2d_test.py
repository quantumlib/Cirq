# Copyright 2025 The Cirq Developers
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

"""Tests for generating and validating 2D GHZ state circuits."""

from typing import cast
from unittest import mock

import networkx as nx
import numpy as np
import pytest

import cirq
import cirq.experiments.ghz_2d as ghz_2d


def _create_mock_graph():
    qubits = cirq.GridQubit.rect(6, 6)
    g = nx.Graph()
    for q in qubits:
        g.add_node(q)
        if q.col + 1 < 6:
            g.add_edge(q, cirq.GridQubit(q.row, q.col + 1))
        if q.row + 1 < 6:
            g.add_edge(q, cirq.GridQubit(q.row + 1, q.col))
    return g, cirq.GridQubit(3, 3)


graph, center_qubit = _create_mock_graph()


@pytest.mark.parametrize("num_qubits", list(range(1, len(graph.nodes) + 1)))
@pytest.mark.parametrize("randomized", [True, False])
@pytest.mark.parametrize("add_dd_and_align_right", [True, False])
def test_ghz_circuits_size(num_qubits: int, randomized: bool, add_dd_and_align_right: bool) -> None:
    """Tests the size of the GHZ circuits."""
    circuit = ghz_2d.generate_2d_ghz_circuit(
        center_qubit,
        graph,
        num_qubits=num_qubits,
        randomized=randomized,
        add_dd_and_align_right=add_dd_and_align_right,
    )
    assert len(circuit.all_qubits()) == num_qubits


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5, 6, 8, 10])
@pytest.mark.parametrize("randomized", [True, False])
@pytest.mark.parametrize("add_dd_and_align_right", [True, False])  # , True
def test_ghz_circuits_state(
    num_qubits: int, randomized: bool, add_dd_and_align_right: bool
) -> None:
    """Tests the state vector form of the GHZ circuits."""

    circuit = ghz_2d.generate_2d_ghz_circuit(
        center_qubit,
        graph,
        num_qubits=num_qubits,
        randomized=randomized,
        add_dd_and_align_right=add_dd_and_align_right,
    )

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    state = result.final_state_vector

    np.testing.assert_allclose(np.abs(state[0]), 1 / np.sqrt(2), atol=1e-7)
    np.testing.assert_allclose(np.abs(state[-1]), 1 / np.sqrt(2), atol=1e-7)

    if num_qubits > 1:
        np.testing.assert_allclose(state[1:-1], 0)


def test_transform_circuit_properties() -> None:
    """Tests that _transform_circuit preserves circuit properties."""
    circuit = ghz_2d.generate_2d_ghz_circuit(
        center_qubit, graph, num_qubits=9, randomized=False, add_dd_and_align_right=False
    )
    transformed_circuit = ghz_2d._transform_circuit(circuit)

    assert transformed_circuit.all_qubits() == circuit.all_qubits()

    assert len(transformed_circuit) >= len(circuit)

    final_moment = transformed_circuit[-1]
    assert not any(isinstance(op.gate, cirq.MeasurementGate) for op in final_moment)

    assert cirq.equal_up_to_global_phase(circuit.unitary(), transformed_circuit.unitary())


def manhattan_distance(q1: cirq.GridQubit, q2: cirq.GridQubit) -> int:
    """Calculates the Manhattan distance between two GridQubits."""
    return abs(q1.row - q2.row) + abs(q1.col - q2.col)


@pytest.mark.parametrize("num_qubits", [2, 4, 9, 15, 20])
def test_ghz_circuits_bfs_order(num_qubits: int) -> None:
    """Verifies that the circuit construction maintains BFS order"""

    circuit = ghz_2d.generate_2d_ghz_circuit(
        center_qubit,
        graph,
        num_qubits=num_qubits,
        randomized=False,  # Test must run on the deterministic BFS order
        add_dd_and_align_right=False,  # Test must run on the raw circuit
    )

    max_dist_seen = 0

    for moment in circuit:
        for op in moment:
            if isinstance(op.gate, cirq.CZPowGate):
                qubits = op.qubits

                dist_q0 = manhattan_distance(center_qubit, cast(cirq.GridQubit, qubits[0]))
                dist_q1 = manhattan_distance(center_qubit, cast(cirq.GridQubit, qubits[1]))

                child_qubit_distance = max(dist_q0, dist_q1)

                if child_qubit_distance > max_dist_seen:
                    assert child_qubit_distance == max_dist_seen + 1
                    max_dist_seen = child_qubit_distance

                assert child_qubit_distance <= max_dist_seen

    included_qubits = circuit.all_qubits()
    if included_qubits:
        max_dist_required = max(
            manhattan_distance(center_qubit, cast(cirq.GridQubit, q)) for q in included_qubits
        )
        assert max_dist_seen == max_dist_required


def test_ghz_invalid_inputs():
    """Tests that the function raises errors for invalid inputs."""

    with pytest.raises(ValueError, match="num_qubits must be a positive integer."):
        ghz_2d.generate_2d_ghz_circuit(center_qubit, graph, num_qubits=0)  # invalid

    with pytest.raises(
        ValueError, match="num_qubits cannot exceed the total number of qubits on the processor."
    ):
        ghz_2d.generate_2d_ghz_circuit(
            center_qubit, graph, num_qubits=len(graph.nodes) + 1  # invalid
        )


def test_assert_dynamical_decoupling_is_applied():
    """Verifies that cirq.transformers.add_dynamical_decoupling is called
    exactly once when the add_dd_and_align_right flag is True.
    """

    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    q2 = cirq.GridQubit(0, 2)
    center_qubit = q0
    graph_3q = nx.Graph([(q0, q1), (q1, q2)])

    with mock.patch('cirq.transformers.add_dynamical_decoupling') as mock_dd:
        mock_dd.return_value = cirq.Circuit()

        # Flag is True
        transformed_circuit_actual = ghz_2d.generate_2d_ghz_circuit(
            center_qubit, graph_3q, num_qubits=3, add_dd_and_align_right=True
        )

        mock_dd.assert_called_once()

        mock_dd.reset_mock()

        # Flag is False
        base_circuit_actual = ghz_2d.generate_2d_ghz_circuit(
            center_qubit, graph_3q, num_qubits=3, add_dd_and_align_right=False
        )

        mock_dd.assert_not_called()

        assert transformed_circuit_actual != base_circuit_actual
