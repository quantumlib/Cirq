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

from typing import Iterable, List

from unittest import mock
import numpy as np
import pytest

import cirq
from cirq_google.line.placement.anneal import (
    _STATE,
    AnnealSequenceSearch,
    AnnealSequenceSearchStrategy,
    index_2d,
)
from cirq_google.line.placement.chip import chip_as_adjacency_list


class FakeDevice(cirq.Device):
    def __init__(self, qubits):
        self.qubits = qubits

    @property
    def metadata(self):
        return cirq.DeviceMetadata(self.qubits, None)


def _create_device(qubits: Iterable[cirq.GridQubit]):
    return FakeDevice(qubits)


@mock.patch('cirq_google.line.placement.optimization.anneal_minimize')
def test_search_calls_anneal_minimize(anneal_minimize):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    seqs = [[q00, q01]]
    edges = {(q00, q01)}
    anneal_minimize.return_value = seqs, edges

    assert AnnealSequenceSearch(_create_device([]), seed=0xF00D0000).search() == seqs
    anneal_minimize.assert_called_once_with(
        mock.ANY, mock.ANY, mock.ANY, mock.ANY, trace_func=mock.ANY
    )


@mock.patch('cirq_google.line.placement.optimization.anneal_minimize')
def test_search_calls_anneal_minimize_reversed(anneal_minimize):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    seqs = [[q01, q00]]
    edges = {(q00, q01)}
    anneal_minimize.return_value = seqs, edges

    assert AnnealSequenceSearch(_create_device([]), seed=0xF00D0001).search() == seqs
    anneal_minimize.assert_called_once_with(
        mock.ANY, mock.ANY, mock.ANY, mock.ANY, trace_func=mock.ANY
    )


@mock.patch('cirq_google.line.placement.optimization.anneal_minimize')
def test_search_converts_trace_func(anneal_minimize):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    seqs = [[q00, q01]]
    edges = {(q00, q01)}
    anneal_minimize.return_value = seqs, edges
    trace_func = mock.Mock()

    assert (
        AnnealSequenceSearch(_create_device([]), seed=0xF00D0002).search(trace_func=trace_func)
        == seqs
    )
    wrapper_func = anneal_minimize.call_args[1]['trace_func']

    wrapper_func((seqs, edges), 1.0, 2.0, 3.0, True)
    trace_func.assert_called_once_with(seqs, 1.0, 2.0, 3.0, True)


def test_quadratic_sum_cost_calculates_quadratic_cost():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q03 = cirq.GridQubit(0, 3)

    def calculate_cost(seqs: List[List[cirq.GridQubit]]):
        qubits: List[cirq.GridQubit] = []
        for seq in seqs:
            qubits += seq
        return AnnealSequenceSearch(_create_device(qubits), seed=0xF00D0003)._quadratic_sum_cost(
            (seqs, set())
        )

    assert np.isclose(calculate_cost([[q00]]), -1.0)
    assert np.isclose(calculate_cost([[q00, q01]]), -1.0)
    assert np.isclose(calculate_cost([[q00], [q01]]), -(0.5**2 + 0.5**2))
    assert np.isclose(calculate_cost([[q00], [q01, q02, q03]]), -(0.25**2 + 0.75**2))


def test_force_edges_active_move_does_not_change_input():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    search = AnnealSequenceSearch(_create_device([q00, q01, q10, q11]), seed=0xF00D0004)
    seqs, edges = search._create_initial_solution()
    seqs_copy, edges_copy = list(seqs), edges.copy()
    search._force_edges_active_move((seqs, edges))
    assert seqs == seqs_copy
    assert edges == edges_copy


def test_force_edges_active_move_calls_force_edge_active_move():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    search = AnnealSequenceSearch(_create_device([q00, q01, q10, q11]), seed=0xF00D0005)
    with mock.patch.object(search, '_force_edge_active_move') as force_edge_active_move:
        search._force_edges_active_move(search._create_initial_solution())
        force_edge_active_move.assert_called_with(mock.ANY)


def test_force_edge_active_move_does_not_change_input():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    search = AnnealSequenceSearch(_create_device([q00, q01, q10, q11]), seed=0xF00D0006)
    seqs, edges = search._create_initial_solution()
    seqs_copy, edges_copy = list(seqs), edges.copy()
    search._force_edge_active_move((seqs, edges))
    assert seqs_copy == seqs
    assert edges_copy == edges


def test_force_edge_active_move_quits_when_no_free_edge():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    search = AnnealSequenceSearch(_create_device([q00, q01]), seed=0xF00D0007)
    seqs, edges = search._create_initial_solution()
    assert search._force_edge_active_move((seqs, edges)) == (seqs, edges)


def test_force_edge_active_move_calls_force_edge_active():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q02 = cirq.GridQubit(0, 2)
    q12 = cirq.GridQubit(1, 2)
    device = _create_device([q00, q01, q10, q11, q02, q12])
    search = AnnealSequenceSearch(device, seed=0xF00D0008)
    with mock.patch.object(search, '_force_edge_active') as force_edge_active:
        solution = search._create_initial_solution()
        search._force_edge_active_move(solution)
        force_edge_active.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY)

        _, args, _ = force_edge_active.mock_calls[0]
        seqs, edge, _ = args

        # Verify that edge is not active edge, on some sequence.
        for seq in seqs:
            for i in range(1, len(seq)):
                assert not (seq[i - 1] == edge[0] and seq[i] == edge[1])
                assert not (seq[i - 1] == edge[1] and seq[i] == edge[0])

        # Verify that edge is a valid edge
        assert edge[0] in chip_as_adjacency_list(device)[edge[1]]


def test_force_edge_active_creates_valid_solution_different_sequnces():
    q00, q10, q20, q30 = [cirq.GridQubit(x, 0) for x in range(4)]
    q01, q11, q21, q31 = [cirq.GridQubit(x, 1) for x in range(4)]
    qubits = [q00, q10, q20, q30, q01, q11, q21, q31]
    search = AnnealSequenceSearch(_create_device(qubits), seed=0xF00D0009)

    # +-+-+-+ -> +-+-+-+
    #            |
    # +-+-+-+    +-+-+-+
    assert search._force_edge_active(
        [[q00, q10, q20, q30], [q01, q11, q21, q31]], (q00, q01), lambda: True
    ) == [[q30, q20, q10, q00, q01, q11, q21, q31]]

    # +-+-+-+ -> +-+-+-+
    #                  |
    # +-+-+-+    +-+-+-+
    assert search._force_edge_active(
        [[q00, q10, q20, q30], [q01, q11, q21, q31]], (q30, q31), lambda: True
    ) == [[q00, q10, q20, q30, q31, q21, q11, q01]]

    # +-+-+-+ -> + +-+-+
    #              |
    # +-+-+-+    + +-+-+
    assert search._force_edge_active(
        [[q00, q10, q20, q30], [q01, q11, q21, q31]], (q10, q11), lambda: True
    ) == [[q30, q20, q10, q11, q21, q31], [q00], [q01]]

    # +-+-+-+ -> +-+ +-+
    #              |
    # +-+-+-+    +-+ +-+
    assert search._force_edge_active(
        [[q00, q10, q20, q30], [q01, q11, q21, q31]], (q10, q11), lambda: False
    ) == [[q00, q10, q11, q01], [q20, q30], [q21, q31]]


def test_force_edge_active_creates_valid_solution_single_sequence():
    q00, q10, q20, q30 = [cirq.GridQubit(x, 0) for x in range(4)]
    q01, q11, q21, q31 = [cirq.GridQubit(x, 1) for x in range(4)]
    c = [q00, q10, q20, q30, q01, q11, q21, q31]
    search = AnnealSequenceSearch(_create_device(c), seed=0xF00D0010)

    # +-+-+-+ -> +-+-+ +
    # |          |     |
    # +-+-+-+    +-+-+-+
    assert search._force_edge_active(
        [[q30, q20, q10, q00, q01, q11, q21, q31]], (q30, q31), lambda: True
    ) == [[q20, q10, q00, q01, q11, q21, q31, q30]]

    # +-+-+-+ -> +-+-+-+
    # |          |     |
    # +-+-+-+    +-+-+ +
    assert search._force_edge_active(
        [[q31, q21, q11, q01, q00, q10, q20, q30]], (q30, q31), lambda: True
    ) == [[q21, q11, q01, q00, q10, q20, q30, q31]]

    # +-+-+-+ -> +-+-+ +
    # |          |     |
    # +-+-+-+    +-+-+ +
    assert search._force_edge_active(
        [[q30, q20, q10, q00, q01, q11, q21, q31]], (q30, q31), lambda: False
    ) == [[q30, q31], [q20, q10, q00, q01, q11, q21]]

    # +-+-+-+ -> +-+ +-+
    # |          |   |
    # +-+-+-+    +-+-+ +
    assert search._force_edge_active(
        [[q30, q20, q10, q00, q01, q11, q21, q31]], (q20, q21), lambda: True
    ) == [[q10, q00, q01, q11, q21, q20, q30], [q31]]

    # +-+-+-+ -> +-+ +-+
    # |          |   |
    # +-+-+-+    +-+ +-+
    assert search._force_edge_active(
        [[q30, q20, q10, q00, q01, q11, q21, q31]], (q20, q21), lambda: False
    ) == [[q30, q20, q21, q31], [q10, q00, q01, q11]]

    # +-+-+-+ -> +-+ +-+
    # |          |   |
    # +-+-+ +    +-+-+ +
    assert search._force_edge_active(
        [[q30, q20, q10, q00, q01, q11, q21], [q31]], (q20, q21), lambda: True
    ) == [[q31], [q10, q00, q01, q11, q21, q20, q30]]

    # +-+-+-+ -> +-+ +-+
    # |          |   |
    # +-+-+ +    +-+ + +
    assert search._force_edge_active(
        [[q30, q20, q10, q00, q01, q11, q21], [q31]], (q20, q21), lambda: False
    ) == [[q31], [q30, q20, q21], [q10, q00, q01, q11]]

    # +-+-+ + -> +-+ + +
    # |          |   |
    # +-+-+-+    +-+-+ +
    assert search._force_edge_active(
        [[q20, q10, q00, q01, q11, q21, q31], [q30]], (q20, q21), lambda: True
    ) == [[q30], [q10, q00, q01, q11, q21, q20], [q31]]

    # +-+-+ + -> +-+-+ +
    # |          |   |
    # +-+-+-+    +-+ +-+
    samples = iter([True, False])
    assert search._force_edge_active(
        [[q20, q10, q00, q01, q11, q21, q31], [q30]], (q20, q21), lambda: next(samples)
    ) == [[q30], [q31, q21, q20, q10, q00, q01, q11]]

    # +-+-+ + -> +-+ + +
    # |          |   |
    # +-+-+-+    +-+ +-+
    assert search._force_edge_active(
        [[q20, q10, q00, q01, q11, q21, q31], [q30]], (q20, q21), lambda: False
    ) == [[q30], [q20, q21, q31], [q10, q00, q01, q11]]


def test_create_initial_solution_creates_valid_solution():
    def check_chip(qubits: List[cirq.GridQubit]):
        _verify_valid_state(
            qubits,
            AnnealSequenceSearch(
                _create_device(qubits), seed=0xF00D0011
            )._create_initial_solution(),
        )

    q00, q01, q02 = [cirq.GridQubit(0, x) for x in range(3)]
    q10, q11, q12 = [cirq.GridQubit(1, x) for x in range(3)]

    check_chip([q00, q10])
    check_chip([q00, q10, q01])
    check_chip([q00, q10, q01])
    check_chip([q00, q10, q01, q11])
    check_chip([q00, q10, q02, q12])
    check_chip([q00, q10, q11, q02])
    check_chip([q00, q10, q02])
    check_chip([q00, q10, q01, q11, q02, q12])


def test_normalize_edge_normalizes():
    q00, q01 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    q10, q11 = cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)
    search = AnnealSequenceSearch(_create_device([]), seed=0xF00D0012)

    assert search._normalize_edge((q00, q01)) == (q00, q01)
    assert search._normalize_edge((q01, q00)) == (q00, q01)

    assert search._normalize_edge((q01, q10)) == (q01, q10)
    assert search._normalize_edge((q10, q01)) == (q01, q10)

    assert search._normalize_edge((q00, q11)) == (q00, q11)
    assert search._normalize_edge((q11, q00)) == (q00, q11)


def test_choose_random_edge_chooses():
    q00, q11, q22 = [cirq.GridQubit(x, x) for x in range(3)]
    e0, e1, e2 = (q00, q11), (q11, q22), (q22, q00)
    search = AnnealSequenceSearch(_create_device([]), seed=0xF00D0013)
    assert search._choose_random_edge(set()) is None
    assert search._choose_random_edge({e0}) == e0
    assert search._choose_random_edge({e0, e1, e2}) in [e0, e1, e2]


def _verify_valid_state(qubits: List[cirq.GridQubit], state: _STATE):
    seqs, edges = state
    search = AnnealSequenceSearch(_create_device(qubits), seed=0xF00D0014)
    c_adj = chip_as_adjacency_list(_create_device(qubits))

    # Check if every edge is normalized
    for e in edges:
        assert search._normalize_edge(e) == e

    # Check if every edge is valid
    for n0, n1 in edges:
        assert n0 in c_adj[n1]

    # Check if every edge is present
    for n0 in c_adj:
        for n1 in c_adj[n0]:
            assert (n0, n1) in edges or (n1, n0) in edges
            assert (n0, n1) in edges or (n1, n0) in edges

    c_set = set(qubits)

    # Check if every node in the sequences appears exactly once
    for seq in seqs:
        for n in seq:
            c_set.remove(n)

    # Check that no node is missing
    assert not c_set


def test_anneal_search_method_calls():
    q00, q01 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    device = _create_device([q00, q01])
    length = 1
    seed = 1

    method = AnnealSequenceSearchStrategy(None, seed)
    assert len(method.place_line(device, length)) == length


def test_index_2d():
    assert index_2d([[1, 2], [3]], 1) == (0, 0)
    assert index_2d([[1, 2], [3]], 2) == (0, 1)
    assert index_2d([[1, 2], [3]], 3) == (1, 0)
    with pytest.raises(ValueError):
        _ = index_2d([[1, 2], [3]], 4)

    with pytest.raises(ValueError):
        _ = index_2d([], 1)
    with pytest.raises(ValueError):
        _ = index_2d([[]], 1)

    assert index_2d([[], ['a']], 'a') == (1, 0)
    assert index_2d([['a', 'a']], 'a') == (0, 0)
    assert index_2d([['a'], ['a']], 'a') == (0, 0)
    assert index_2d([['a', 'a'], ['a']], 'a') == (0, 0)
