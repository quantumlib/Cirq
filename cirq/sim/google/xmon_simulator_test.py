# Copyright 2018 Google LLC
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

"""Tests for xmon_simulator."""

from cirq.sim.google import xmon_simulator
import numpy as np
import pytest


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_initial_state(num_prefix_qubits):
    for initial_state in range(2 ** 3):
        with xmon_simulator.XmonSimulator(
            num_qubits=3,
            num_prefix_qubits=num_prefix_qubits,
            initial_state=initial_state,
            shard_for_small_num_qubits=False) as s:
            expected = np.zeros(2 ** 3, dtype=np.complex64)
            expected[initial_state] = 1.0
            np.testing.assert_almost_equal(expected, s.current_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_reset_state(num_prefix_qubits):
    with xmon_simulator.XmonSimulator(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=0,
        shard_for_small_num_qubits=False) as s:
        for initial_state in range(2 ** 3):
            expected = np.zeros(2 ** 3, dtype=np.complex64)
            expected[initial_state] = 1.0
            s.reset_state(initial_state)
            np.testing.assert_almost_equal(expected, s.current_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_x0(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 0, 0.25, 0.0)
    expected = 1j * np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_x1(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 1, 0.25, 0.0)
    expected = 1j * np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_x2(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 2, 0.25, 0.0)
    expected = 1j * np.array([
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_y0(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 0, 0.25, 0.25)
    expected = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, -1, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_y1(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 1, 0.25, 0.25)
    expected = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_y2(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 2, 0.25, 0.25)
    expected = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [-1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_xy0(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 0, 0.25, 0.125)
    p = np.exp(0.25j * np.pi)
    m = np.exp(-0.25j * np.pi)
    expected = 1j * np.array([
        [0, m, 0, 0, 0, 0, 0, 0],
        [p, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, m, 0, 0, 0, 0],
        [0, 0, p, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, m, 0, 0],
        [0, 0, 0, 0, p, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, m],
        [0, 0, 0, 0, 0, 0, p, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_xy1(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 1, 0.25, 0.125)
    p = np.exp(0.25j * np.pi)
    m = np.exp(-0.25j * np.pi)
    expected = 1j * np.array([
        [0, 0, m, 0, 0, 0, 0, 0],
        [0, 0, 0, m, 0, 0, 0, 0],
        [p, 0, 0, 0, 0, 0, 0, 0],
        [0, p, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, m, 0],
        [0, 0, 0, 0, 0, 0, 0, m],
        [0, 0, 0, 0, p, 0, 0, 0],
        [0, 0, 0, 0, 0, p, 0, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_xy_xy2(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 2, 0.25, 0.125)
    p = np.exp(0.25j * np.pi)
    m = np.exp(-0.25j * np.pi)
    expected = 1j * np.array([
        [0, 0, 0, 0, m, 0, 0, 0],
        [0, 0, 0, 0, 0, m, 0, 0],
        [0, 0, 0, 0, 0, 0, m, 0],
        [0, 0, 0, 0, 0, 0, 0, m],
        [p, 0, 0, 0, 0, 0, 0, 0],
        [0, p, 0, 0, 0, 0, 0, 0],
        [0, 0, p, 0, 0, 0, 0, 0],
        [0, 0, 0, p, 0, 0, 0, 0],
    ])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_phase_z0(num_prefix_qubits):
    result = compute_phases_matrix(num_prefix_qubits, {(0,): 0.25})
    expected = 1j * np.diag([1, -1, 1, -1, 1, -1, 1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_phase_z1(num_prefix_qubits):
    result = compute_phases_matrix(num_prefix_qubits, {(1,): 0.25})
    expected = 1j * np.diag([1, 1, -1, -1, 1, 1, -1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_phase_z2(num_prefix_qubits):
    result = compute_phases_matrix(num_prefix_qubits, {(2,): 0.25})
    expected = 1j * np.diag([1, 1, 1, 1, -1, -1, -1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits,indices',
                         ((0, (0, 1)), (2, (0, 1)), (0, (1, 0)), (2, (1, 0))))
def test_phase_cz01(num_prefix_qubits, indices):
    result = compute_phases_matrix(num_prefix_qubits, {indices: 0.25})
    expected = 1j * np.diag([1, 1, 1, -1, 1, 1, 1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits,indices',
                         ((0, (1, 2)), (2, (1, 2)), (0, (2, 1)), (2, (2, 1))))
def test_phase_cz12(num_prefix_qubits, indices):
    result = compute_phases_matrix(num_prefix_qubits, {indices: 0.25})
    expected = 1j * np.diag([1, 1, 1, 1, 1, 1, -1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits,indices',
                         ((0, (0, 2)), (2, (0, 2)), (0, (2, 0)), (2, (2, 0))))
def test_phase_cz02(num_prefix_qubits, indices):
    result = compute_phases_matrix(num_prefix_qubits, {indices: 0.25})
    expected = 1j * np.diag([1, 1, 1, 1, 1, -1, 1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_multiple_phases(num_prefix_qubits):
    result = compute_phases_matrix(num_prefix_qubits, {
        (1,): 0.25,
        (0, 2): 0.25
    })
    expected = -1 * np.diag([1, 1, -1, -1, 1, -1, -1, 1])
    np.testing.assert_almost_equal(result, expected)


def compute_xy_matrix(num_prefix_qubits, index, turns,
    rotation_axis_turns):
    return compute_matrix(
        num_prefix_qubits,
        fn=lambda s: s.simulate_xy(index, turns, rotation_axis_turns))


def compute_phases_matrix(num_prefix_qubits, phase_map):
    return compute_matrix(
        num_prefix_qubits, fn=lambda s: s.simulate_phases(phase_map))


def compute_matrix(num_prefix_qubits, fn):
    columns = []
    with xmon_simulator.XmonSimulator(
         num_qubits=3,
         num_prefix_qubits=num_prefix_qubits,
         initial_state=0,
         shard_for_small_num_qubits=False) as s:
        for x in range(8):
            s.reset_state(x)
            fn(s)
            columns.append(s.current_state)
    return np.array(columns).transpose()


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_measurement(num_prefix_qubits):
    with xmon_simulator.XmonSimulator(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        shard_for_small_num_qubits=False) as s:
        for i in range(3):
            assert not s.simulate_measurement(i)
            expected = np.zeros(2 ** 3, dtype=np.complex64)
            expected[0] = 1.0
            np.testing.assert_almost_equal(s.current_state, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_measurement_bit_flip(num_prefix_qubits):
    with xmon_simulator.XmonSimulator(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        shard_for_small_num_qubits=False) as s:
        for i in range(3):
            s.simulate_xy(i, 0.25, 0)
        for i in range(3):
            assert s.simulate_measurement(i)
            expected = np.zeros(2 ** 3, dtype=np.complex64)
            # Single qubit operation is jX, so we pick up a -j here.
            expected[7] = -1.0j
            np.testing.assert_almost_equal(s.current_state, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_measurement_state_update(num_prefix_qubits):
    np.random.seed(3)
    with xmon_simulator.XmonSimulator(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        shard_for_small_num_qubits=False) as s:
        # 1/sqrt(2)(I+iX) gate.
        for i in range(3):
            s.simulate_xy(i, 0.125, 0)
        # Check state before measurements.
        single_qubit_state = np.array([1, 1j]) / np.sqrt(2)
        two_qubit_state = np.kron(single_qubit_state, single_qubit_state)
        expected = np.kron(two_qubit_state, single_qubit_state).flatten()
        np.testing.assert_almost_equal(expected, s.current_state)
        assert not s.simulate_measurement(0)
        # Check state after collapse of first qubit state.
        expected = np.kron(two_qubit_state, np.array([1, 0])).flatten()
        np.testing.assert_almost_equal(expected, s.current_state)
        assert not s.simulate_measurement(1)
        # Check state after collapse of second qubit state.
        expected = np.kron(single_qubit_state,
                           np.array([1, 0, 0, 0])).flatten()
        np.testing.assert_almost_equal(expected, s.current_state, decimal=6)
        assert s.simulate_measurement(2)
        expected = np.array([0, 0, 0, 0, 1j, 0, 0, 0])
        # Check final state after collapse of third qubit state.
        np.testing.assert_almost_equal(expected, s.current_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_measurement_randomness_sanity(num_prefix_qubits):
    np.random.seed(15)
    with xmon_simulator.XmonSimulator(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        shard_for_small_num_qubits=False) as s:
        assert_measurements(s, [False, False, False])
        for i in range(3):
            s.simulate_xy(i, 0.125, 0)
        assert_measurements(s, [True, True, False])
        for i in range(3):
            s.simulate_xy(i, 0.125, 0)
        assert_measurements(s, [True, True, True])
        for i in range(3):
            s.simulate_xy(i, 0.125, 0)
        assert_measurements(s, [True, False, True])
        for i in range(3):
            s.simulate_xy(i, 0.125, 0)
        assert_measurements(s, [False, False, False])


def assert_measurements(s, results):
    for i, result in enumerate(results):
        assert result == s.simulate_measurement(i)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_large_circuit_unitary(num_prefix_qubits):
    moments = random_moments(5, 40)
    columns = []
    with xmon_simulator.XmonSimulator(
        num_qubits=5,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=0,
        shard_for_small_num_qubits=False) as s:
        for initial_state in range(2 ** 5):
            s.reset_state(initial_state)
            for moment in moments:
                phase_map = {}
                for op in moment:
                    if op[0] == 'z':
                        phase_map[(op[1],)] = op[2]
                    elif op[0] == 'cz':
                        phase_map[(op[1], op[2])] = op[3]
                    elif op[0] == 'xy':
                        s.simulate_xy(op[1], op[2], op[3])
                s.simulate_phases(phase_map)
            columns.append(s.current_state)
    unitary = np.array(columns).transpose()
    np.testing.assert_almost_equal(
        np.dot(unitary, np.conj(unitary.T)), np.eye(2 ** 5), decimal=6)


def random_moments(num_qubits, num_ops):
    ops = []
    for _ in range(num_ops):
        which = np.random.choice(['z', 'xy', 'cz'])
        if which == 'xy':
            ops.append(('xy', np.random.randint(num_qubits), np.random.random(),
                        np.random.random()))
        elif which == 'z':
            ops.append(('z', np.random.randint(num_qubits), np.random.random()))
        elif which == 'cz':
            ops.append(('z', np.random.randint(num_qubits), np.random.random(),
                        np.random.random()))

    current_moment = num_qubits * [0]
    moments = [[]]

    for op in ops:
        if op[0] == 'xy' or op[0] == 'z':
            index = op[1]
            new_moment = current_moment[index]
            if len(moments) == new_moment:
                moments.append([])
            moments[new_moment].append(op)
            current_moment[index] = new_moment + 1
        elif op[0] == 'cz':
            index0 = op[1]
            index1 = op[2]
            new_moment = max(index0, index1)
            if len(moments) == new_moment:
                moments.append([])
            moments[new_moment].append(op)
            current_moment[index0] = new_moment + 1
            current_moment[index1] = new_moment + 1
    return moments


def _set_global_state(num_prefix_qubits):
    """Sets up global state for testing global level methods."""
    with xmon_simulator.XmonSimulator(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        shard_for_small_num_qubits=False):
        pass


def test_num_prefix_none():
    """Sanity check that setting num_prefix to none still shards correctly."""
    with xmon_simulator.XmonSimulator(
        num_qubits=5, shard_for_small_num_qubits=False) as s:
        expected = np.zeros(2 ** 5, dtype=np.complex64)
        expected[0] = 1.0
        np.testing.assert_almost_equal(expected, s.current_state)


def test_shard_for_small_number_qubits():
    """Sanity check that the no-sharding works with small number of qubits."""
    with xmon_simulator.XmonSimulator(num_qubits=5) as s:
        expected = np.zeros(2 ** 5, dtype=np.complex64)
        expected[0] = 1.0
        np.testing.assert_almost_equal(expected, s.current_state)


def test_shard_for_more_prefix_qubits_than_qubits():
    """Sanity check that the no-sharding works with small number of qubits."""
    with xmon_simulator.XmonSimulator(
        num_qubits=2, num_prefix_qubits=3,
        shard_for_small_num_qubits=False) as s:
        expected = np.zeros(2 ** 2, dtype=np.complex64)
        expected[0] = 1.0
        np.testing.assert_almost_equal(expected, s.current_state)
