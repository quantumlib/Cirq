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

from cirq.sim.google import xmon_stepper
import numpy as np
import pytest


def test_no_thread_pool():
    pool = xmon_stepper.ThreadlessPool()
    result = pool.map(lambda x: x + 1, range(10))
    assert result == [x + 1 for x in range(10)]
    # No ops.
    pool.close()
    pool.join()


def test_no_thread_pool_no_chunking():
    pool = xmon_stepper.ThreadlessPool()
    with pytest.raises(AssertionError):
        pool.map(lambda x: x + 1, range(10), chunksize=1)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_initial_state_computational_basis(num_prefix_qubits):
    for initial_state in range(2 ** 3):
        with xmon_stepper.Stepper(
            num_qubits=3,
            num_prefix_qubits=num_prefix_qubits,
            initial_state=initial_state,
            min_qubits_before_shard=0) as s:
            expected = np.zeros(2 ** 3, dtype=np.complex64)
            expected[initial_state] = 1.0
            np.testing.assert_almost_equal(expected, s.current_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_initial_state_full_state(num_prefix_qubits):
    initial_state = np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0],
                             dtype=np.complex64)
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=initial_state,
        min_qubits_before_shard=0) as s:
        np.testing.assert_almost_equal(initial_state, s.current_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_initial_state_full_state_complex(num_prefix_qubits):
    initial_state = np.array([0.5j, 0.5, 0.5, 0.5, 0, 0, 0, 0],
                             dtype=np.complex64)
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=initial_state,
        min_qubits_before_shard=0) as s:
        np.testing.assert_almost_equal(initial_state, s.current_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_initial_state_wrong_dtype(num_prefix_qubits):
    initial_state = np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0], dtype=np.float32)
    with pytest.raises(ValueError):
        xmon_stepper.Stepper(
            num_qubits=3,
            num_prefix_qubits=num_prefix_qubits,
            initial_state=initial_state,
            min_qubits_before_shard=0)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_initial_state_not_normalized(num_prefix_qubits):
    initial_state = np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    with pytest.raises(ValueError):
        xmon_stepper.Stepper(
            num_qubits=3,
            num_prefix_qubits=num_prefix_qubits,
            initial_state=initial_state,
            min_qubits_before_shard=0)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_reset_state_computational_basis(num_prefix_qubits):
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=0,
        min_qubits_before_shard=0) as s:
        for initial_state in range(2 ** 3):
            expected = np.zeros(2 ** 3, dtype=np.complex64)
            expected[initial_state] = 1.0
            s.reset_state(initial_state)
            np.testing.assert_almost_equal(expected, s.current_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_reset_state_full_state(num_prefix_qubits):
    reset_state = np.array([0.5, 0.5, 0.5, 0.5, 0, 0, 0, 0], dtype=np.complex64)
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=0,
        min_qubits_before_shard=0) as s:
        s.reset_state(reset_state)
        np.testing.assert_almost_equal(reset_state, s.current_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_reset_state_not_normalized(num_prefix_qubits):
    reset_state = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0], dtype=np.complex64)
    with pytest.raises(ValueError):
        with xmon_stepper.Stepper(
            num_qubits=3,
            num_prefix_qubits=num_prefix_qubits,
            initial_state=0,
            min_qubits_before_shard=0) as s:
            s.reset_state(reset_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_reset_state_wrong_dtype(num_prefix_qubits):
    reset_state = np.array([0.5, 0.5, 0.5, 0, 0, 0, 0, 0], dtype=np.float32)
    with pytest.raises(ValueError):
        with xmon_stepper.Stepper(
            num_qubits=3,
            num_prefix_qubits=num_prefix_qubits,
            initial_state=0,
            min_qubits_before_shard=0) as s:
            s.reset_state(reset_state)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_w_x0(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 0, 1.0, 0.0)
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
def test_w_x1(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 1, 1.0, 0.0)
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
def test_w_x2(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 2, 1.0, 0.0)
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
def test_w_y0(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 0, 1.0, 0.5)
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
def test_w_y1(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 1, 1.0, 0.5)
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
def test_w_y2(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 2, 1.0, 0.5)
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
def test_w_xplusy0(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 0, 1.0, 0.25)
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
def test_w_xplusy1(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 1, 1.0, 0.25)
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
def test_w_xplusy2(num_prefix_qubits):
    result = compute_xy_matrix(num_prefix_qubits, 2, 1.0, 0.25)
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
    result = compute_phases_matrix(num_prefix_qubits, {(0,): 1.0})
    expected = 1j * np.diag([1, -1, 1, -1, 1, -1, 1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_phase_z1(num_prefix_qubits):
    result = compute_phases_matrix(num_prefix_qubits, {(1,): 1.0})
    expected = 1j * np.diag([1, 1, -1, -1, 1, 1, -1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_phase_z2(num_prefix_qubits):
    result = compute_phases_matrix(num_prefix_qubits, {(2,): 1.0})
    expected = 1j * np.diag([1, 1, 1, 1, -1, -1, -1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits,indices',
                         ((0, (0, 1)), (2, (0, 1)), (0, (1, 0)), (2, (1, 0))))
def test_phase_cz01(num_prefix_qubits, indices):
    result = compute_phases_matrix(num_prefix_qubits, {indices: 1.0})
    expected = np.diag([1, 1, 1, -1, 1, 1, 1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits,indices',
                         ((0, (1, 2)), (2, (1, 2)), (0, (2, 1)), (2, (2, 1))))
def test_phase_cz12(num_prefix_qubits, indices):
    result = compute_phases_matrix(num_prefix_qubits, {indices: 1.0})
    expected = np.diag([1, 1, 1, 1, 1, 1, -1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits,indices',
                         ((0, (0, 2)), (2, (0, 2)), (0, (2, 0)), (2, (2, 0))))
def test_phase_cz02(num_prefix_qubits, indices):
    result = compute_phases_matrix(num_prefix_qubits, {indices: 1.0})
    expected = np.diag([1, 1, 1, 1, 1, -1, 1, -1])
    np.testing.assert_almost_equal(result, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_multiple_phases(num_prefix_qubits):
    result = compute_phases_matrix(num_prefix_qubits, {
        (1,): 1.0,
        (0, 2): 1.0
    })
    expected = 1j * np.diag([1, 1, -1, -1, 1, -1, -1, 1])
    np.testing.assert_almost_equal(result, expected)


def compute_xy_matrix(num_prefix_qubits, index, turns,
                      rotation_axis_turns):
    return compute_matrix(
        num_prefix_qubits,
        fn=lambda s: s.simulate_w(index, turns, rotation_axis_turns))


def compute_phases_matrix(num_prefix_qubits, phase_map):
    return compute_matrix(
        num_prefix_qubits, fn=lambda s: s.simulate_phases(phase_map))


def compute_matrix(num_prefix_qubits, fn):
    columns = []
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=0,
        min_qubits_before_shard=0) as s:
        for x in range(8):
            s.reset_state(x)
            fn(s)
            columns.append(s.current_state)
    return np.array(columns).transpose()


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_measurement(num_prefix_qubits):
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        min_qubits_before_shard=0) as s:
        for i in range(3):
            assert not s.simulate_measurement(i)
            expected = np.zeros(2 ** 3, dtype=np.complex64)
            expected[0] = 1.0
            np.testing.assert_almost_equal(s.current_state, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_measurement_bit_flip(num_prefix_qubits):
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        min_qubits_before_shard=0) as s:
        for i in range(3):
            s.simulate_w(i, 1.0, 0)
        for i in range(3):
            assert s.simulate_measurement(i)
            expected = np.zeros(2 ** 3, dtype=np.complex64)
            # Single qubit operation is jX, so we pick up a -j here.
            expected[7] = -1.0j
            np.testing.assert_almost_equal(s.current_state, expected)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_measurement_state_update(num_prefix_qubits):
    np.random.seed(3)
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        min_qubits_before_shard=0) as s:
        # 1/sqrt(2)(I+iX) gate.
        for i in range(3):
            s.simulate_w(i, 0.5, 0)
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
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        min_qubits_before_shard=0) as s:
        assert_measurements(s, [False, False, False])
        for i in range(3):
            s.simulate_w(i, 0.5, 0)
        assert_measurements(s, [True, True, False])
        for i in range(3):
            s.simulate_w(i, 0.5, 0)
        assert_measurements(s, [True, True, True])
        for i in range(3):
            s.simulate_w(i, 0.5, 0)
        assert_measurements(s, [True, False, True])
        for i in range(3):
            s.simulate_w(i, 0.5, 0)
        assert_measurements(s, [False, False, False])


def assert_measurements(s, results):
    for i, result in enumerate(results):
        assert result == s.simulate_measurement(i)


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_non_context_manager(num_prefix_qubits):
    np.random.seed(15)
    stepper = xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=1,
        min_qubits_before_shard=0)
    np.testing.assert_almost_equal(stepper.current_state,
                                   np.array([0, 1, 0, 0, 0, 0, 0, 0],
                                            dtype=np.complex64))
    stepper.__exit__()

    stepper.reset_state(0)
    np.testing.assert_almost_equal(stepper.current_state,
                                   np.array([1, 0, 0, 0, 0, 0, 0, 0],
                                            dtype=np.complex64))
    stepper.__exit__()

    stepper.simulate_w(0, 1.0, 0)
    np.testing.assert_almost_equal(stepper.current_state,
                                   np.array([0, 1j, 0, 0, 0, 0, 0, 0],
                                            dtype=np.complex64))
    stepper.__exit__()

    stepper.simulate_phases({(0, ): 1.0})
    np.testing.assert_almost_equal(stepper.current_state,
                                   np.array([0, 1, 0, 0, 0, 0, 0, 0],
                                            dtype=np.complex64))
    stepper.__exit__()

    result = stepper.simulate_measurement(0)
    assert result
    stepper.__exit__()


@pytest.mark.parametrize('num_prefix_qubits', (0, 2))
def test_large_circuit_unitary(num_prefix_qubits):
    moments = random_moments(5, 40)
    columns = []
    with xmon_stepper.Stepper(
        num_qubits=5,
        num_prefix_qubits=num_prefix_qubits,
        initial_state=0,
        min_qubits_before_shard=0) as s:
        for initial_state in range(2 ** 5):
            s.reset_state(initial_state)
            for moment in moments:
                phase_map = {}
                for op in moment:
                    if op[0] == 'expz':
                        phase_map[(op[1],)] = op[2]
                    elif op[0] == 'exp11':
                        phase_map[(op[1], op[2])] = op[3]
                    elif op[0] == 'expw':
                        s.simulate_w(op[1], op[2], op[3])
                s.simulate_phases(phase_map)
            columns.append(s.current_state)
    unitary = np.matrix(columns).transpose()
    np.testing.assert_almost_equal(
        np.dot(unitary, unitary.H), np.eye(2 ** 5), decimal=6)


def random_moments(num_qubits, num_ops):
    ops = []
    for _ in range(num_ops):
        which = np.random.choice(['expz', 'expw', 'exp11'])
        if which == 'expw':
            ops.append(
                ('expw', np.random.randint(num_qubits), 2 * np.random.random(),
                 2 * np.random.random()))
        elif which == 'expz':
            ops.append(
                ('expz', np.random.randint(num_qubits), 2 * np.random.random()))
        elif which == 'exp11':
            ops.append(
                ('exp11', np.random.randint(num_qubits),
                 np.random.randint(num_qubits),
                 2 * np.random.random()))
    current_moment = num_qubits * [0]
    moments = [[]]

    for op in ops:
        if op[0] == 'expw' or op[0] == 'expz':
            index = op[1]
            new_moment = current_moment[index]
            if len(moments) == new_moment:
                moments.append([])
            moments[new_moment].append(op)
            current_moment[index] = new_moment + 1
        elif op[0] == 'exp11':
            index0 = op[1]
            index1 = op[2]
            new_moment = max(current_moment[index0], current_moment[index1])
            if len(moments) == new_moment:
                moments.append([])
            moments[new_moment].append(op)
            current_moment[index0] = new_moment + 1
            current_moment[index1] = new_moment + 1
    return moments


def _set_global_state(num_prefix_qubits):
    """Sets up global state for testing global level methods."""
    with xmon_stepper.Stepper(
        num_qubits=3,
        num_prefix_qubits=num_prefix_qubits,
        min_qubits_before_shard=0):
        pass


def test_num_prefix_none():
    """Sanity check that setting num_prefix to none still shards correctly."""
    with xmon_stepper.Stepper(
        num_qubits=5, min_qubits_before_shard=0) as s:
        expected = np.zeros(2 ** 5, dtype=np.complex64)
        expected[0] = 1.0
        np.testing.assert_almost_equal(expected, s.current_state)


def test_shard_for_small_number_qubits():
    """Sanity check that the no-sharding works with small number of qubits."""
    with xmon_stepper.Stepper(num_qubits=5) as s:
        expected = np.zeros(2 ** 5, dtype=np.complex64)
        expected[0] = 1.0
        np.testing.assert_almost_equal(expected, s.current_state)


def test_shard_for_more_prefix_qubits_than_qubits():
    """Sanity check that the no-sharding works with small number of qubits."""
    with xmon_stepper.Stepper(
        num_qubits=2, num_prefix_qubits=3,
        min_qubits_before_shard=0) as s:
        expected = np.zeros(2 ** 2, dtype=np.complex64)
        expected[0] = 1.0
        np.testing.assert_almost_equal(expected, s.current_state)
