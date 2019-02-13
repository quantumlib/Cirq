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

import numpy as np
import pytest

import tensorflow as tf

from cirq.contrib.tpu import (
    circuit_to_tensorflow_runnable
)
import cirq


def _assert_evaluates_correctly(circuit: cirq.Circuit,
                                up_to_global_phase: bool = False):
    r = circuit_to_tensorflow_runnable(circuit)
    v1 = circuit.apply_unitary_effect_to_state(dtype=np.complex64)
    with tf.Session() as session:
        v2 = session.run(r.compute(), feed_dict=r.feed_dict)
    assert v1.shape == v2.shape
    if up_to_global_phase:
        cirq.testing.assert_allclose_up_to_global_phase(v1, v2, atol=1e-6)
    else:
        np.testing.assert_allclose(v1, v2, atol=1e-6)


@pytest.mark.parametrize('n', range(10))
def test_circuit_to_compute_and_feed_dict_small(n: int):
    qs = cirq.LineQubit.range(n)
    c = cirq.Circuit.from_ops(
        [cirq.X(q)**(0.13 * i + 0.1) for i, q in enumerate(qs)],
        [[cirq.CZ(a, b), cirq.X(a)**0.5, cirq.H(b)]
         for a in qs
         for b in qs
         if a < b]
    )
    _assert_evaluates_correctly(c)


def test_circuit_to_compute_and_feed_dict_big():
    n = 16
    qs = cirq.LineQubit.range(n)
    c = cirq.Circuit.from_ops(
        [cirq.H(q) for i, q in enumerate(qs)],
        cirq.CZ(qs[1], qs[3]),
        cirq.CZ(qs[1], qs[10]),
        cirq.CZ(qs[1], qs[14]),
        cirq.CZ(qs[6], qs[14]),
        cirq.CZ(qs[7], qs[14]),
        cirq.CZ(qs[8], qs[14]),
    )
    _assert_evaluates_correctly(c)


def test_circuit_to_compute_and_feed_dict_vector_custom_start_state():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit.from_ops(cirq.CZ(a, b))

    tf.reset_default_graph()
    r = circuit_to_tensorflow_runnable(c, initial_state=0)
    with tf.Session() as session:
        v = session.run(r.compute(), feed_dict=r.feed_dict)
    np.testing.assert_allclose(v, np.array([1, 0, 0, 0]), atol=1e-6)

    tf.reset_default_graph()
    r = circuit_to_tensorflow_runnable(c, initial_state=1)
    with tf.Session() as session:
        v = session.run(r.compute(), feed_dict=r.feed_dict)
    np.testing.assert_allclose(v, np.array([0, 1, 0, 0]), atol=1e-6)

    tf.reset_default_graph()
    r = circuit_to_tensorflow_runnable(c, initial_state=3)
    with tf.Session() as session:
        v = session.run(r.compute(), feed_dict=r.feed_dict)
    np.testing.assert_allclose(v, np.array([0, 0, 0, -1]), atol=1e-6)

    tf.reset_default_graph()
    r = circuit_to_tensorflow_runnable(
        c, initial_state=np.array([0.5, 0.5, 0.5, 0.5]))
    with tf.Session() as session:
        v = session.run(r.compute(), feed_dict=r.feed_dict)
    np.testing.assert_allclose(v, np.array([0.5, 0.5, 0.5, -0.5]), atol=1e-6)

    tf.reset_default_graph()
    with pytest.raises(ValueError):
        _ = circuit_to_tensorflow_runnable(
            c,
            initial_state='zero please')


def test_circuit_to_compute_and_feed_dict_allows_terminal_measurements():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit.from_ops(cirq.H(q), cirq.measure(q), cirq.H(q))
    with pytest.raises(ValueError):
        _ = circuit_to_tensorflow_runnable(c)

    c = cirq.Circuit.from_ops(cirq.H(q), cirq.measure(q))
    _assert_evaluates_correctly(c)


def test_circuit_to_compute_and_feed_dict_works_on_unknown_ops():
    qs = cirq.LineQubit.range(10)

    class PhasedSwapGate(cirq.TwoQubitGate):
        def _unitary_(self):
            return np.array([
                [1, 0, 0, 0],
                [0, 0, -1j, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 1j],
            ])

    phased_swap = PhasedSwapGate()

    c = cirq.Circuit.from_ops(
        [cirq.Y(q)**(0.13 * i + 0.1) for i, q in enumerate(qs)],
        cirq.CCX(qs[0], qs[4], qs[8])**0.5,
        phased_swap(qs[0], qs[1]),
        phased_swap(qs[3], qs[9]),
        phased_swap(qs[0], qs[6]),
        phased_swap(qs[9], qs[8]))
    _assert_evaluates_correctly(c, up_to_global_phase=True)
