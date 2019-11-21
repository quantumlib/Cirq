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

import numpy as np
import sympy

import cirq


def test_cz00_str():
    assert str(cirq.CZ00) == 'CZ00'
    assert str(cirq.CZ00**0.5) == 'CZ00**0.5'
    assert str(cirq.CZ00**-0.25) == 'CZ00**-0.25'


def test_cz00_repr():
    assert repr(cirq.CZ00) == 'cirq.CZ00'
    assert repr(cirq.CZ00**0.5) == '(cirq.CZ00**0.5)'
    assert repr(cirq.CZ00**-0.25) == '(cirq.CZ00**-0.25)'


def test_cz00_unitary():
    assert np.allclose(
        cirq.unitary(cirq.CZ00),
        np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(cirq.CZ00**0.5),
        np.array([[1j, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(cirq.CZ00**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(cirq.CZ00**-0.5),
        np.array([[-1j, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))


def test_trace_distance():
    foo = sympy.Symbol('foo')
    scz00 = cirq.CZ00**foo
    assert cirq.trace_distance_bound(scz00) == 1.0

    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.CZ00**(1 / 9)),
                          np.sin(np.pi / 18))


def test_gates():
    # Pick a qubit.
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)

    # Create a circuit
    circuit = cirq.Circuit(
        cirq.CZ00(q0, q1)**0.5,
    )
    cirq.final_wavefunction(circuit)
