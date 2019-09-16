# Copyright 2019 The Cirq Developers
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
"""Tests for the Quantum Volume benchmarker."""

from dev_tools.profiling import quantum_volume
import cirq


def test_generate_model_circuit():
    """Test that a model circuit is randomly generated."""
    model_circuit = quantum_volume.generate_model_circuit(3, 3)

    assert len(model_circuit) == 25


def test_compute_heavy_set():
    """Test that the heavy set can be computed from a given circuit."""
    a = cirq.GridQubit(0, 0)
    b = cirq.GridQubit(0, 1)
    c = cirq.GridQubit(0, 2)
    model_circuit = cirq.Circuit([
        cirq.Moment([]),
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
        cirq.Moment([]),
        cirq.Moment([cirq.CNOT(a, c)]),
        cirq.Moment([cirq.Z(a), cirq.H(b)]),
        cirq.Moment([cirq.measure(a),
                     cirq.measure(b),
                     cirq.measure(c)])
    ])
    assert quantum_volume.compute_heavy_set(model_circuit, 1234120) == ['101']


def test_args_have_defaults():
    """Test that every argument has a default set."""
    kwargs = quantum_volume.parse_arguments([])
    for _, v in kwargs.items():
        assert v is not None


def test_main_loop():
    """Test that the main loop is able to run without erring."""
    # Keep test from taking a long time by lowering repetitions.
    args = '--num_qubits 5 --depth 5 --num_repetitions 1'.split()
    quantum_volume.main(**quantum_volume.parse_arguments(args))


def test_parse_args():
    """Test that an argument string is parsed correctly."""
    args = ('--num_qubits 5 --depth 5 --num_repetitions 200').split()
    kwargs = quantum_volume.parse_arguments(args)
    assert kwargs == {
        'num_qubits': 5,
        'depth': 5,
        'num_repetitions': 200,
    }
