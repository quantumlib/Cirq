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

import pytest

import cirq
from cirq.transformers import gauge_compiling as gc


def test_add_gauge_merges_gates():
    tr = gc.IdleMomentsGauge(2, gauges='pauli')

    circuit = cirq.Circuit.from_moments([], [], [], cirq.X(cirq.q(0)), [], [], cirq.X(cirq.q(0)))
    transformed_circuit = tr(circuit, rng_or_seed=0)

    assert transformed_circuit == cirq.Circuit.from_moments(
        [],
        [],
        [],
        cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1, z_exponent=0)(cirq.q(0)),
        [],
        [],
        cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1, z_exponent=0)(cirq.q(0)),
    )


def test_add_gauge_respects_ignore_tag():
    tr = gc.IdleMomentsGauge(2, gauges='pauli')

    circuit = cirq.Circuit.from_moments(
        cirq.X(cirq.q(0)), [], [], cirq.X(cirq.q(0)).with_tags('ignore')
    )
    transformed_circuit = tr(
        circuit, context=cirq.TransformerContext(tags_to_ignore=("ignore",)), rng_or_seed=0
    )
    assert transformed_circuit == cirq.Circuit.from_moments(
        cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1, z_exponent=0)(cirq.q(0)),
        [],
        cirq.Z(cirq.q(0)),
        cirq.X(cirq.q(0)).with_tags('ignore'),
    )


def test_add_gauge_respects_ignore_moment():
    tr = gc.IdleMomentsGauge(2, gauges='pauli')

    circuit = cirq.Circuit.from_moments(
        cirq.X(cirq.q(0)), [], [], cirq.Moment(cirq.X(cirq.q(0))).with_tags('ignore')
    )
    transformed_circuit = tr(
        circuit, context=cirq.TransformerContext(tags_to_ignore=("ignore",)), rng_or_seed=0
    )
    assert transformed_circuit == cirq.Circuit.from_moments(
        cirq.PhasedXZGate(axis_phase_exponent=0.5, x_exponent=1, z_exponent=0)(cirq.q(0)),
        [],
        cirq.Z(cirq.q(0)),
        cirq.Moment(cirq.X(cirq.q(0))).with_tags('ignore'),
    )


def test_add_gauge_on_prefix():
    tr = gc.IdleMomentsGauge(3, gauges='clifford', gauge_beginning=True)

    circuit = cirq.Circuit.from_moments([], [], [], cirq.CNOT(cirq.q(0), cirq.q(1)))
    transformed_circuit = tr(circuit, rng_or_seed=0)
    assert transformed_circuit == cirq.Circuit.from_moments(
        [
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[20](cirq.q(0)),
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[15](cirq.q(1)),
        ],
        [],
        [
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[20](cirq.q(0)) ** -1,
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[15](cirq.q(1)) ** -1,
        ],
        cirq.CNOT(cirq.q(0), cirq.q(1)),
    )


def test_add_gauge_on_prefix_with_merge():
    tr = gc.IdleMomentsGauge(3, gauges=[cirq.Y], gauge_beginning=True)

    circuit = cirq.Circuit.from_moments([], [], [], cirq.X(cirq.q(0)))
    transformed_circuit = tr(circuit, rng_or_seed=0)
    assert transformed_circuit == cirq.Circuit.from_moments(
        [cirq.Y(cirq.q(0))],
        [],
        [],
        cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1)(cirq.q(0)),
    )


def test_add_gauge_on_suffix():
    tr = gc.IdleMomentsGauge(3, gauges='inv_clifford', gauge_ending=True)

    circuit = cirq.Circuit.from_moments(cirq.CNOT(cirq.q(0), cirq.q(1)), [], [], [])
    transformed_circuit = tr(circuit, rng_or_seed=0)
    assert transformed_circuit == cirq.Circuit.from_moments(
        cirq.CNOT(cirq.q(0), cirq.q(1)),
        [
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[20](cirq.q(0)) ** -1,
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[15](cirq.q(1)) ** -1,
        ],
        [],
        [
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[20](cirq.q(0)),
            cirq.SingleQubitCliffordGate.all_single_qubit_cliffords[15](cirq.q(1)),
        ],
    )


def test_add_gauge_on_suffix_with_merge():
    tr = gc.IdleMomentsGauge(3, gauges=[cirq.Y], gauge_ending=True)

    circuit = cirq.Circuit.from_moments(cirq.X(cirq.q(0)), [], [], [])
    transformed_circuit = tr(circuit, rng_or_seed=0)
    assert transformed_circuit == cirq.Circuit.from_moments(
        cirq.PhasedXZGate(axis_phase_exponent=0, x_exponent=0, z_exponent=1)(cirq.q(0)),
        [],
        [],
        cirq.Y(cirq.q(0)),
    )


def test_add_gauge_respects_min_length():
    tr = gc.IdleMomentsGauge(2, gauges=[cirq.X])

    circuit = cirq.Circuit.from_moments(cirq.X(cirq.q(0)), [], cirq.X(cirq.q(0)))
    transformed_circuit = tr(circuit, rng_or_seed=0)
    assert transformed_circuit == circuit


def test_context_with_deep_raises():
    tr = gc.IdleMomentsGauge(2, gauges=[cirq.X])

    circuit = cirq.Circuit.from_moments(cirq.X(cirq.q(0)), [], cirq.X(cirq.q(0)))
    with pytest.raises(
        ValueError, match="IdleMomentsGauge doesn't support deep TransformerContext"
    ):
        _ = tr(circuit, context=cirq.TransformerContext(deep=True), rng_or_seed=0)


def test_gauge_with_invalid_name_raises():
    with pytest.raises(ValueError, match='valid gauge'):
        _ = gc.IdleMomentsGauge(2, gauges='invalid')


def test_repr():
    assert repr(gc.IdleMomentsGauge(3, gauges='pauli')) == (
        'IdleMomentsGauge(min_length=3, gauges="pauli", '
        'gauge_beginning=False, gauge_ending=False)'
    )
    assert repr(gc.IdleMomentsGauge(4, gauges='clifford')) == (
        'IdleMomentsGauge(min_length=4, gauges="clifford", '
        'gauge_beginning=False, gauge_ending=False)'
    )
    assert repr(gc.IdleMomentsGauge(5, gauges='inv_clifford')) == (
        'IdleMomentsGauge(min_length=5, gauges="inv_clifford", '
        'gauge_beginning=False, gauge_ending=False)'
    )
    assert repr(gc.IdleMomentsGauge(6, gauges=[cirq.X])) == (
        'IdleMomentsGauge(min_length=6, gauges=(cirq.X,), '
        'gauge_beginning=False, gauge_ending=False)'
    )
