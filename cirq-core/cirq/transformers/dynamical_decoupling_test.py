# Copyright 2024 The Cirq Developers
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
from cirq import DynamicalDecouplingModel, add_dynamical_decoupling
import pytest


def assert_dd(
    input_circuit: cirq.Circuit, expected_circuit: cirq.Circuit, dd_model: DynamicalDecouplingModel
):
    updated_circuit = add_dynamical_decoupling(input_circuit, dd_model=dd_model)
    cirq.testing.assert_same_circuits(updated_circuit, expected_circuit)


def test_insert_provided_schema():
    a = cirq.NamedQubit("a")
    b = cirq.NamedQubit("b")
    c = cirq.NamedQubit("c")

    # No insertion as there is no room for a dd sequence.
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)), cirq.Moment(cirq.CNOT(a, b)), cirq.Moment(cirq.H(b))
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)), cirq.Moment(cirq.CNOT(a, b)), cirq.Moment(cirq.H(b))
        ),
        dd_model=DynamicalDecouplingModel.from_schema("XX_PAIR"),
    )

    # Insert one XX_PAIR dynamical decoupling sequence in idle moments.
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.measure_each(a, b, c)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c), cirq.X(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.X(a)),
            cirq.Moment(cirq.measure_each(a, b, c)),
        ),
        dd_model=DynamicalDecouplingModel.from_schema("XX_PAIR"),
    )

    # Insert one XX_PAIR dynamical decoupling sequence in idle moments.
    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.measure_each(a, b, c)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c), cirq.Y(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.Y(a)),
            cirq.Moment(cirq.measure_each(a, b, c)),
        ),
        dd_model=DynamicalDecouplingModel.from_schema("YY_PAIR"),
    )


def test_insert_by_customized_dd_sequence():
    a = cirq.NamedQubit("a")
    b = cirq.NamedQubit("b")
    c = cirq.NamedQubit("c")

    assert_dd(
        input_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.CNOT(b, c)),
            cirq.Moment(cirq.measure_each(a, b, c)),
        ),
        expected_circuit=cirq.Circuit(
            cirq.Moment(cirq.H(a)),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(cirq.CNOT(b, c), cirq.X(a)),
            cirq.Moment(cirq.CNOT(b, c), cirq.X(a)),
            cirq.Moment(cirq.measure_each(a, b, c)),
        ),
        dd_model=DynamicalDecouplingModel.from_base_dd_sequence([cirq.XPowGate(), cirq.XPowGate()]),
    )


def test_dd_model_constructor():
    # Succeed
    DynamicalDecouplingModel.from_schema("XX_PAIR")
    DynamicalDecouplingModel.from_schema("YY_PAIR")
    DynamicalDecouplingModel.from_base_dd_sequence(
        [cirq.XPowGate(), cirq.XPowGate(), cirq.YPowGate(), cirq.YPowGate()]
    )
    # Fail
    with pytest.raises(ValueError, match="Specify either schema or base_dd_sequence"):
        DynamicalDecouplingModel()
    with pytest.raises(ValueError, match="Invalid schema name."):
        DynamicalDecouplingModel.from_schema("unimplemented_schema")
    with pytest.raises(ValueError, match="Invalid dynamical decoupling sequence. Expect more than one gates."):
        DynamicalDecouplingModel.from_base_dd_sequence([cirq.XPowGate()])
    with pytest.raises(ValueError, match="Invalid dynamical decoupling sequence"):
        DynamicalDecouplingModel.from_base_dd_sequence([cirq.XPowGate(), cirq.YPowGate()])
