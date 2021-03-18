# Copyright 2021 The Cirq Developers
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
from cirq.testing.devices import ValidatingTestDevice


def test_validating_types_and_qubits():
    dev = ValidatingTestDevice(
        allowed_qubit_types=(cirq.GridQubit,),
        allowed_gates=(cirq.XPowGate,),
        qubits={cirq.GridQubit(0, 0)},
        name='test',
    )

    dev.validate_operation(cirq.X(cirq.GridQubit(0, 0)))

    with pytest.raises(ValueError, match="Unsupported qubit type"):
        dev.validate_operation(cirq.X(cirq.NamedQubit("a")))

    with pytest.raises(ValueError, match="Qubit not on device"):
        dev.validate_operation(cirq.X(cirq.GridQubit(1, 0)))

    with pytest.raises(ValueError, match="Unsupported gate type"):
        dev.validate_operation(cirq.Y(cirq.GridQubit(0, 0)))


def test_validating_locality():
    dev = ValidatingTestDevice(
        allowed_qubit_types=(cirq.GridQubit,),
        allowed_gates=(cirq.CZPowGate, cirq.MeasurementGate),
        qubits=set(cirq.GridQubit.rect(3, 3)),
        name='test',
        validate_locality=True,
    )

    dev.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    dev.validate_operation(cirq.measure(cirq.GridQubit(0, 0), cirq.GridQubit(0, 2)))

    with pytest.raises(ValueError, match="Non-local interaction"):
        dev.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 2)))

    with pytest.raises(ValueError, match="GridQubit must be an allowed qubit type"):
        ValidatingTestDevice(
            allowed_qubit_types=(cirq.NamedQubit,),
            allowed_gates=(cirq.CZPowGate, cirq.MeasurementGate),
            qubits=set(cirq.GridQubit.rect(3, 3)),
            name='test',
            validate_locality=True,
        )


def test_autodecompose():
    dev = ValidatingTestDevice(
        allowed_qubit_types=(cirq.LineQubit,),
        allowed_gates=(
            cirq.XPowGate,
            cirq.ZPowGate,
            cirq.CZPowGate,
            cirq.YPowGate,
            cirq.MeasurementGate,
        ),
        qubits=set(cirq.LineQubit.range(3)),
        name='test',
        validate_locality=False,
        auto_decompose_gates=(cirq.CCXPowGate,),
    )

    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.CCX(a, b, c), device=dev)
    decomposed = cirq.decompose(cirq.CCX(a, b, c))
    assert circuit.moments == cirq.Circuit(decomposed).moments

    with pytest.raises(ValueError, match="Unsupported gate type: cirq.TOFFOLI"):
        dev = ValidatingTestDevice(
            allowed_qubit_types=(cirq.LineQubit,),
            allowed_gates=(
                cirq.XPowGate,
                cirq.ZPowGate,
                cirq.CZPowGate,
                cirq.YPowGate,
                cirq.MeasurementGate,
            ),
            qubits=set(cirq.LineQubit.range(3)),
            name='test',
            validate_locality=False,
            auto_decompose_gates=tuple(),
        )

        a, b, c = cirq.LineQubit.range(3)
        cirq.Circuit(cirq.CCX(a, b, c), device=dev)


def test_repr():
    dev = ValidatingTestDevice(
        allowed_qubit_types=(cirq.GridQubit,),
        allowed_gates=(cirq.CZPowGate, cirq.MeasurementGate),
        qubits=set(cirq.GridQubit.rect(3, 3)),
        name='test',
        validate_locality=True,
    )
    assert repr(dev) == 'test'


def test_defaults():
    dev = ValidatingTestDevice(qubits={cirq.GridQubit(0, 0)})
    assert repr(dev) == 'ValidatingTestDevice'
    assert dev.allowed_qubit_types == (cirq.GridQubit,)
    assert not dev.validate_locality
    assert not dev.auto_decompose_gates
