# Copyright 2022 The Cirq Developers
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

from datetime import timedelta
from typing import List

import pytest

import cirq
from cirq_aqt import aqt_device, aqt_device_metadata


@pytest.fixture
def qubits() -> List[cirq.LineQubit]:
    return cirq.LineQubit.range(3)


@pytest.fixture
def device(qubits) -> aqt_device.AQTDevice:
    ms = cirq.Duration(millis=1)
    return aqt_device.AQTDevice(
        measurement_duration=100 * ms,
        twoq_gates_duration=200 * ms,
        oneq_gates_duration=10 * ms,
        qubits=qubits,
    )


class NotImplementedOperation(cirq.Operation):
    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        raise NotImplementedError()


def test_init_qubits(device, qubits):
    ms = cirq.Duration(millis=1)
    assert device.qubits == frozenset(qubits)
    with pytest.raises(TypeError, match="NamedQubit"):
        aqt_device.AQTDevice(
            measurement_duration=100 * ms,
            twoq_gates_duration=200 * ms,
            oneq_gates_duration=10 * ms,
            qubits=[cirq.LineQubit(0), cirq.NamedQubit("a")],
        )


@pytest.mark.parametrize('ms', [cirq.Duration(millis=1), timedelta(milliseconds=1)])
def test_init_durations(ms, qubits):
    dev = aqt_device.AQTDevice(
        qubits=qubits,
        measurement_duration=100 * ms,
        twoq_gates_duration=200 * ms,
        oneq_gates_duration=10 * ms,
    )
    assert dev.metadata.twoq_gates_duration == cirq.Duration(millis=200)
    assert dev.metadata.oneq_gates_duration == cirq.Duration(millis=10)
    assert dev.metadata.measurement_duration == cirq.Duration(millis=100)


def test_metadata(device, qubits):
    assert isinstance(device.metadata, aqt_device_metadata.AQTDeviceMetadata)
    assert device.metadata.qubit_set == frozenset(qubits)


def test_repr(device):
    assert repr(device) == (
        "cirq_aqt.aqt_device.AQTDevice("
        "measurement_duration=cirq.Duration(millis=100), "
        "twoq_gates_duration=cirq.Duration(millis=200), "
        "oneq_gates_duration=cirq.Duration(millis=10), "
        "qubits=[cirq.LineQubit(0), cirq.LineQubit(1), "
        "cirq.LineQubit(2)])"
    )
    cirq.testing.assert_equivalent_repr(device, setup_code='import cirq\nimport cirq_aqt\n')


def test_validate_measurement_non_adjacent_qubits_ok(device):
    device.validate_operation(
        cirq.GateOperation(cirq.MeasurementGate(2, 'key'), (cirq.LineQubit(0), cirq.LineQubit(1)))
    )


def test_validate_operation_existing_qubits(device):
    device.validate_operation(cirq.GateOperation(cirq.XX, (cirq.LineQubit(0), cirq.LineQubit(1))))
    device.validate_operation(cirq.Z(cirq.LineQubit(0)))
    device.validate_operation(
        cirq.PhasedXPowGate(phase_exponent=0.75, exponent=0.25, global_shift=0.1).on(
            cirq.LineQubit(1)
        )
    )

    with pytest.raises(ValueError):
        device.validate_operation(cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(-1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.Z(cirq.LineQubit(-1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.CZ(cirq.LineQubit(1), cirq.LineQubit(1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.X(cirq.NamedQubit("q1")))


def test_validate_operation_supported_gate(device):
    class MyGate(cirq.Gate):
        def num_qubits(self):
            return 1

    device.validate_operation(cirq.GateOperation(cirq.Z, [cirq.LineQubit(0)]))

    assert MyGate().num_qubits() == 1
    with pytest.raises(ValueError):
        device.validate_operation(cirq.GateOperation(MyGate(), [cirq.LineQubit(0)]))
    with pytest.raises(ValueError):
        device.validate_operation(NotImplementedOperation())


def test_aqt_device_eq(device):
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: device)


def test_validate_circuit_repeat_measurement_keys(device):
    circuit = cirq.Circuit()
    circuit.append(
        [cirq.measure(cirq.LineQubit(0), key='a'), cirq.measure(cirq.LineQubit(1), key='a')]
    )

    with pytest.raises(ValueError, match='Measurement key a repeated'):
        device.validate_circuit(circuit)


def test_aqt_device_str(device):
    assert str(device) == "q(0)───q(1)───q(2)"


def test_aqt_device_pretty_repr(device):
    cirq.testing.assert_repr_pretty(device, "q(0)───q(1)───q(2)")
    cirq.testing.assert_repr_pretty(device, "AQTDevice(...)", cycle=True)


def test_at(device):
    assert device.at(-1) is None
    assert device.at(0) == cirq.LineQubit(0)
    assert device.at(2) == cirq.LineQubit(2)
