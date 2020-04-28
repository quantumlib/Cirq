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

from datetime import timedelta
import pytest

import numpy as np

import cirq
import cirq.ion as ci


def ion_device(chain_length: int, use_timedelta=False) -> ci.IonDevice:
    ms = (1000 * cirq.Duration(nanos=1) if not use_timedelta else timedelta(
        microseconds=1))
    return ci.IonDevice(  # type: ignore
        measurement_duration=100 * ms,  # type: ignore
        twoq_gates_duration=200 * ms,  # type: ignore
        oneq_gates_duration=10 * ms,  # type: ignore
        qubits=cirq.LineQubit.range(chain_length))


class NotImplementedOperation(cirq.Operation):
    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        raise NotImplementedError()


def test_init():
    d = ion_device(3)
    ms = 1000 * cirq.Duration(nanos=1)
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)

    assert d.qubits == {q0, q1, q2}
    assert d.duration_of(cirq.Z(q0)) == 10 * ms
    assert d.duration_of(cirq.measure(q0)) == 100 * ms
    assert d.duration_of(cirq.measure(q0, q1)) == 100 * ms
    assert d.duration_of(cirq.ops.XX(q0, q1)) == 200 * ms
    with pytest.raises(ValueError):
        _ = d.duration_of(cirq.SingleQubitGate().on(q0))


def test_init_timedelta():
    d = ion_device(3, use_timedelta=True)
    ms = 1000*cirq.Duration(nanos=1)
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)

    assert d.qubits == {q0, q1, q2}
    assert d.duration_of(cirq.Z(q0)) == 10 * ms
    assert d.duration_of(cirq.measure(q0)) == 100 * ms
    assert d.duration_of(cirq.measure(q0, q1)) == 100 * ms
    assert d.duration_of(cirq.ops.XX(q0, q1)) == 200 * ms
    with pytest.raises(ValueError):
        _ = d.duration_of(cirq.SingleQubitGate().on(q0))


def test_decomposition():
    d = ion_device(3)
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    assert d.decompose_operation(cirq.H(q0)) == [
        cirq.rx(np.pi * 1.0).on(cirq.LineQubit(0)),
        cirq.ry(np.pi * -0.5).on(cirq.LineQubit(0))
    ]
    circuit = cirq.Circuit()
    circuit.append([cirq.X(q0), cirq.CNOT(q0, q1)])
    ion_circuit = d.decompose_circuit(circuit)
    d.validate_circuit(ion_circuit)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, ion_circuit, atol=1e-6)


def test_repr():
    d = ion_device(3)

    assert repr(d) == ("IonDevice("
                       "measurement_duration=cirq.Duration(micros=100), "
                       "twoq_gates_duration=cirq.Duration(micros=200), "
                       "oneq_gates_duration=cirq.Duration(micros=10) "
                       "qubits=[cirq.LineQubit(0), cirq.LineQubit(1), "
                       "cirq.LineQubit(2)])")


def test_validate_measurement_non_adjacent_qubits_ok():
    d = ion_device(3)

    d.validate_operation(cirq.GateOperation(
        cirq.MeasurementGate(2), (cirq.LineQubit(0), cirq.LineQubit(1))))


def test_validate_operation_existing_qubits():
    d = ion_device(3)

    d.validate_operation(cirq.GateOperation(
        cirq.XX,
        (cirq.LineQubit(0), cirq.LineQubit(1))))
    d.validate_operation(cirq.Z(cirq.LineQubit(0)))
    d.validate_operation(
        cirq.PhasedXPowGate(phase_exponent=0.75,
                            exponent=0.25,
                            global_shift=0.1).on(cirq.LineQubit(1)))

    with pytest.raises(ValueError):
        d.validate_operation(
            cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(-1)))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.Z(cirq.LineQubit(-1)))
    with pytest.raises(ValueError):
        d.validate_operation(
            cirq.CZ(cirq.LineQubit(1), cirq.LineQubit(1)))
    with pytest.raises(ValueError):
        d.validate_operation(
            cirq.X(cirq.NamedQubit("q1"))
        )


def test_validate_operation_supported_gate():
    d = ion_device(3)

    class MyGate(cirq.Gate):

        def num_qubits(self):
            return 1

    d.validate_operation(cirq.GateOperation(cirq.Z, [cirq.LineQubit(0)]))

    assert MyGate().num_qubits() == 1
    with pytest.raises(ValueError):
        d.validate_operation(cirq.GateOperation(
            MyGate(), [cirq.LineQubit(0)]))
    with pytest.raises(ValueError):
        d.validate_operation(NotImplementedOperation())


def test_can_add_operation_into_moment():
    d = ion_device(3)
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)
    q3 = cirq.LineQubit(3)
    circuit = cirq.Circuit()
    circuit.append(cirq.XX(q0, q1))
    for moment in circuit:
        assert not d.can_add_operation_into_moment(cirq.XX(q2, q0), moment)
        assert not d.can_add_operation_into_moment(cirq.XX(q1, q2), moment)
        assert d.can_add_operation_into_moment(cirq.XX(q2, q3), moment)
        assert d.can_add_operation_into_moment(cirq.Z(q3), moment)


def test_ion_device_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: ion_device(3))
    eq.make_equality_group(
        lambda: ion_device(4))


def test_validate_circuit_repeat_measurement_keys():
    d = ion_device(3)

    circuit = cirq.Circuit()
    circuit.append([cirq.measure(cirq.LineQubit(0), key='a'),
                    cirq.measure(cirq.LineQubit(1), key='a')])

    with pytest.raises(ValueError, match='Measurement key a repeated'):
        d.validate_circuit(circuit)


def test_ion_device_str():
    assert str(ion_device(3)).strip() == """
0───1───2
    """.strip()


def test_at():
    d = ion_device(3)
    assert d.at(-1) is None
    assert d.at(0) == cirq.LineQubit(0)
    assert d.at(2) == cirq.LineQubit(2)


def test_qubit_set():
    assert ion_device(3).qubit_set() == frozenset(cirq.LineQubit.range(3))