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

import pytest

import cirq
import cirq_aqt
import cirq_aqt.aqt_device as cad
import sympy.utilities.matchpy_connector


def aqt_device(chain_length: int, use_timedelta=False) -> cad.AQTDevice:
    ms = 1000 * cirq.Duration(nanos=1) if not use_timedelta else timedelta(microseconds=1)
    return cad.AQTDevice(  # type: ignore
        measurement_duration=100 * ms,  # type: ignore
        twoq_gates_duration=200 * ms,  # type: ignore
        oneq_gates_duration=10 * ms,  # type: ignore
        qubits=cirq.LineQubit.range(chain_length),
    )


class NotImplementedOperation(cirq.Operation):
    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        raise NotImplementedError()


def test_init():
    d = aqt_device(3)
    ms = 1000 * cirq.Duration(nanos=1)
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)

    assert d.qubits == {q0, q1, q2}
    assert d.duration_of(cirq.Z(q0)) == 10 * ms
    assert d.duration_of(cirq.measure(q0)) == 100 * ms
    assert d.duration_of(cirq.measure(q0, q1)) == 100 * ms
    assert d.duration_of(cirq.XX(q0, q1)) == 200 * ms
    with pytest.raises(ValueError, match="Unsupported gate type"):
        _ = d.duration_of(cirq.I(q0))

    with pytest.raises(TypeError, match="NamedQubit"):
        _ = cad.AQTDevice(
            measurement_duration=ms,
            twoq_gates_duration=ms,
            oneq_gates_duration=ms,
            qubits=[cirq.LineQubit(0), cirq.NamedQubit("a")],
        )


def test_metadata():
    d = aqt_device(3)
    assert d.metadata.qubit_set == frozenset(
        {cirq.LineQubit(0), cirq.LineQubit(1), cirq.LineQubit(2)}
    )
    assert len(d.metadata.nx_graph.edges()) == 3


def test_init_timedelta():
    d = aqt_device(3, use_timedelta=True)
    ms = 1000 * cirq.Duration(nanos=1)
    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    q2 = cirq.LineQubit(2)

    assert d.qubits == {q0, q1, q2}
    assert d.duration_of(cirq.Z(q0)) == 10 * ms
    assert d.duration_of(cirq.measure(q0)) == 100 * ms
    assert d.duration_of(cirq.measure(q0, q1)) == 100 * ms
    assert d.duration_of(cirq.XX(q0, q1)) == 200 * ms
    with pytest.raises(ValueError):
        _ = d.duration_of(cirq.testing.SingleQubitGate().on(q0))


def test_repr():
    d = aqt_device(3)
    assert repr(d) == (
        "cirq_aqt.aqt_device.AQTDevice("
        "measurement_duration=cirq.Duration(micros=100), "
        "twoq_gates_duration=cirq.Duration(micros=200), "
        "oneq_gates_duration=cirq.Duration(micros=10), "
        "qubits=[cirq.LineQubit(0), cirq.LineQubit(1), "
        "cirq.LineQubit(2)])"
    )
    device = cirq_aqt.aqt_device.get_aqt_device(5)
    cirq.testing.assert_equivalent_repr(device, setup_code='import cirq\nimport cirq_aqt\n')


def test_validate_measurement_non_adjacent_qubits_ok():
    d = aqt_device(3)

    d.validate_operation(
        cirq.GateOperation(cirq.MeasurementGate(2, 'key'), (cirq.LineQubit(0), cirq.LineQubit(1)))
    )


def test_validate_operation_existing_qubits():
    d = aqt_device(3)

    d.validate_operation(cirq.GateOperation(cirq.XX, (cirq.LineQubit(0), cirq.LineQubit(1))))
    d.validate_operation(cirq.Z(cirq.LineQubit(0)))
    d.validate_operation(
        cirq.PhasedXPowGate(phase_exponent=0.75, exponent=0.25, global_shift=0.1).on(
            cirq.LineQubit(1)
        )
    )

    with pytest.raises(ValueError):
        d.validate_operation(cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(-1)))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.Z(cirq.LineQubit(-1)))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.CZ(cirq.LineQubit(1), cirq.LineQubit(1)))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.X(cirq.NamedQubit("q1")))


def test_validate_operation_supported_gate():
    d = aqt_device(3)

    class MyGate(cirq.Gate):
        def num_qubits(self):
            return 1

    d.validate_operation(cirq.GateOperation(cirq.Z, [cirq.LineQubit(0)]))

    assert MyGate().num_qubits() == 1
    with pytest.raises(ValueError):
        d.validate_operation(cirq.GateOperation(MyGate(), [cirq.LineQubit(0)]))
    with pytest.raises(ValueError):
        d.validate_operation(NotImplementedOperation())


def test_aqt_device_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: aqt_device(3))
    eq.make_equality_group(lambda: aqt_device(4))


def test_validate_circuit_repeat_measurement_keys():
    d = aqt_device(3)

    circuit = cirq.Circuit()
    circuit.append(
        [cirq.measure(cirq.LineQubit(0), key='a'), cirq.measure(cirq.LineQubit(1), key='a')]
    )

    with pytest.raises(ValueError, match='Measurement key a repeated'):
        d.validate_circuit(circuit)


def test_aqt_device_str():
    assert str(aqt_device(3)) == "q(0)───q(1)───q(2)"


def test_aqt_device_pretty_repr():
    cirq.testing.assert_repr_pretty(aqt_device(3), "q(0)───q(1)───q(2)")
    cirq.testing.assert_repr_pretty(aqt_device(3), "AQTDevice(...)", cycle=True)


def test_at():
    d = aqt_device(3)
    assert d.at(-1) is None
    assert d.at(0) == cirq.LineQubit(0)
    assert d.at(2) == cirq.LineQubit(2)


def test_decompose_parameterized_gates():
    theta = sympy.Symbol("theta")
    q = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q[0]) ** theta, cirq.XX(*q) ** theta)
    d = aqt_device(3)
    assert d.gateset.validate(d.decompose_circuit(circuit))
    assert d.gateset._decompose_two_qubit_operation(cirq.CZ(*q) ** theta, 0) is NotImplemented
