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

import cirq
import cirq.neutral_atoms as neutral_atoms


def square_device(
    width: int, height: int, holes=(), max_controls=2, use_timedelta=False
) -> neutral_atoms.NeutralAtomDevice:
    us = cirq.Duration(nanos=10 ** 3) if not use_timedelta else timedelta(microseconds=1)
    ms = cirq.Duration(nanos=10 ** 6) if not use_timedelta else timedelta(microseconds=1000)
    return neutral_atoms.NeutralAtomDevice(  # type: ignore
        measurement_duration=50 * ms,  # type: ignore
        gate_duration=100 * us,  # type: ignore
        control_radius=1.5,
        max_parallel_z=3,
        max_parallel_xy=3,
        max_parallel_c=max_controls,
        qubits=[
            cirq.GridQubit(row, col)
            for col in range(width)
            for row in range(height)
            if cirq.GridQubit(row, col) not in holes
        ],
    )


def test_init():
    d = square_device(2, 2, holes=[cirq.GridQubit(1, 1)])
    us = cirq.Duration(nanos=10 ** 3)
    ms = cirq.Duration(nanos=10 ** 6)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)

    assert d.qubits == {q10, q00, q01}
    assert d.duration_of(cirq.GateOperation(cirq.IdentityGate(1), [q00])) == 100 * us
    assert d.duration_of(cirq.measure(q00)) == 50 * ms
    with pytest.raises(ValueError):
        _ = d.duration_of(cirq.SingleQubitGate().on(q00))


def test_init_timedelta():
    d = square_device(2, 2, holes=[cirq.GridQubit(1, 1)], use_timedelta=True)
    us = cirq.Duration(nanos=10 ** 3)
    ms = cirq.Duration(nanos=10 ** 6)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)

    assert d.qubits == {q10, q00, q01}
    assert d.duration_of(cirq.GateOperation(cirq.IdentityGate(1), [q00])) == 100 * us
    assert d.duration_of(cirq.measure(q00)) == 50 * ms
    with pytest.raises(ValueError):
        _ = d.duration_of(cirq.SingleQubitGate().on(q00))


def test_init_errors():
    line = cirq.LineQubit.range(3)
    us = cirq.Duration(nanos=10 ** 3)
    ms = cirq.Duration(nanos=10 ** 6)
    with pytest.raises(ValueError, match="Unsupported qubit type"):
        _ = neutral_atoms.NeutralAtomDevice(
            measurement_duration=50 * ms,
            gate_duration=100 * us,
            control_radius=1.5,
            max_parallel_z=3,
            max_parallel_xy=3,
            max_parallel_c=3,
            qubits=line,
        )
    with pytest.raises(ValueError, match="max_parallel_c must be less"):
        _ = neutral_atoms.NeutralAtomDevice(
            measurement_duration=50 * ms,
            gate_duration=100 * us,
            control_radius=1.5,
            max_parallel_z=3,
            max_parallel_xy=3,
            max_parallel_c=4,
            qubits=[cirq.GridQubit(0, 0)],
        )


def test_decompose_error():
    d = square_device(2, 2, holes=[cirq.GridQubit(1, 1)])
    for op in d.decompose_operation((cirq.CCZ ** 1.5).on(*(d.qubit_list()))):
        d.validate_operation(op)


def test_validate_gate_errors():
    d = square_device(1, 1)

    d.validate_gate(cirq.IdentityGate(4))
    with pytest.raises(ValueError, match="controlled gates must have integer exponents"):
        d.validate_gate(cirq.CNotPowGate(exponent=0.5))
    with pytest.raises(ValueError, match="Unsupported gate"):
        d.validate_gate(cirq.SingleQubitGate())


def test_validate_operation_errors():
    d = square_device(3, 3)

    class bad_op(cirq.Operation):
        def bad_op(self):
            pass

        def qubits(self):
            pass

        def with_qubits(self, new_qubits):
            pass

    with pytest.raises(ValueError, match="Unsupported operation"):
        d.validate_operation(bad_op())
    not_on_device_op = cirq.parallel_gate_op(
        cirq.X, *[cirq.GridQubit(row, col) for col in range(4) for row in range(4)]
    )
    with pytest.raises(ValueError, match="Qubit not on device"):
        d.validate_operation(not_on_device_op)
    with pytest.raises(ValueError, match="Too many qubits acted on in parallel by"):
        d.validate_operation(cirq.CCX.on(*d.qubit_list()[0:3]))
    with pytest.raises(ValueError, match="are too far away"):
        d.validate_operation(cirq.CZ.on(cirq.GridQubit(0, 0), cirq.GridQubit(2, 2)))
    with pytest.raises(ValueError, match="Too many Z gates in parallel"):
        d.validate_operation(cirq.parallel_gate_op(cirq.Z, *d.qubits))
    with pytest.raises(ValueError, match="Bad number of XY gates in parallel"):
        d.validate_operation(cirq.parallel_gate_op(cirq.X, *d.qubit_list()[1:]))
    with pytest.raises(ValueError, match="ParallelGate over MeasurementGate is not supported"):
        d.validate_operation(
            cirq.ParallelGate(cirq.MeasurementGate(1, key='a'), 4)(*d.qubit_list()[:4])
        )


def test_validate_moment_errors():
    d = square_device(3, 3)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    q12 = cirq.GridQubit(1, 2)
    q02 = cirq.GridQubit(0, 2)
    q04 = cirq.GridQubit(0, 4)
    q03 = cirq.GridQubit(0, 3)
    q20 = cirq.GridQubit(2, 0)
    q21 = cirq.GridQubit(2, 1)

    m = cirq.Moment([cirq.Z.on(q00), (cirq.Z ** 2).on(q01)])
    with pytest.raises(ValueError, match="Non-identical simultaneous "):
        d.validate_moment(m)
    m = cirq.Moment([cirq.X.on(q00), cirq.Y.on(q01)])
    with pytest.raises(ValueError, match="Non-identical simultaneous "):
        d.validate_moment(m)
    m = cirq.Moment([cirq.CNOT.on(q00, q01), cirq.CZ.on(q12, q02)])
    with pytest.raises(ValueError, match="Non-identical simultaneous "):
        d.validate_moment(m)
    m = cirq.Moment([cirq.CNOT.on(q00, q01), cirq.CNOT.on(q12, q02)])
    with pytest.raises(ValueError, match="Too many qubits acted on by controlled gates"):
        d.validate_moment(m)
    m = cirq.Moment([cirq.CNOT.on(q00, q01), cirq.Z.on(q02)])
    with pytest.raises(
        ValueError,
        match="Can't perform non-controlled operations at same time as controlled operations",
    ):
        d.validate_moment(m)
    m = cirq.Moment(cirq.Z.on_each(*d.qubits))
    with pytest.raises(ValueError, match="Too many simultaneous Z gates"):
        d.validate_moment(m)
    m = cirq.Moment(cirq.X.on_each(*(d.qubit_list()[1:])))
    with pytest.raises(ValueError, match="Bad number of simultaneous XY gates"):
        d.validate_moment(m)
    m = cirq.Moment([cirq.MeasurementGate(1, 'a').on(q00), cirq.Z.on(q01)])
    with pytest.raises(
        ValueError, match="Measurements can't be simultaneous with other operations"
    ):
        d.validate_moment(m)
    d.validate_moment(cirq.Moment([cirq.X.on(q00), cirq.Z.on(q01)]))
    us = cirq.Duration(nanos=10 ** 3)
    ms = cirq.Duration(nanos=10 ** 6)
    d2 = neutral_atoms.NeutralAtomDevice(
        measurement_duration=50 * ms,
        gate_duration=100 * us,
        control_radius=1.5,
        max_parallel_z=4,
        max_parallel_xy=4,
        max_parallel_c=4,
        qubits=[cirq.GridQubit(row, col) for col in range(2) for row in range(2)],
    )
    m = cirq.Moment([cirq.CNOT.on(q00, q01), cirq.CNOT.on(q10, q11)])
    with pytest.raises(ValueError, match="Interacting controlled operations"):
        d2.validate_moment(m)
    d2 = neutral_atoms.NeutralAtomDevice(
        measurement_duration=50 * ms,
        gate_duration=100 * us,
        control_radius=1.1,
        max_parallel_z=6,
        max_parallel_xy=6,
        max_parallel_c=6,
        qubits=[cirq.GridQubit(row, col) for col in range(5) for row in range(5)],
    )
    m = cirq.Moment([cirq.CZ.on(q00, q01), cirq.CZ.on(q03, q04), cirq.CZ.on(q20, q21)])
    d2.validate_moment(m)
    m = cirq.Moment([cirq.CZ.on(q00, q01), cirq.CZ.on(q02, q03), cirq.CZ.on(q10, q11)])
    with pytest.raises(ValueError, match="Interacting controlled operations"):
        d2.validate_moment(m)


def test_can_add_operation_into_moment_coverage():
    d = square_device(2, 2)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    m = cirq.Moment([cirq.X.on(q00)])
    assert not d.can_add_operation_into_moment(cirq.X.on(q00), m)
    assert not d.can_add_operation_into_moment(cirq.CZ.on(q01, q10), m)
    assert d.can_add_operation_into_moment(cirq.Z.on(q01), m)


def test_validate_circuit_errors():
    d = square_device(2, 2, max_controls=3)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    c = cirq.Circuit()
    c.append(cirq.parallel_gate_op(cirq.X, *d.qubits))
    c.append(cirq.CCZ.on(q00, q01, q10))
    c.append(cirq.parallel_gate_op(cirq.Z, q00, q01, q10))
    m = cirq.Moment(cirq.X.on_each(q00, q01) + cirq.Z.on_each(q10, q11))
    c.append(m)
    c.append(cirq.measure_each(*d.qubits))
    d.validate_circuit(c)
    c.append(cirq.Moment([cirq.X.on(q00)]))
    with pytest.raises(ValueError, match="Non-empty moment after measurement"):
        d.validate_circuit(c)


def test_repr():
    d = square_device(1, 1)
    cirq.testing.assert_equivalent_repr(d)


def test_str():
    assert (
        str(square_device(2, 2)).strip()
        == """
(0, 0)───(0, 1)
│        │
│        │
(1, 0)───(1, 1)
    """.strip()
    )


def test_qubit_set():
    assert square_device(2, 2).qubit_set() == frozenset(cirq.GridQubit.square(2, 0, 0))
