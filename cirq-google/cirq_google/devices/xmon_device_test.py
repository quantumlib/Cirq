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

import pytest

import cirq_google as cg
import cirq
import cirq.testing


def square_device(width: int, height: int, holes=()) -> cg.XmonDevice:
    ns = cirq.Duration(nanos=1)
    return cg.XmonDevice(
        measurement_duration=ns,
        exp_w_duration=2 * ns,
        exp_11_duration=3 * ns,
        qubits=[
            cirq.GridQubit(row, col)
            for col in range(width)
            for row in range(height)
            if cirq.GridQubit(col, row) not in holes
        ],
    )


class NotImplementedOperation(cirq.Operation):
    def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
        raise NotImplementedError()

    @property
    def qubits(self):
        raise NotImplementedError()


def test_init():
    d = square_device(2, 2, holes=[cirq.GridQubit(1, 1)])
    ns = cirq.Duration(nanos=1)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)

    assert d.qubits == {q00, q01, q10}
    assert d.duration_of(cirq.Z(q00)) == 0 * ns
    assert d.duration_of(cirq.measure(q00)) == ns
    assert d.duration_of(cirq.measure(q00, q01)) == ns
    assert d.duration_of(cirq.X(q00)) == 2 * ns
    assert d.duration_of(cirq.CZ(q00, q01)) == 3 * ns
    with pytest.raises(ValueError):
        _ = d.duration_of(cirq.SingleQubitGate().on(q00))


def test_init_timedelta():
    from datetime import timedelta

    timedelta_duration = timedelta(microseconds=1)
    d = cg.XmonDevice(
        measurement_duration=timedelta_duration,
        exp_w_duration=2 * timedelta_duration,
        exp_11_duration=3 * timedelta_duration,
        qubits=[cirq.GridQubit(row, col) for col in range(2) for row in range(2)],
    )
    microsecond = cirq.Duration(nanos=1000)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)

    assert d.qubits == {q00, q01, q10, q11}
    assert d.duration_of(cirq.Z(q00)) == 0 * microsecond
    assert d.duration_of(cirq.measure(q00)) == microsecond
    assert d.duration_of(cirq.measure(q00, q01)) == microsecond
    assert d.duration_of(cirq.X(q00)) == 2 * microsecond
    assert d.duration_of(cirq.CZ(q00, q01)) == 3 * microsecond


def test_repr():
    d = square_device(2, 2, holes=[])

    assert repr(d) == (
        "XmonDevice("
        "measurement_duration=cirq.Duration(nanos=1), "
        "exp_w_duration=cirq.Duration(nanos=2), "
        "exp_11_duration=cirq.Duration(nanos=3) "
        "qubits=[cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), "
        "cirq.GridQubit(1, 0), "
        "cirq.GridQubit(1, 1)])"
    )


def test_can_add_operation_into_moment():
    d = square_device(2, 2)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    m = cirq.Moment([cirq.CZ(q00, q01)])
    assert not d.can_add_operation_into_moment(cirq.CZ(q10, q11), m)


def test_validate_moment():
    d = square_device(2, 2)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    m = cirq.Moment([cirq.CZ(q00, q01), cirq.CZ(q10, q11)])
    with pytest.raises(ValueError):
        d.validate_moment(m)


def test_validate_operation_adjacent_qubits():
    d = square_device(3, 3)

    d.validate_operation(cirq.GateOperation(cirq.CZ, (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))))

    with pytest.raises(ValueError, match='Non-local interaction'):
        d.validate_operation(
            cirq.GateOperation(cirq.CZ, (cirq.GridQubit(0, 0), cirq.GridQubit(2, 0)))
        )


def test_validate_measurement_non_adjacent_qubits_ok():
    d = square_device(3, 3)

    d.validate_operation(
        cirq.GateOperation(
            cirq.MeasurementGate(2, 'a'), (cirq.GridQubit(0, 0), cirq.GridQubit(2, 0))
        )
    )


def test_validate_operation_existing_qubits():
    d = square_device(3, 3, holes=[cirq.GridQubit(1, 1)])

    d.validate_operation(cirq.GateOperation(cirq.CZ, (cirq.GridQubit(0, 0), cirq.GridQubit(1, 0))))
    d.validate_operation(cirq.Z(cirq.GridQubit(0, 0)))

    with pytest.raises(ValueError):
        d.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(-1, 0)))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.Z(cirq.GridQubit(-1, 0)))
    with pytest.raises(ValueError):
        d.validate_operation(cirq.CZ(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)))


class MyGate(cirq.Gate):
    def num_qubits(self):
        return 1


q = cirq.GridQubit.rect(1, 3)
matrix_gate = cirq.MatrixGate(cirq.testing.random_unitary(2))


@pytest.mark.parametrize(
    'op,is_valid',
    [
        (cirq.Z(cirq.GridQubit(0, 0)), True),
        (cirq.Z(cirq.GridQubit(0, 0)).with_tags('test_tag'), True),
        (
            cirq.Z(cirq.GridQubit(0, 0)).with_tags('test_tag').controlled_by(cirq.GridQubit(0, 1)),
            True,
        ),
        (
            cirq.Z(cirq.GridQubit(0, 0)).controlled_by(cirq.GridQubit(0, 1)).with_tags('test_tag'),
            True,
        ),
        (NotImplementedOperation(), False),
        (MyGate()(cirq.GridQubit(0, 0)), False),
    ],
)
def test_validate_operation_supported_gate(op, is_valid):
    d = square_device(3, 3)
    if is_valid:
        d.validate_operation(op)
    else:
        with pytest.raises(ValueError):
            d.validate_operation(op)


def test_validate_circuit_repeat_measurement_keys():
    d = square_device(3, 3)

    circuit = cirq.Circuit()
    circuit.append(
        [cirq.measure(cirq.GridQubit(0, 0), key='a'), cirq.measure(cirq.GridQubit(0, 1), key='a')]
    )

    with pytest.raises(ValueError, match='Measurement key a repeated'):
        d.validate_circuit(circuit)


def test_xmon_device_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: square_device(3, 3))
    eq.make_equality_group(lambda: square_device(3, 3, holes=[cirq.GridQubit(1, 1)]))
    eq.make_equality_group(
        lambda: cg.XmonDevice(
            cirq.Duration(nanos=1), cirq.Duration(nanos=2), cirq.Duration(nanos=3), []
        )
    )
    eq.make_equality_group(
        lambda: cg.XmonDevice(
            cirq.Duration(nanos=1), cirq.Duration(nanos=1), cirq.Duration(nanos=1), []
        )
    )


def test_xmon_device_str():
    assert (
        str(square_device(2, 2)).strip()
        == """
(0, 0)───(0, 1)
│        │
│        │
(1, 0)───(1, 1)
    """.strip()
    )


def test_xmon_device_repr_pretty():
    cirq.testing.assert_repr_pretty(
        square_device(2, 2),
        """
(0, 0)───(0, 1)
│        │
│        │
(1, 0)───(1, 1)
    """.strip(),
    )

    cirq.testing.assert_repr_pretty(square_device(2, 2), "cirq_google.XmonDevice(...)", cycle=True)


def test_at():
    d = square_device(3, 3)
    assert d.at(-1, -1) is None
    assert d.at(0, 0) == cirq.GridQubit(0, 0)

    assert d.at(-1, 1) is None
    assert d.at(0, 1) == cirq.GridQubit(0, 1)
    assert d.at(1, 1) == cirq.GridQubit(1, 1)
    assert d.at(2, 1) == cirq.GridQubit(2, 1)
    assert d.at(3, 1) is None

    assert d.at(1, -1) is None
    assert d.at(1, 0) == cirq.GridQubit(1, 0)
    assert d.at(1, 1) == cirq.GridQubit(1, 1)
    assert d.at(1, 2) == cirq.GridQubit(1, 2)
    assert d.at(1, 3) is None


def test_row_and_col():
    d = square_device(2, 3)
    assert d.col(-1) == []
    assert d.col(0) == [cirq.GridQubit(0, 0), cirq.GridQubit(1, 0), cirq.GridQubit(2, 0)]
    assert d.col(1) == [cirq.GridQubit(0, 1), cirq.GridQubit(1, 1), cirq.GridQubit(2, 1)]
    assert d.col(2) == []
    assert d.col(5000) == []

    assert d.row(-1) == []
    assert d.row(0) == [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)]
    assert d.row(1) == [cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)]
    assert d.row(2) == [cirq.GridQubit(2, 0), cirq.GridQubit(2, 1)]
    assert d.row(3) == []

    b = cg.Bristlecone
    assert b.col(0) == [cirq.GridQubit(5, 0)]
    assert b.row(0) == [cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)]
    assert b.col(1) == [cirq.GridQubit(4, 1), cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)]


def test_qubit_set():
    assert cg.Foxtail.qubit_set() == frozenset(cirq.GridQubit.rect(2, 11, 0, 0))
