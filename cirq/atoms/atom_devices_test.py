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

import cirq
import cirq.atoms as atoms


def square_device(width: int, height: int, holes=(),
                  max_controls=2) -> atoms.AQuA:
    us = cirq.Duration(nanos=10**3)
    ms = cirq.Duration(nanos=10**6)
    return atoms.AQuA(measurement_duration=50 * ms,
                      gate_duration=100 * us,
                      control_radius=1.5,
                      max_parallel_z=3,
                      max_parallel_xy=3,
                      max_parallel_c=max_controls,
                      qubits=[cirq.GridQubit(row, col)
                              for col in range(width)
                              for row in range(height)
                              if cirq.GridQubit(row, col) not in holes])


def test_init():
    d = square_device(2, 2, holes=[cirq.GridQubit(1, 1)])
    us = cirq.Duration(nanos=10 ** 3)
    ms = cirq.Duration(nanos=10 ** 6)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)

    assert d.qubits == {q10, q00, q01}
    assert d.duration_of(cirq.GateOperation(
        cirq.IdentityGate(1), [q00])) == 100 * us
    assert d.duration_of(cirq.measure(q00)) == 50 * ms
    with pytest.raises(ValueError):
        _ = d.duration_of(cirq.SingleQubitGate().on(q00))


def test_init_errors():
    line = cirq.LineQubit.range(3)
    us = cirq.Duration(nanos=10 ** 3)
    ms = cirq.Duration(nanos=10 ** 6)
    with pytest.raises(ValueError) as bad_qubit_type:
        _ = atoms.AQuA(measurement_duration=50 * ms,
                       gate_duration=100 * us,
                       control_radius=1.5,
                       max_parallel_z=3,
                       max_parallel_xy=3,
                       max_parallel_c=3,
                       qubits= line)
    assert "Unsupported qubit type" in str(bad_qubit_type.value)
    with pytest.raises(ValueError) as bad_parallel_parameters:
        _ = atoms.AQuA(measurement_duration=50 * ms,
                       gate_duration=100 * us,
                       control_radius=1.5,
                       max_parallel_z=3,
                       max_parallel_xy=3,
                       max_parallel_c=4,
                       qubits= [cirq.GridQubit(0,0)])
    assert "max_parallel_c must be less" in str(bad_parallel_parameters)


def test_validate_gate_errors():
    d = square_device(1,1)

    d.validate_gate(cirq.IdentityGate(4))
    with pytest.raises(ValueError) as bad_exp:
        d.validate_gate(cirq.CNotPowGate(exponent=0.5))
    assert 'controlled gates must have integer exponents' == str(bad_exp.value)
    with pytest.raises(ValueError) as bad_gate:
        d.validate_gate(cirq.SingleQubitGate())
    assert 'Unsupported gate' in str(bad_gate.value)


def test_validate_operation_errors():
    d = square_device(3, 3)

    class bad_op(cirq.Operation):

        def bad_op(self):
            pass

        def qubits(self):
            pass

        def with_qubits(self, new_qubits):
            pass

    with pytest.raises(ValueError) as bad_operation:
        d.validate_operation(bad_op())
    assert "Unsupported operation" in str(bad_operation.value)
    not_on_device_op = cirq.ParallelGateOperation(cirq.X,
                                                  [cirq.GridQubit(row, col)
                                                   for col in range(4)
                                                   for row in range(4)])
    with pytest.raises(ValueError) as bad_qubits:
        d.validate_operation(not_on_device_op)
    assert "Qubit not on device" in str(bad_qubits.value)
    with pytest.raises(ValueError) as control_limit:
        d.validate_operation(cirq.CCX.on(*d.qubit_list()[0:3]))
    assert "Too many qubits acted on in parallel by" in str(control_limit.value)
    with pytest.raises(ValueError) as too_far:
        d.validate_operation(cirq.CZ.on(cirq.GridQubit(0, 0),
                                        cirq.GridQubit(2, 2)))
    assert "are too far away" in str(too_far.value)
    with pytest.raises(ValueError) as too_many_z:
        d.validate_operation(cirq.ParallelGateOperation(cirq.Z, d.qubits))
    assert "Too many Z gates in parallel" == str(too_many_z.value)
    with pytest.raises(ValueError) as bad_xy_num:
        d.validate_operation(cirq.ParallelGateOperation(cirq.X,
                                                        d.qubit_list()[1:]))
    assert "Bad number of XY gates in parallel" == str(bad_xy_num.value)


def test_validate_moment_errors():
    d = square_device(3, 3)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q12 = cirq.GridQubit(1, 2)
    q02 = cirq.GridQubit(0, 2)

    m = cirq.Moment([cirq.Z.on(q00), (cirq.Z**2).on(q01)])
    with pytest.raises(ValueError) as non_id_z:
        d.validate_moment(m)
    assert "Non-identical Parallel Z gates" == str(non_id_z.value)
    m = cirq.Moment([cirq.X.on(q00), cirq.Y.on(q01)])
    with pytest.raises(ValueError) as non_id_xy:
        d.validate_moment(m)
    assert "Non-identical Parallel XY gates" == str(non_id_xy.value)
    m = cirq.Moment([cirq.CNOT.on(q00, q01), cirq.CZ.on(q12, q02)])
    with pytest.raises(ValueError) as non_id_c:
        d.validate_moment(m)
    assert "Non-identical Parallel Controlled Gates" == str(non_id_c.value)
    m = cirq.Moment([cirq.CNOT.on(q00, q01), cirq.CNOT.on(q12, q02)])
    with pytest.raises(ValueError) as too_many_c:
        d.validate_moment(m)
    assert "Too many qubits acted on by controlled gates" == str(
        too_many_c.value)
    m = cirq.Moment([cirq.CNOT.on(q00, q01), cirq.Z.on(q02)])
    with pytest.raises(ValueError) as c_blocks:
        d.validate_moment(m)
    assert ("Can't perform non-controlled operations"
            " at same time as controlled operations" == str(c_blocks.value))
    m = cirq.Moment(cirq.Z.on_each(*d.qubits))
    with pytest.raises(ValueError) as too_many_z:
        d.validate_moment(m)
    assert "Too many simultaneous Z gates" == str(too_many_z.value)
    m = cirq.Moment(cirq.X.on_each(*(d.qubit_list()[1:])))
    with pytest.raises(ValueError) as bad_xy:
        d.validate_moment(m)
    assert "Bad number of simultaneous XY gates" == str(bad_xy.value)
    m = cirq.Moment([cirq.MeasurementGate(1).on(q00), cirq.Z.on(q01)])
    with pytest.raises(ValueError) as measurement:
        d.validate_moment(m)
    assert ("Measurements can't be simultaneous with other operations" ==
            str(measurement.value))
    d.validate_moment(cirq.Moment([cirq.X.on(q00), cirq.Z.on(q01)]))


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
    c.append(cirq.ParallelGateOperation(cirq.X, d.qubits))
    c.append(cirq.CCZ.on(q00, q01, q10))
    c.append(cirq.ParallelGateOperation(cirq.Z, [q00, q01, q10]))
    m = cirq.Moment(cirq.X.on_each(q00, q01) + cirq.Z.on_each(q10, q11))
    c.append(m)
    c.append(cirq.measure_each(*d.qubits))
    d.validate_circuit(c)
    c.append(cirq.Moment([cirq.X.on(q00)]))
    with pytest.raises(ValueError) as non_terminal_measurement:
        d.validate_circuit(c)
    assert ("Non-empty moment after measurement" ==
            str(non_terminal_measurement.value))


def test_validate_scheduled_operation_errors():
    d = square_device(2, 2)
    s = cirq.Schedule(device=cirq.UnconstrainedDevice)
    q00 = cirq.GridQubit(0, 0)
    so = cirq.ScheduledOperation(cirq.Timestamp(), cirq.Duration(nanos=1),
                                 cirq.X.on(q00))
    with pytest.raises(ValueError) as too_short:
        d.validate_scheduled_operation(s, so)
    assert "Incompatible operation duration" == str(too_short.value)


def test_validate_schedule_errors():
    d = square_device(2, 2, max_controls=3)
    s = cirq.Schedule(device=cirq.UnconstrainedDevice)
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q10 = cirq.GridQubit(1, 0)
    q11 = cirq.GridQubit(1, 1)
    us = cirq.Duration(nanos=10**3)
    ms = cirq.Duration(nanos=10**6)
    msone = cirq.Timestamp(nanos=10**6)
    mstwo = cirq.Timestamp(nanos=2*10**6)
    msthree = cirq.Timestamp(nanos=3*10**6)
    for qubit in d.qubits:
        s.include(cirq.ScheduledOperation(cirq.Timestamp(nanos=0), 100*us,
                                          cirq.X.on(qubit)))
    s.include(cirq.ScheduledOperation(msone, 100*us,
                                      cirq.TOFFOLI.on(q00,q01,q10)))
    s.include(cirq.ScheduledOperation(mstwo, 100*us, cirq.ParallelGateOperation(
        cirq.X, [q00, q01])))
    s.include(cirq.ScheduledOperation(mstwo, 100*us, cirq.ParallelGateOperation(
        cirq.Z, [q10, q11])))
    for qubit in d.qubits:
        s.include(cirq.ScheduledOperation(msthree,
                                          50*ms,
                                          cirq.GateOperation(
                                              cirq.MeasurementGate(1, qubit),
                                              [qubit])))
    d.validate_schedule(s)
    s.include(cirq.ScheduledOperation(cirq.Timestamp(nanos=10**9), 100*us,
                                      cirq.X.on(q00)))
    with pytest.raises(ValueError) as terminal:
        d.validate_schedule(s)
    assert "Non-measurement operation after measurement" == str(terminal.value)

def test_repr():
    d = square_device(1, 1)
    assert repr(d) == ("AQuA("
                       "measurement_duration=cirq.Duration(picos=50000000000), "
                       "gate_duration=cirq.Duration(picos=100000000), "
                       "max_parallel_z=3, "
                       "max_parallel_xy=3, "
                       "max_parallel_c=2, "
                       "control_radius=1.5, "
                       "qubits=[cirq.GridQubit(0, 0)])")


def test_str():
    assert str(square_device(2, 2)).strip() == """
(0, 0)───(0, 1)
│        │
│        │
(1, 0)───(1, 1)
    """.strip()

















