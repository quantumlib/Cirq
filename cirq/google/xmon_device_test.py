# Copyright 2018 Google LLC
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

from cirq import ops
from cirq.google import XmonDevice
from cirq.time import Duration


def square_device(width, height, holes=()):
    ns = Duration(nanos=1)
    return XmonDevice(measurement_duration=ns,
                      exp_w_duration=2 * ns,
                      exp_11_duration=3 * ns,
                      qubits=[ops.QubitLoc(x, y)
                              for x in range(width)
                              for y in range(height)
                              if ops.QubitLoc(x, y) not in holes])


def test_init():
    d = square_device(2, 2, holes=[ops.QubitLoc(1, 1)])
    ns = Duration(nanos=1)
    q00 = ops.QubitLoc(0, 0)
    q01 = ops.QubitLoc(0, 1)
    q10 = ops.QubitLoc(1, 0)

    assert d.qubits == {q00, q01, q10}
    assert d.duration_of(ops.Operation(ops.ExpZGate(), (q00,))) == 0 * ns
    assert d.duration_of(ops.Operation(ops.MeasurementGate(),
                                       (q00,))) == ns
    assert d.duration_of(ops.Operation(ops.ExpWGate(), (q00,))) == 2 * ns
    assert d.duration_of(ops.Operation(ops.Exp11Gate(), (q00, q01))) == 3 * ns


def test_validate_operation_adjacent_qubits():
    d = square_device(3, 3)

    d.validate_operation(ops.Operation(
        ops.Exp11Gate(),
        (ops.QubitLoc(0, 0), ops.QubitLoc(1, 0))))

    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(
            ops.Exp11Gate(),
            (ops.QubitLoc(0, 0), ops.QubitLoc(2, 0))))


def test_validate_operation_existing_qubits():
    d = square_device(3, 3, holes=[ops.QubitLoc(1, 1)])

    d.validate_operation(ops.Operation(
        ops.Exp11Gate(),
        (ops.QubitLoc(0, 0), ops.QubitLoc(1, 0))))
    d.validate_operation(ops.Operation(ops.ExpZGate(), (ops.QubitLoc(0, 0),)))

    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(
            ops.Exp11Gate(),
            (ops.QubitLoc(0, 0), ops.QubitLoc(-1, 0))))
    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(ops.ExpZGate(),
                                           (ops.QubitLoc(-1, 0),)))
    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(
            ops.Exp11Gate(),
            (ops.QubitLoc(1, 0), ops.QubitLoc(1, 1))))


def test_validate_operation_supported_gate():
    d = square_device(3, 3)

    class MyGate(ops.Gate):
        pass

    d.validate_operation(ops.Operation(ops.ExpZGate(), [ops.QubitLoc(0, 0)]))
    with pytest.raises(ValueError):
        d.validate_operation(ops.Operation(MyGate, [ops.QubitLoc(0, 0)]))
