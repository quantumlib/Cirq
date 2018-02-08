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
from cirq.circuits import Circuit
from cirq.google import xmon_schedule_greedy, XmonDevice
from cirq.ops import Operation
from cirq.schedules import Schedule, ScheduledOperation, Timestamp
from cirq.time import Duration


def square_device(width, height, holes=()):
    return


def test_xmon_schedule_greedy():
    ns = Duration(nanos=1)
    q00 = ops.QubitLoc(0, 0)
    q01 = ops.QubitLoc(0, 1)
    q10 = ops.QubitLoc(1, 0)
    q11 = ops.QubitLoc(1, 1)
    d = XmonDevice(measurement_duration=ns,
                   exp_w_duration=20 * ns,
                   exp_11_duration=50 * ns,
                   qubits=[q00, q01, q10, q11])

    c = Circuit()
    c.append([
        ops.Exp11Gate().on(q01, q00),
        ops.Exp11Gate().on(q00, q10),
        ops.Exp11Gate().on(q10, q11),
        ops.Exp11Gate().on(q11, q01),
        ops.ExpWGate().on(q00),
        ops.ExpWGate().on(q11),
        ops.Exp11Gate().on(q01, q00),
        ops.Exp11Gate().on(q00, q10),
        ops.ExpWGate().on(q11),
        ops.Exp11Gate().on(q01, q00),
        ops.Exp11Gate().on(q00, q10),
        ops.ExpWGate().on(q11),
    ])

    s = xmon_schedule_greedy(d, c)
    assert str(s) == """CZ(0_0, 1_0) during [t=0, t=50000)
CZ(1_0, 1_1) during [t=50000, t=100000)
CZ(0_1, 0_0) during [t=100000, t=150000)
CZ(1_1, 0_1) during [t=150000, t=200000)
X(0_0) during [t=200000, t=220000)
X(1_1) during [t=200000, t=220000)
CZ(0_0, 1_0) during [t=220000, t=270000)
CZ(0_0, 1_0) during [t=270000, t=320000)
X(1_1) during [t=320000, t=340000)
X(1_1) during [t=340000, t=360000)"""
