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

mkey_m = cirq.MeasurementKey('m')
two_qubits = tuple(cirq.LineQubit.range(2))


def test_init_empty():
    cd = cirq.ClassicalData()
    assert cd.measurements is not None
    assert not cd.measurements
    assert cd.keys() is not None
    assert not cd.keys()
    assert cd.measured_qubits is not None
    assert not cd.measured_qubits


def test_init_properties():
    cd = cirq.ClassicalData({mkey_m: (0, 1)}, {mkey_m: two_qubits})
    assert cd.measurements == {mkey_m: (0, 1)}
    assert cd.keys() == (mkey_m,)
    assert cd.measured_qubits == {mkey_m: two_qubits}


def test_record_measurement():
    cd = cirq.ClassicalData()
    cd.record_measurement(mkey_m, (0, 1), two_qubits)
    assert cd.measurements == {mkey_m: (0, 1)}
    assert cd.keys() == (mkey_m,)
    assert cd.measured_qubits == {mkey_m: two_qubits}


def test_record_measurement_errors():
    cd = cirq.ClassicalData()
    with pytest.raises(ValueError, match='3 measurements but 2 qubits'):
        cd.record_measurement(mkey_m, (0, 1, 2), two_qubits)
    cd.record_measurement(mkey_m, (0, 1), two_qubits)
    with pytest.raises(ValueError, match='Measurement already logged to key'):
        cd.record_measurement(mkey_m, (0, 1), two_qubits)


def test_get_int():
    cd = cirq.ClassicalData()
    cd.record_measurement(mkey_m, (0, 1), two_qubits)
    assert cd.get_int(mkey_m) == 1
    cd = cirq.ClassicalData()
    cd.record_measurement(mkey_m, (1, 1), two_qubits)
    assert cd.get_int(mkey_m) == 3
    cd = cirq.ClassicalData()
    cd.record_measurement(mkey_m, (8,), two_qubits)
    assert cd.get_int(mkey_m) == 8
    cd = cirq.ClassicalData()
    cd.record_measurement(mkey_m, (1, 1), (cirq.LineQid.range(2, dimension=3)))
    assert cd.get_int(mkey_m) == 4


def test_copy():
    cd = cirq.ClassicalData({mkey_m: (0, 1)}, {mkey_m: two_qubits})
    cd1 = cd.copy()
    assert cd1 is not cd
    assert cd1 == cd
    assert cd1.measurements is not cd.measurements
    assert cd1.measurements == cd.measurements
    assert cd1.measured_qubits is not cd.measured_qubits
    assert cd1.measured_qubits == cd.measured_qubits


def test_repr():
    cd = cirq.ClassicalData({mkey_m: (0, 1)}, {mkey_m: two_qubits})
    assert repr(cd) == (
        "cirq.ClassicalData("
        "measurements={cirq.MeasurementKey(name='m'): (0, 1)}, "
        "measured_qubits={cirq.MeasurementKey(name='m'): (cirq.LineQubit(0), cirq.LineQubit(1))})"
    )
