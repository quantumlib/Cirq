# Copyright 2019 The Cirq Developers
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

import datetime
import pytest

import matplotlib as mpl
from google.protobuf.text_format import Merge

import cirq
import cirq.google as cg
from cirq.google.api import v2

_CALIBRATION_DATA = Merge(
    """
    timestamp_ms: 1562544000021,
    metrics: [{
        name: 'xeb',
        targets: ['0_0', '0_1'],
        values: [{
            double_val: .9999
        }]
    }, {
        name: 'xeb',
        targets: ['0_0', '1_0'],
        values: [{
            double_val: .9998
        }]
    }, {
        name: 't1',
        targets: ['0_0'],
        values: [{
            double_val: 321
        }]
    }, {
        name: 't1',
        targets: ['0_1'],
        values: [{
            double_val: 911
        }]
    }, {
        name: 't1',
        targets: ['1_0'],
        values: [{
            double_val: 505
        }]
    }, {
        name: 'globalMetric',
        values: [{
            int32_val: 12300
        }]
    }]
""",
    v2.metrics_pb2.MetricsSnapshot(),
)


def test_calibration_metrics_dictionary():
    calibration = cg.Calibration(_CALIBRATION_DATA)

    t1s = calibration['t1']
    assert t1s == {
        (cirq.GridQubit(0, 0),): [321],
        (cirq.GridQubit(0, 1),): [911],
        (cirq.GridQubit(1, 0),): [505],
    }
    assert len(calibration) == 3

    assert 't1' in calibration
    assert 't2' not in calibration

    for qubits, values in t1s.items():
        assert len(qubits) == 1
        assert len(values) == 1

    with pytest.raises(TypeError, match="was 1"):
        _ = calibration[1]
    with pytest.raises(KeyError, match='not-it'):
        _ = calibration['not-it']


def test_calibration_str():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    assert str(calibration) == "Calibration(keys=['globalMetric', 't1', 'xeb'])"


def test_calibration_repr():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    cirq.testing.assert_equivalent_repr(calibration)


def test_calibration_timestamp_str():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    assert calibration.timestamp_str(tz=datetime.timezone.utc) == '2019-07-08 00:00:00.021021+00:00'
    assert (
        calibration.timestamp_str(tz=datetime.timezone(datetime.timedelta(hours=1)))
        == '2019-07-08 01:00:00.021021+01:00'
    )


def test_to_proto():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    assert calibration == cg.Calibration(calibration.to_proto())
    invalid_value = cg.Calibration(metrics={'metric': {(cirq.GridQubit(1, 1),): [1.1, {}]}})
    with pytest.raises(ValueError, match='Unsupported metric value'):
        invalid_value.to_proto()


def test_value_to_float():
    assert cg.Calibration.value_to_float([1.1]) == 1.1
    assert cg.Calibration.value_to_float([0.7, 0.5]) == 0.7
    assert cg.Calibration.value_to_float([7]) == 7

    with pytest.raises(ValueError, match='was empty'):
        cg.Calibration.value_to_float([])
    with pytest.raises(ValueError, match='could not convert string to float'):
        cg.Calibration.value_to_float(['went for a walk'])


def test_calibrations_with_string_key():
    calibration = cg.Calibration(metrics={'metric1': {('alpha',): [0.1]}})
    expected_proto = Merge(
        """
        metrics: [{
          name: 'metric1'
          targets: ['alpha']
          values: [{double_val: 0.1}]
        }]
    """,
        v2.metrics_pb2.MetricsSnapshot(),
    )
    assert expected_proto == calibration.to_proto()
    assert calibration == cg.Calibration(expected_proto)
    assert calibration == cg.Calibration(calibration.to_proto())

    with pytest.raises(ValueError, match='was not a qubit'):
        calibration.key_to_qubit('alpha')


def test_calibration_heatmap():
    calibration = cg.Calibration(_CALIBRATION_DATA)

    heatmap = calibration.heatmap('t1')
    figure = mpl.figure.Figure()
    axes = figure.add_subplot(111)
    heatmap.plot(axes)
