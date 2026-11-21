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

from __future__ import annotations

import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from google.protobuf.text_format import Merge

import cirq
import cirq_google as cg
from cirq_google.api import v2

_CALIBRATION_DATA = Merge(
    """
    timestamp_ms: 1562544000021,
    metrics: [{
        name: 'cz_inferred_gate_error_pauli',
        targets: ['0_0', '0_1'],
        values: [{
            double_val: 0.002
        }]
    }, {
        name: 'cz_inferred_gate_error_pauli',
        targets: ['0_0', '1_0'],
        values: [{
            double_val: 0.0015
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
    }, {
        name: 'zero_error',
        targets: ['0_0'],
        values: [{
            double_val: 0.01
        }]
    }, {
        name: 'zero_error',
        targets: ['0_1'],
        values: [{
            double_val: 0.01
        }]
    }, {
        name: 'zero_error',
        targets: ['1_0'],
        values: [{
            double_val: 0.01
        }]
    }, {
        name: 'one_error',
        targets: ['0_0'],
        values: [{
            double_val: 0.02
        }]
    }, {
        name: 'one_error',
        targets: ['0_1'],
        values: [{
            double_val: 0.02
        }]
    }, {
        name: 'one_error',
        targets: ['1_0'],
        values: [{
            double_val: 0.02
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
    assert len(calibration) == 5

    assert 't1' in calibration
    assert 't2' not in calibration

    for qubits, values in t1s.items():
        assert len(qubits) == 1
        assert len(values) == 1

    with pytest.raises(TypeError, match="was 1"):
        _ = calibration[1]
    with pytest.raises(KeyError, match='not-it'):
        _ = calibration['not-it']


def test_calibration_metric_sets() -> None:
    calibration = cg.Calibration(
        Merge(
            """
            metric_sets: [{
                name: 'sq_rb_pauli_error',
                measurements: [{
                    targets: ['0_0'],
                    value: {double_val: 0.001},
                    uncertainty: {double_val: 0.0001}
                }, {
                    targets: ['0_1'],
                    value: {double_val: 0.002}
                }]
            }, {
                name: 'cz_inferred_gate_error',
                measurements: [{
                    targets: ['0_0', '0_1'],
                    value: {double_val: 0.005},
                    uncertainty: {double_val: 0.0005}
                }]
            }, {
                name: 'globalMetric',
                measurements: [{value: {int32_val: 12300}}]
            }, {
                name: 'noValue',
                measurements: [{targets: ['0_0']}]
            }]
            """,
            v2.metrics_pb2.MetricsSnapshot(),
        )
    )
    q00, q01 = cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)
    assert dict(calibration) == {
        'sq_rb_pauli_error': {(q00,): [0.001], (q01,): [0.002]},
        'sq_rb_pauli_error_ERR': {(q00,): [0.0001]},
        'cz_inferred_gate_error': {(q00, q01): [0.005]},
        'cz_inferred_gate_error_ERR': {(q00, q01): [0.0005]},
        'globalMetric': {(): [12300]},
        'noValue': {(q00,): []},
    }


def test_calibration_metric_sets_equivalent_to_metrics() -> None:
    legacy = Merge(
        """
        timestamp_ms: 1562544000021,
        metrics: [
            {name: 't1', targets: ['0_0'], values: [{double_val: 321}]},
            {name: 't1_ERR', targets: ['0_0'], values: [{double_val: 3}]},
            {name: 't1', targets: ['0_1'], values: [{double_val: 911}]},
            {name: 't1_ERR', targets: ['0_1'], values: [{double_val: 9}]}
        ]
        """,
        v2.metrics_pb2.MetricsSnapshot(),
    )
    condensed = Merge(
        """
        timestamp_ms: 1562544000021,
        metric_sets: [{
            name: 't1',
            measurements: [
                {targets: ['0_0'], value: {double_val: 321}, uncertainty: {double_val: 3}},
                {targets: ['0_1'], value: {double_val: 911}, uncertainty: {double_val: 9}}
            ]
        }]
        """,
        v2.metrics_pb2.MetricsSnapshot(),
    )
    both = v2.metrics_pb2.MetricsSnapshot()
    both.MergeFrom(legacy)
    both.MergeFrom(condensed)

    assert cg.Calibration(legacy) == cg.Calibration(condensed) == cg.Calibration(both)
    assert cg.Calibration(condensed).to_proto() == cg.Calibration(legacy).to_proto()


def test_calibration_metric_sets_take_precedence() -> None:
    calibration = cg.Calibration(
        Merge(
            """
            metrics: [
                {name: 'globalMetric', values: [{int32_val: 1}]},
                {name: 't1', targets: ['0_0'], values: [{double_val: 1}]},
                {name: 't1', targets: ['0_1'], values: [{double_val: 2}]},
                {name: 't1_ERR', targets: ['0_0'], values: [{double_val: 0.1}]},
                {name: 't1_ERR', targets: ['0_1'], values: [{double_val: 0.2}]}
            ]
            metric_sets: [
                {name: 'globalMetric', measurements: [{value: {int32_val: 3}}]},
                {
                    name: 't1',
                    measurements: [
                        {targets: ['0_0'], value: {double_val: 4}, uncertainty: {double_val: 0.4}}
                    ]
                }
            ]
            """,
            v2.metrics_pb2.MetricsSnapshot(),
        )
    )
    assert calibration['globalMetric'] == {(): [3]}
    assert calibration['t1'] == {(cirq.GridQubit(0, 0),): [4], (cirq.GridQubit(0, 1),): [2]}
    assert calibration['t1_ERR'] == {(cirq.GridQubit(0, 0),): [0.4], (cirq.GridQubit(0, 1),): [0.2]}


def test_calibration_metric_sets_multiple_global_metrics() -> None:
    snapshot = Merge(
        """
        metric_sets: [{
            name: 'globalMetric',
            measurements: [{value: {int32_val: 1}}, {value: {int32_val: 2}}]
        }]
        """,
        v2.metrics_pb2.MetricsSnapshot(),
    )
    with pytest.raises(AssertionError, match='Only one metric of a given name'):
        _ = cg.Calibration(snapshot)


def test_calibration_str():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    str_cal = "Calibration(keys=['cz_inferred_gate_error_pauli', 'globalMetric', 'one_error', 't1',"
    str_cal += " 'zero_error'])"
    assert str(calibration) == str_cal


def test_calibration_repr():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    cirq.testing.assert_equivalent_repr(calibration, setup_code="import cirq\nimport cirq_google")


def test_calibration_timestamp_str():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    assert calibration.timestamp_str(tz=datetime.timezone.utc) == '2019-07-08 00:00:00.021000+00:00'
    assert (
        calibration.timestamp_str(tz=datetime.timezone(datetime.timedelta(hours=1)))
        == '2019-07-08 01:00:00.021000+01:00'
    )
    calibration = cg.Calibration(v2.metrics_pb2.MetricsSnapshot(timestamp_ms=1723795877630))
    assert calibration.timestamp_str(tz=datetime.timezone.utc) == '2024-08-16 08:11:17.630000+00:00'


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


def test_key_to_qubits():
    qubits = (cirq.GridQubit(1, 1), cirq.GridQubit(1, 2))
    assert cg.Calibration.key_to_qubit(qubits) == cirq.GridQubit(1, 1)
    assert cg.Calibration.key_to_qubits(qubits) == (cirq.GridQubit(1, 1), cirq.GridQubit(1, 2))

    with pytest.raises(ValueError, match='was not a tuple of qubits'):
        cg.Calibration.key_to_qubit('alpha')
    with pytest.raises(ValueError, match='was not a tuple of grid qubits'):
        cg.Calibration.key_to_qubits('alpha')


def test_calibration_heatmap():
    calibration = cg.Calibration(_CALIBRATION_DATA)

    heatmap = calibration.heatmap('t1')
    figure = mpl.figure.Figure()
    axes = figure.add_subplot(111)
    heatmap.plot(axes)
    assert axes.get_title() == 'T1'

    heatmap = calibration.heatmap('cz_inferred_gate_error_pauli')
    figure = mpl.figure.Figure()
    axes = figure.add_subplot(999)
    heatmap.plot(axes)
    assert axes.get_title() == 'Cz Inferred Gate Error Pauli'

    heatmap = calibration.heatmap('readout')
    figure = mpl.figure.Figure()
    axes = figure.add_subplot(999)
    heatmap.plot(axes)
    assert axes.get_title() == 'Readout'

    with pytest.raises(ValueError, match="one or two qubits.*multi_qubit"):
        multi_qubit_data = Merge(
            """metrics: [{
                name: 'multi_qubit',
                targets: ['0_0', '0_1', '1_0'],
                values: [{double_val: 0.999}]}]""",
            v2.metrics_pb2.MetricsSnapshot(),
        )
        cg.Calibration(multi_qubit_data).heatmap('multi_qubit')
    with pytest.raises(ValueError, match="single metric values.*multi_value"):
        multi_qubit_data = Merge(
            """metrics: [{
                name: 'multi_value',
                targets: ['0_0'],
                values: [{double_val: 0.999}, {double_val: 0.001}]}]""",
            v2.metrics_pb2.MetricsSnapshot(),
        )
        cg.Calibration(multi_qubit_data).heatmap('multi_value')


@pytest.mark.usefixtures('closefigures')
def test_calibration_plot_histograms():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    _, ax = plt.subplots(1, 1)
    calibration.plot_histograms(
        ['t1', 'cz_inferred_gate_error_pauli', 'readout'], ax, labels=['T1', 'XEB', 'readout']
    )
    assert len(ax.get_lines()) == 5

    with pytest.raises(ValueError, match="single metric values.*multi_value"):
        multi_qubit_data = Merge(
            """metrics: [{
                name: 'multi_value',
                targets: ['0_0'],
                values: [{double_val: 0.999}, {double_val: 0.001}]}]""",
            v2.metrics_pb2.MetricsSnapshot(),
        )
        cg.Calibration(multi_qubit_data).plot_histograms('multi_value')


@pytest.mark.usefixtures('closefigures')
def test_calibration_plot():
    calibration = cg.Calibration(_CALIBRATION_DATA)
    _, axs = calibration.plot('cz_inferred_gate_error_pauli')
    assert len(axs[1].get_lines()) == 2
