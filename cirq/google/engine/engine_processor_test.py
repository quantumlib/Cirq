# Copyright 2020 The Cirq Developers
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

from unittest import mock
import datetime
import pytest
from google.protobuf.text_format import Merge
import cirq.google as cg
from cirq.google.api import v2
from cirq.google.engine.engine import EngineContext
from cirq.google.engine.client.quantum_v1alpha1 import types as qtypes


def _to_any(proto):
    any_proto = qtypes.any_pb2.Any()
    any_proto.Pack(proto)
    return any_proto


def _to_timestamp(json_string):
    timestamp_proto = qtypes.timestamp_pb2.Timestamp()
    timestamp_proto.FromJsonString(json_string)
    return timestamp_proto


_CALIBRATION = qtypes.QuantumCalibration(
    name='projects/a/processors/p/calibrations/1562715599',
    timestamp=_to_timestamp('2019-07-09T23:39:59Z'),
    data=_to_any(
        Merge(
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
        targets: ['q0_0'],
        values: [{
            double_val: 321
        }]
    }, {
        name: 't1',
        targets: ['q0_1'],
        values: [{
            double_val: 911
        }]
    }, {
        name: 't1',
        targets: ['q1_0'],
        values: [{
            double_val: 505
        }]
    }, {
        name: 'globalMetric',
        values: [{
            int32_val: 12300
        }]
    }]
""", v2.metrics_pb2.MetricsSnapshot())))

_DEVICE_SPEC = _to_any(
    Merge(
        """
valid_gate_sets: [{
    name: 'test_set',
    valid_gates: [{
        id: 'x',
        number_of_qubits: 1,
        gate_duration_picos: 1000,
        valid_targets: ['1q_targets']
    }]
}],
valid_qubits: ['0_0', '1_1'],
valid_targets: [{
    name: '1q_targets',
    target_ordering: SYMMETRIC,
    targets: [{
        ids: ['0_0']
    }]
}]
""", v2.device_pb2.DeviceSpecification()))


@pytest.fixture(scope='session', autouse=True)
def mock_grpc_client():
    with mock.patch('cirq.google.engine.engine_client'
                    '.quantum.QuantumEngineServiceClient') as _fixture:
        yield _fixture


def test_engine():
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert processor.engine().project_id == 'a'


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_processor')
def test_health(get_processor):
    get_processor.return_value = qtypes.QuantumProcessor(
        health=qtypes.QuantumProcessor.Health.OK)
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=qtypes.QuantumProcessor(
            health=qtypes.QuantumProcessor.Health.DOWN))
    assert processor.health() == 'OK'


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_processor')
def test_expected_down_time(get_processor):
    processor = cg.EngineProcessor('a',
                                   'p',
                                   EngineContext(),
                                   _processor=qtypes.QuantumProcessor())
    assert not processor.expected_down_time()

    get_processor.return_value = qtypes.QuantumProcessor(
        expected_down_time=qtypes.timestamp_pb2.Timestamp(seconds=1581515101))

    assert cg.EngineProcessor(
        'a', 'p', EngineContext()).expected_down_time() == datetime.datetime(
            2020, 2, 12, 13, 45, 1)
    get_processor.assert_called_once()


def test_expected_recovery_time():
    processor = cg.EngineProcessor('a',
                                   'p',
                                   EngineContext(),
                                   _processor=qtypes.QuantumProcessor())
    assert not processor.expected_recovery_time()
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=qtypes.QuantumProcessor(
            expected_recovery_time=qtypes.timestamp_pb2.Timestamp(
                seconds=1581515101)))
    assert processor.expected_recovery_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1)


def test_supported_languages():
    processor = cg.EngineProcessor('a',
                                   'p',
                                   EngineContext(),
                                   _processor=qtypes.QuantumProcessor())
    assert processor.supported_languages() == []
    processor = cg.EngineProcessor('a',
                                   'p',
                                   EngineContext(),
                                   _processor=qtypes.QuantumProcessor(
                                       supported_languages=['lang1', 'lang2']))
    assert processor.supported_languages() == ['lang1', 'lang2']


def test_get_device_specification():
    processor = cg.EngineProcessor('a',
                                   'p',
                                   EngineContext(),
                                   _processor=qtypes.QuantumProcessor())
    assert not processor.get_device_specification()

    # Construct expected device proto based on example
    expected = v2.device_pb2.DeviceSpecification()
    gs = expected.valid_gate_sets.add()
    gs.name = 'test_set'
    gates = gs.valid_gates.add()
    gates.id = 'x'
    gates.number_of_qubits = 1
    gates.gate_duration_picos = 1000
    gates.valid_targets.extend(['1q_targets'])
    expected.valid_qubits.extend(['0_0', '1_1'])
    target = expected.valid_targets.add()
    target.name = '1q_targets'
    target.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = target.targets.add()
    new_target.ids.extend(['0_0'])

    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=qtypes.QuantumProcessor(device_spec=_DEVICE_SPEC))
    assert processor.get_device_specification() == expected


@mock.patch('cirq.google.engine.engine_client.EngineClient.list_calibrations')
def test_list_calibrations(list_calibrations):
    list_calibrations.return_value = [_CALIBRATION]
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert [c.timestamp for c in processor.list_calibrations()
           ] == [1562544000021]
    list_calibrations.assert_called_with('a', 'p', '')
    assert [c.timestamp for c in processor.list_calibrations(1562500000000)
           ] == [1562544000021]
    list_calibrations.assert_called_with('a', 'p', 'timestamp >= 1562500000000')
    assert [
        c.timestamp for c in processor.list_calibrations(
            latest_timestamp_seconds=1562600000000)
    ] == [1562544000021]
    list_calibrations.assert_called_with('a', 'p', 'timestamp <= 1562600000000')
    assert [
        c.timestamp
        for c in processor.list_calibrations(1562500000000, 1562600000000)
    ] == [1562544000021]
    list_calibrations.assert_called_with(
        'a', 'p', 'timestamp >= 1562500000000 AND timestamp <= 1562600000000')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_calibration')
def test_get_calibration(get_calibration):
    get_calibration.return_value = _CALIBRATION
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    calibration = processor.get_calibration(1562544000021)
    assert calibration.timestamp == 1562544000021
    assert set(calibration.keys()) == {'xeb', 't1', 'globalMetric'}
    get_calibration.assert_called_once_with('a', 'p', 1562544000021)


@mock.patch(
    'cirq.google.engine.engine_client.EngineClient.get_current_calibration')
def test_current_calibration(get_current_calibration):
    get_current_calibration.return_value = _CALIBRATION
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    calibration = processor.get_current_calibration()
    assert calibration.timestamp == 1562544000021
    assert set(calibration.keys()) == {'xeb', 't1', 'globalMetric'}
    get_current_calibration.assert_called_once_with('a', 'p')


@mock.patch(
    'cirq.google.engine.engine_client.EngineClient.get_current_calibration')
def test_missing_latest_calibration(get_current_calibration):
    get_current_calibration.return_value = None
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert not processor.get_current_calibration()
    get_current_calibration.assert_called_once_with('a', 'p')


def test_str():
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert str(
        processor) == 'EngineProcessor(project_id=\'a\', processor_id=\'p\')'
