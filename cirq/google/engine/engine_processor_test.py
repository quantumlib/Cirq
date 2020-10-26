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
import freezegun
from google.protobuf.duration_pb2 import Duration
from google.protobuf.text_format import Merge
from google.protobuf.timestamp_pb2 import Timestamp
import cirq
import cirq.google as cg
from cirq.google.api import v2
from cirq.google.engine.engine import EngineContext
from cirq.google.engine.client.quantum_v1alpha1 import enums as qenums
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


_GATE_SET = cg.SerializableGateSet(
    gate_set_name='x_gate_set',
    serializers=[
        cg.GateOpSerializer(gate_type=cirq.XPowGate,
                            serialized_gate_id='x',
                            args=[])
    ],
    deserializers=[
        cg.GateOpDeserializer(serialized_gate_id='x',
                              gate_constructor=cirq.XPowGate,
                              args=[])
    ],
)


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


def test_get_device():
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=qtypes.QuantumProcessor(device_spec=_DEVICE_SPEC))
    device = processor.get_device(gate_sets=[_GATE_SET])
    assert device.qubits == [cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)]
    device.validate_operation(cirq.X(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.X(cirq.GridQubit(1, 2)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.Y(cirq.GridQubit(0, 0)))


def test_get_missing_device():
    processor = cg.EngineProcessor('a',
                                   'p',
                                   EngineContext(),
                                   _processor=qtypes.QuantumProcessor())
    with pytest.raises(ValueError, match='device specification'):
        _ = processor.get_device(gate_sets=[_GATE_SET])


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


@mock.patch('cirq.google.engine.engine_client.EngineClient.create_reservation')
def test_create_reservation(create_reservation):
    name = 'projects/proj/processors/p0/reservations/psherman-wallaby-way'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    create_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.create_reservation(
        datetime.datetime.fromtimestamp(1000000000),
        datetime.datetime.fromtimestamp(1000003600), ['dstrain@google.com'])
    create_reservation.assert_called_once_with(
        'proj', 'p0', datetime.datetime.fromtimestamp(1000000000),
        datetime.datetime.fromtimestamp(1000003600), ['dstrain@google.com'])


@mock.patch('cirq.google.engine.engine_client.EngineClient.delete_reservation')
def test_delete_reservation(delete_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    delete_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor._delete_reservation('rid') == result
    delete_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq.google.engine.engine_client.EngineClient.cancel_reservation')
def test_cancel_reservation(cancel_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    cancel_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor._cancel_reservation('rid') == result
    cancel_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_reservation')
@mock.patch('cirq.google.engine.engine_client.EngineClient.delete_reservation')
def test_remove_reservation_delete(delete_reservation, get_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    now = int(datetime.datetime.now().timestamp())
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=now + 20000),
        end_time=Timestamp(seconds=now + 23610),
        whitelisted_users=['dstrain@google.com'],
    )
    get_reservation.return_value = result
    delete_reservation.return_value = result
    processor = cg.EngineProcessor(
        'proj', 'p0', EngineContext(),
        qtypes.QuantumProcessor(schedule_frozen_period=Duration(seconds=10000)))
    assert processor.remove_reservation('rid') == result
    delete_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_reservation')
@mock.patch('cirq.google.engine.engine_client.EngineClient.cancel_reservation')
def test_remove_reservation_cancel(cancel_reservation, get_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    now = int(datetime.datetime.now().timestamp())
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=now + 10),
        end_time=Timestamp(seconds=now + 3610),
        whitelisted_users=['dstrain@google.com'],
    )
    get_reservation.return_value = result
    cancel_reservation.return_value = result
    processor = cg.EngineProcessor(
        'proj', 'p0', EngineContext(),
        qtypes.QuantumProcessor(schedule_frozen_period=Duration(seconds=10000)))
    assert processor.remove_reservation('rid') == result
    cancel_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_reservation')
def test_remove_reservation_not_found(get_reservation):
    get_reservation.return_value = None
    processor = cg.EngineProcessor(
        'proj', 'p0', EngineContext(),
        qtypes.QuantumProcessor(schedule_frozen_period=Duration(seconds=10000)))
    with pytest.raises(ValueError):
        processor.remove_reservation('rid')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_processor')
@mock.patch('cirq.google.engine.engine_client.EngineClient.get_reservation')
def test_remove_reservation_failures(get_reservation, get_processor):
    name = 'projects/proj/processors/p0/reservations/rid'
    now = int(datetime.datetime.now().timestamp())
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=now + 10),
        end_time=Timestamp(seconds=now + 3610),
        whitelisted_users=['dstrain@google.com'],
    )
    get_reservation.return_value = result
    get_processor.return_value = None

    # no processor
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    with pytest.raises(ValueError):
        processor.remove_reservation('rid')

    # No freeze period defined
    processor = cg.EngineProcessor('proj', 'p0', EngineContext(),
                                   qtypes.QuantumProcessor())
    with pytest.raises(ValueError):
        processor.remove_reservation('rid')


@mock.patch('cirq.google.engine.engine_client.EngineClient.get_reservation')
def test_get_reservation(get_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    get_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.get_reservation('rid') == result
    get_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq.google.engine.engine_client.EngineClient.update_reservation')
def test_update_reservation(update_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    result = qtypes.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    start = datetime.datetime.fromtimestamp(1000000000)
    end = datetime.datetime.fromtimestamp(1000003600)
    update_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.update_reservation('rid', start, end,
                                        ['dstrain@google.com']) == result
    update_reservation.assert_called_once_with(
        'proj',
        'p0',
        'rid',
        start=start,
        end=end,
        whitelisted_users=['dstrain@google.com'])


@mock.patch('cirq.google.engine.engine_client.EngineClient.list_reservations')
def test_list_reservation(list_reservations):
    name = 'projects/proj/processors/p0/reservations/rid'
    results = [
        qtypes.QuantumReservation(
            name=name,
            start_time=Timestamp(seconds=1000000000),
            end_time=Timestamp(seconds=1000003600),
            whitelisted_users=['dstrain@google.com'],
        ),
        qtypes.QuantumReservation(
            name=name + '2',
            start_time=Timestamp(seconds=1000003600),
            end_time=Timestamp(seconds=1000007200),
            whitelisted_users=['wcourtney@google.com'],
        ),
    ]
    list_reservations.return_value = results
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.list_reservations(
        datetime.datetime.fromtimestamp(1000000000),
        datetime.datetime.fromtimestamp(1000010000)) == results
    list_reservations.assert_called_once_with(
        'proj', 'p0', 'start_time < 1000010000 AND end_time > 1000000000')


@mock.patch('cirq.google.engine.engine_client.EngineClient.list_time_slots')
def test_get_schedule(list_time_slots):
    results = [
        qtypes.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000020000),
            end_time=Timestamp(seconds=1000040000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
            maintenance_config=qtypes.QuantumTimeSlot.MaintenanceConfig(
                title='Testing',
                description='Testing some new configuration.',
            ),
        ),
        qtypes.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000010000),
            end_time=Timestamp(seconds=1000020000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.RESERVATION,
            reservation_config=qtypes.QuantumTimeSlot.ReservationConfig(
                project_id='super_secret_quantum'),
        )
    ]
    list_time_slots.return_value = results
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.get_schedule(
        datetime.datetime.fromtimestamp(1000000000),
        datetime.datetime.fromtimestamp(1000050000)) == results
    list_time_slots.assert_called_once_with(
        'proj', 'p0', 'start_time < 1000050000 AND end_time > 1000000000')


@mock.patch('cirq.google.engine.engine_client.EngineClient.list_time_slots')
def test_get_schedule_filter_by_time_slot(list_time_slots):
    results = [
        qtypes.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000020000),
            end_time=Timestamp(seconds=1000040000),
            slot_type=qenums.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
            maintenance_config=qtypes.QuantumTimeSlot.MaintenanceConfig(
                title='Testing',
                description='Testing some new configuration.',
            ),
        )
    ]
    list_time_slots.return_value = results
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())

    assert processor.get_schedule(
        datetime.datetime.fromtimestamp(1000000000),
        datetime.datetime.fromtimestamp(1000050000),
        qenums.QuantumTimeSlot.TimeSlotType.MAINTENANCE) == results
    list_time_slots.assert_called_once_with(
        'proj', 'p0', 'start_time < 1000050000 AND end_time > 1000000000 AND ' +
        'time_slot_type = MAINTENANCE')


@freezegun.freeze_time()
@mock.patch('cirq.google.engine.engine_client.EngineClient.list_time_slots')
def test_get_schedule_time_filter_behavior(list_time_slots):
    list_time_slots.return_value = []
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())

    now = int(datetime.datetime.now().timestamp())
    in_two_weeks = int(
        (datetime.datetime.now() + datetime.timedelta(weeks=2)).timestamp())
    processor.get_schedule()
    list_time_slots.assert_called_with(
        'proj', 'p0', f'start_time < {in_two_weeks} AND end_time > {now}')

    with pytest.raises(ValueError, match='from_time of type'):
        processor.get_schedule(from_time=object())

    with pytest.raises(ValueError, match='to_time of type'):
        processor.get_schedule(to_time=object())

    processor.get_schedule(from_time=None, to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', '')

    processor.get_schedule(from_time=datetime.timedelta(0), to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {now}')

    processor.get_schedule(from_time=datetime.timedelta(seconds=200),
                           to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {now + 200}')

    test_timestamp = datetime.datetime.utcfromtimestamp(52)
    utc_ts = int(test_timestamp.timestamp())
    processor.get_schedule(from_time=test_timestamp, to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {utc_ts}')

    processor.get_schedule(from_time=None, to_time=datetime.timedelta(0))
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {now}')

    processor.get_schedule(from_time=None,
                           to_time=datetime.timedelta(seconds=200))
    list_time_slots.assert_called_with('proj', 'p0',
                                       f'start_time < {now + 200}')

    processor.get_schedule(from_time=None, to_time=test_timestamp)
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {utc_ts}')


@freezegun.freeze_time()
@mock.patch('cirq.google.engine.engine_client.EngineClient.list_reservations')
def test_list_reservations_time_filter_behavior(list_reservations):
    list_reservations.return_value = []
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())

    now = int(datetime.datetime.now().timestamp())
    in_two_weeks = int(
        (datetime.datetime.now() + datetime.timedelta(weeks=2)).timestamp())
    processor.list_reservations()
    list_reservations.assert_called_with(
        'proj', 'p0', f'start_time < {in_two_weeks} AND end_time > {now}')

    with pytest.raises(ValueError, match='from_time of type'):
        processor.list_reservations(from_time=object())

    with pytest.raises(ValueError, match='to_time of type'):
        processor.list_reservations(to_time=object())

    processor.list_reservations(from_time=None, to_time=None)
    list_reservations.assert_called_with('proj', 'p0', '')

    processor.list_reservations(from_time=datetime.timedelta(0), to_time=None)
    list_reservations.assert_called_with('proj', 'p0', f'end_time > {now}')

    processor.list_reservations(from_time=datetime.timedelta(seconds=200),
                                to_time=None)
    list_reservations.assert_called_with('proj', 'p0',
                                         f'end_time > {now + 200}')

    test_timestamp = datetime.datetime.utcfromtimestamp(52)
    utc_ts = int(test_timestamp.timestamp())
    processor.list_reservations(from_time=test_timestamp, to_time=None)
    list_reservations.assert_called_with('proj', 'p0', f'end_time > {utc_ts}')

    processor.list_reservations(from_time=None, to_time=datetime.timedelta(0))
    list_reservations.assert_called_with('proj', 'p0', f'start_time < {now}')

    processor.list_reservations(from_time=None,
                                to_time=datetime.timedelta(seconds=200))
    list_reservations.assert_called_with('proj', 'p0',
                                         f'start_time < {now + 200}')

    processor.list_reservations(from_time=None, to_time=test_timestamp)
    list_reservations.assert_called_with('proj', 'p0', f'start_time < {utc_ts}')


def test_str():
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert str(
        processor) == 'EngineProcessor(project_id=\'a\', processor_id=\'p\')'
