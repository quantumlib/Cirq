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

import duet
import pytest
import freezegun
import numpy as np

from google.protobuf.duration_pb2 import Duration
from google.protobuf.text_format import Merge
from google.protobuf.timestamp_pb2 import Timestamp
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.engine import engine_client, util
from cirq_google.engine.engine import EngineContext
from cirq_google.cloud import quantum


def _to_timestamp(json_string):
    timestamp_proto = Timestamp()
    timestamp_proto.FromJsonString(json_string)
    return timestamp_proto


_CALIBRATION = quantum.QuantumCalibration(
    name='projects/a/processors/p/calibrations/1562715599',
    timestamp=_to_timestamp('2019-07-09T23:39:59Z'),
    data=util.pack_any(
        v2.metrics_pb2.MetricsSnapshot(
            timestamp_ms=1562544000021,
            metrics=[
                v2.metrics_pb2.Metric(
                    name='xeb',
                    targets=['0_0', '0_1'],
                    values=[v2.metrics_pb2.Value(double_val=0.9999)],
                ),
                v2.metrics_pb2.Metric(
                    name='xeb',
                    targets=['0_0', '1_0'],
                    values=[v2.metrics_pb2.Value(double_val=0.9998)],
                ),
                v2.metrics_pb2.Metric(
                    name='t1', targets=['0_0'], values=[v2.metrics_pb2.Value(double_val=321)]
                ),
                v2.metrics_pb2.Metric(
                    name='t1', targets=['0_1'], values=[v2.metrics_pb2.Value(double_val=911)]
                ),
                v2.metrics_pb2.Metric(
                    name='t1', targets=['0_1'], values=[v2.metrics_pb2.Value(double_val=505)]
                ),
                v2.metrics_pb2.Metric(
                    name='globalMetric', values=[v2.metrics_pb2.Value(int32_val=12300)]
                ),
            ],
        )
    ),
)

_DEVICE_SPEC = util.pack_any(
    Merge(
        """
valid_qubits: "0_0"
valid_qubits: "1_1"
valid_qubits: "2_2"
valid_targets {
  name: "2_qubit_targets"
  target_ordering: SYMMETRIC
  targets {
    ids: "0_0"
    ids: "1_1"
  }
}
valid_gates {
  gate_duration_picos: 1000
  cz {
  }
}
valid_gates {
  phased_xz {
  }
}
""",
        v2.device_pb2.DeviceSpecification(),
    )
)


_CIRCUIT = cirq.Circuit(
    cirq.X(cirq.GridQubit(5, 2)) ** 0.5, cirq.measure(cirq.GridQubit(5, 2), key='result')
)


_RESULTS_V2 = v2.result_pb2.Result(
    sweep_results=[
        v2.result_pb2.SweepResult(
            repetitions=1,
            parameterized_results=[
                v2.result_pb2.ParameterizedResult(
                    params=v2.result_pb2.ParameterDict(assignments={'a': 1}),
                    measurement_results=[
                        v2.result_pb2.MeasurementResult(
                            key='q',
                            qubit_measurement_results=[
                                v2.result_pb2.QubitMeasurementResult(
                                    qubit=v2.program_pb2.Qubit(id='1_1'), results=b'\000\001'
                                )
                            ],
                        )
                    ],
                ),
                v2.result_pb2.ParameterizedResult(
                    params=v2.result_pb2.ParameterDict(assignments={'a': 2}),
                    measurement_results=[
                        v2.result_pb2.MeasurementResult(
                            key='q',
                            qubit_measurement_results=[
                                v2.result_pb2.QubitMeasurementResult(
                                    qubit=v2.program_pb2.Qubit(id='1_1'), results=b'\000\001'
                                )
                            ],
                        )
                    ],
                ),
            ],
        )
    ]
)


_RESULTS2_V2 = v2.result_pb2.Result(
    sweep_results=[
        v2.result_pb2.SweepResult(
            repetitions=1,
            parameterized_results=[
                v2.result_pb2.ParameterizedResult(
                    params=v2.result_pb2.ParameterDict(assignments={'a': 3}),
                    measurement_results=[
                        v2.result_pb2.MeasurementResult(
                            key='q',
                            qubit_measurement_results=[
                                v2.result_pb2.QubitMeasurementResult(
                                    qubit=v2.program_pb2.Qubit(id='1_1'), results=b'\000\001'
                                )
                            ],
                        )
                    ],
                ),
                v2.result_pb2.ParameterizedResult(
                    params=v2.result_pb2.ParameterDict(assignments={'a': 4}),
                    measurement_results=[
                        v2.result_pb2.MeasurementResult(
                            key='q',
                            qubit_measurement_results=[
                                v2.result_pb2.QubitMeasurementResult(
                                    qubit=v2.program_pb2.Qubit(id='1_1'), results=b'\000\001'
                                )
                            ],
                        )
                    ],
                ),
            ],
        )
    ]
)


class FakeEngineContext(EngineContext):
    """Fake engine context for testing."""

    def __init__(self, client: engine_client.EngineClient):
        super().__init__()
        self.client = client


@pytest.fixture(scope='module', autouse=True)
def mock_grpc_client():
    with mock.patch(
        'cirq_google.engine.engine_client.quantum.QuantumEngineServiceClient'
    ) as _fixture:
        yield _fixture


def test_engine():
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert processor.engine().project_id == 'a'


def test_engine_repr():
    processor = cg.EngineProcessor('the-project-id', 'the-processor-id', EngineContext())
    assert 'the-project-id' in repr(processor)
    assert 'the-processor-id' in repr(processor)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor_async')
def test_health(get_processor):
    get_processor.return_value = quantum.QuantumProcessor(health=quantum.QuantumProcessor.Health.OK)
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=quantum.QuantumProcessor(health=quantum.QuantumProcessor.Health.DOWN),
    )
    assert processor.health() == 'OK'


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor_async')
def test_expected_down_time(get_processor):
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    assert not processor.expected_down_time()

    get_processor.return_value = quantum.QuantumProcessor(
        expected_down_time=Timestamp(seconds=1581515101)
    )

    assert cg.EngineProcessor('a', 'p', EngineContext()).expected_down_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc
    )
    get_processor.assert_called_once()


def test_expected_recovery_time():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    assert not processor.expected_recovery_time()
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=quantum.QuantumProcessor(expected_recovery_time=Timestamp(seconds=1581515101)),
    )
    assert processor.expected_recovery_time() == datetime.datetime(
        2020, 2, 12, 13, 45, 1, tzinfo=datetime.timezone.utc
    )


def test_supported_languages():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    assert processor.supported_languages() == []
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=quantum.QuantumProcessor(supported_languages=['lang1', 'lang2']),
    )
    assert processor.supported_languages() == ['lang1', 'lang2']


def test_get_device_specification():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    assert processor.get_device_specification() is None

    # Construct expected device proto based on example
    expected = v2.device_pb2.DeviceSpecification()
    expected.valid_qubits.extend(['0_0', '1_1', '2_2'])
    target = expected.valid_targets.add()
    target.name = '2_qubit_targets'
    target.target_ordering = v2.device_pb2.TargetSet.SYMMETRIC
    new_target = target.targets.add()
    new_target.ids.extend(['0_0', '1_1'])
    gate = expected.valid_gates.add()
    gate.cz.SetInParent()
    gate.gate_duration_picos = 1000
    gate = expected.valid_gates.add()
    gate.phased_xz.SetInParent()

    processor = cg.EngineProcessor(
        'a', 'p', EngineContext(), _processor=quantum.QuantumProcessor(device_spec=_DEVICE_SPEC)
    )
    assert processor.get_device_specification() == expected


def test_get_device():
    processor = cg.EngineProcessor(
        'a', 'p', EngineContext(), _processor=quantum.QuantumProcessor(device_spec=_DEVICE_SPEC)
    )
    device = processor.get_device()
    assert device.metadata.qubit_set == frozenset(
        [cirq.GridQubit(0, 0), cirq.GridQubit(1, 1), cirq.GridQubit(2, 2)]
    )
    device.validate_operation(cirq.X(cirq.GridQubit(2, 2)))
    device.validate_operation(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)))
    with pytest.raises(ValueError):
        device.validate_operation(cirq.X(cirq.GridQubit(1, 2)))
    with pytest.raises(ValueError):
        device.validate_operation(
            cirq.testing.DoesNotSupportSerializationGate()(cirq.GridQubit(0, 0))
        )
    with pytest.raises(ValueError):
        device.validate_operation(cirq.CZ(cirq.GridQubit(1, 1), cirq.GridQubit(2, 2)))


def test_get_missing_device():
    processor = cg.EngineProcessor('a', 'p', EngineContext(), _processor=quantum.QuantumProcessor())
    with pytest.raises(ValueError, match='device specification'):
        _ = processor.get_device()


def test_get_sampler_initializes_default_device_configuration() -> None:
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=quantum.QuantumProcessor(
            default_device_config_key=quantum.DeviceConfigKey(
                run="run", config_alias="config_alias"
            )
        ),
    )
    sampler = processor.get_sampler()

    assert sampler.run_name == "run"
    assert sampler.device_config_name == "config_alias"


def test_get_sampler_uses_custom_default_device_configuration_key() -> None:
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=quantum.QuantumProcessor(
            default_device_config_key=quantum.DeviceConfigKey(
                run="default_run", config_alias="default_config_alias"
            )
        ),
    )
    sampler = processor.get_sampler(run_name="run1", device_config_name="config_alias1")

    assert sampler.run_name == "run1"
    assert sampler.device_config_name == "config_alias1"


@pytest.mark.parametrize(
    'run, snapshot_id, config_alias, error_message',
    [
        ('run', '', '', 'Cannot specify only one of top level identifier and `device_config_name`'),
        (
            '',
            '',
            'config',
            'Cannot specify only one of top level identifier and `device_config_name`',
        ),
        ('run', 'snapshot_id', 'config', 'Cannot specify both `run_name` and `snapshot_id`'),
    ],
)
def test_get_sampler_with_incomplete_device_configuration_errors(
    run, snapshot_id, config_alias, error_message
) -> None:
    processor = cg.EngineProcessor(
        'a',
        'p',
        EngineContext(),
        _processor=quantum.QuantumProcessor(
            default_device_config_key=quantum.DeviceConfigKey(
                run="default_run", config_alias="default_config_alias"
            )
        ),
    )

    with pytest.raises(ValueError, match=error_message):
        processor.get_sampler(
            run_name=run, device_config_name=config_alias, snapshot_id=snapshot_id
        )


def test_get_sampler_loads_processor_with_default_device_configuration() -> None:
    client = mock.Mock(engine_client.EngineClient)
    client.get_processor.return_value = quantum.QuantumProcessor(
        default_device_config_key=quantum.DeviceConfigKey(run="run", config_alias="config_alias")
    )

    processor = cg.EngineProcessor('a', 'p', FakeEngineContext(client=client))
    sampler = processor.get_sampler()

    assert sampler.run_name == "run"
    assert sampler.device_config_name == "config_alias"


@mock.patch('cirq_google.engine.engine_client.EngineClient.list_calibrations_async')
def test_list_calibrations(list_calibrations):
    list_calibrations.return_value = [_CALIBRATION]
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert [c.timestamp for c in processor.list_calibrations()] == [1562544000021]
    list_calibrations.assert_called_with('a', 'p', '')
    assert [c.timestamp for c in processor.list_calibrations(earliest_timestamp=1562500000)] == [
        1562544000021
    ]
    list_calibrations.assert_called_with('a', 'p', 'timestamp >= 1562500000')
    assert [c.timestamp for c in processor.list_calibrations(latest_timestamp=1562600000)] == [
        1562544000021
    ]
    list_calibrations.assert_called_with('a', 'p', 'timestamp <= 1562600000')
    assert [c.timestamp for c in processor.list_calibrations(1562500000, 1562600000)] == [
        1562544000021
    ]
    list_calibrations.assert_called_with(
        'a', 'p', 'timestamp >= 1562500000 AND timestamp <= 1562600000'
    )
    assert [
        c.timestamp
        for c in processor.list_calibrations(
            earliest_timestamp=datetime.datetime.fromtimestamp(1562500000)
        )
    ] == [1562544000021]
    list_calibrations.assert_called_with('a', 'p', 'timestamp >= 1562500000')

    today = datetime.date.today()
    # Use local time to get timestamp
    today_midnight_timestamp = int(
        datetime.datetime(today.year, today.month, today.day).timestamp()
    )
    assert [c.timestamp for c in processor.list_calibrations(earliest_timestamp=today)] == [
        1562544000021
    ]
    list_calibrations.assert_called_with('a', 'p', f'timestamp >= {today_midnight_timestamp}')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_calibration_async')
def test_get_calibration(get_calibration):
    get_calibration.return_value = _CALIBRATION
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    calibration = processor.get_calibration(1562544000021)
    assert calibration.timestamp == 1562544000021
    assert set(calibration.keys()) == {'xeb', 't1', 'globalMetric'}
    get_calibration.assert_called_once_with('a', 'p', 1562544000021)


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_current_calibration_async')
def test_current_calibration(get_current_calibration):
    get_current_calibration.return_value = _CALIBRATION
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    calibration = processor.get_current_calibration()
    assert calibration.timestamp == 1562544000021
    assert set(calibration.keys()) == {'xeb', 't1', 'globalMetric'}
    get_current_calibration.assert_called_once_with('a', 'p')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_current_calibration_async')
def test_missing_latest_calibration(get_current_calibration):
    get_current_calibration.return_value = None
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert not processor.get_current_calibration()
    get_current_calibration.assert_called_once_with('a', 'p')


@mock.patch('cirq_google.engine.engine_client.EngineClient.create_reservation_async')
def test_create_reservation(create_reservation):
    name = 'projects/proj/processors/p0/reservations/psherman-wallaby-way'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    create_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.create_reservation(
        datetime.datetime.fromtimestamp(1000000000),
        datetime.datetime.fromtimestamp(1000003600),
        ['dstrain@google.com'],
    )
    create_reservation.assert_called_once_with(
        'proj',
        'p0',
        datetime.datetime.fromtimestamp(1000000000),
        datetime.datetime.fromtimestamp(1000003600),
        ['dstrain@google.com'],
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_reservation_async')
def test_delete_reservation(delete_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    delete_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor._delete_reservation('rid') == result
    delete_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq_google.engine.engine_client.EngineClient.cancel_reservation_async')
def test_cancel_reservation(cancel_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    cancel_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor._cancel_reservation('rid') == result
    cancel_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_reservation_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.delete_reservation_async')
def test_remove_reservation_delete(delete_reservation, get_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    now = int(datetime.datetime.now().timestamp())
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=now + 20000),
        end_time=Timestamp(seconds=now + 23610),
        whitelisted_users=['dstrain@google.com'],
    )
    get_reservation.return_value = result
    delete_reservation.return_value = result
    processor = cg.EngineProcessor(
        'proj',
        'p0',
        EngineContext(),
        quantum.QuantumProcessor(schedule_frozen_period=Duration(seconds=10000)),
    )
    assert processor.remove_reservation('rid') == result
    delete_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_reservation_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.cancel_reservation_async')
def test_remove_reservation_cancel(cancel_reservation, get_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    now = int(datetime.datetime.now().timestamp())
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=now + 10),
        end_time=Timestamp(seconds=now + 3610),
        whitelisted_users=['dstrain@google.com'],
    )
    get_reservation.return_value = result
    cancel_reservation.return_value = result
    processor = cg.EngineProcessor(
        'proj',
        'p0',
        EngineContext(),
        quantum.QuantumProcessor(schedule_frozen_period=Duration(seconds=10000)),
    )
    assert processor.remove_reservation('rid') == result
    cancel_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_reservation_async')
def test_remove_reservation_not_found(get_reservation):
    get_reservation.return_value = None
    processor = cg.EngineProcessor(
        'proj',
        'p0',
        EngineContext(),
        quantum.QuantumProcessor(schedule_frozen_period=Duration(seconds=10000)),
    )
    with pytest.raises(ValueError):
        processor.remove_reservation('rid')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_processor_async')
@mock.patch('cirq_google.engine.engine_client.EngineClient.get_reservation_async')
def test_remove_reservation_failures(get_reservation, get_processor):
    name = 'projects/proj/processors/p0/reservations/rid'
    now = int(datetime.datetime.now().timestamp())
    result = quantum.QuantumReservation(
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
    processor = cg.EngineProcessor('proj', 'p0', EngineContext(), quantum.QuantumProcessor())
    with pytest.raises(ValueError):
        processor.remove_reservation('rid')


@mock.patch('cirq_google.engine.engine_client.EngineClient.get_reservation_async')
def test_get_reservation(get_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    get_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.get_reservation('rid') == result
    get_reservation.assert_called_once_with('proj', 'p0', 'rid')


@mock.patch('cirq_google.engine.engine_client.EngineClient.update_reservation_async')
def test_update_reservation(update_reservation):
    name = 'projects/proj/processors/p0/reservations/rid'
    result = quantum.QuantumReservation(
        name=name,
        start_time=Timestamp(seconds=1000000000),
        end_time=Timestamp(seconds=1000003600),
        whitelisted_users=['dstrain@google.com'],
    )
    start = datetime.datetime.fromtimestamp(1000000000)
    end = datetime.datetime.fromtimestamp(1000003600)
    update_reservation.return_value = result
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert processor.update_reservation('rid', start, end, ['dstrain@google.com']) == result
    update_reservation.assert_called_once_with(
        'proj', 'p0', 'rid', start=start, end=end, whitelisted_users=['dstrain@google.com']
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.list_reservations_async')
def test_list_reservation(list_reservations):
    name = 'projects/proj/processors/p0/reservations/rid'
    results = [
        quantum.QuantumReservation(
            name=name,
            start_time=Timestamp(seconds=1000000000),
            end_time=Timestamp(seconds=1000003600),
            whitelisted_users=['dstrain@google.com'],
        ),
        quantum.QuantumReservation(
            name=name + '2',
            start_time=Timestamp(seconds=1000003600),
            end_time=Timestamp(seconds=1000007200),
            whitelisted_users=['wcourtney@google.com'],
        ),
    ]
    list_reservations.return_value = results
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert (
        processor.list_reservations(
            datetime.datetime.fromtimestamp(1000000000), datetime.datetime.fromtimestamp(1000010000)
        )
        == results
    )
    list_reservations.assert_called_once_with(
        'proj', 'p0', 'start_time < 1000010000 AND end_time > 1000000000'
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.list_time_slots_async')
def test_get_schedule(list_time_slots):
    results = [
        quantum.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000020000),
            end_time=Timestamp(seconds=1000040000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
            maintenance_config=quantum.QuantumTimeSlot.MaintenanceConfig(
                title='Testing', description='Testing some new configuration.'
            ),
        ),
        quantum.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000010000),
            end_time=Timestamp(seconds=1000020000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.RESERVATION,
            reservation_config=quantum.QuantumTimeSlot.ReservationConfig(
                project_id='super_secret_quantum'
            ),
        ),
    ]
    list_time_slots.return_value = results
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())
    assert (
        processor.get_schedule(
            datetime.datetime.fromtimestamp(1000000000), datetime.datetime.fromtimestamp(1000050000)
        )
        == results
    )
    list_time_slots.assert_called_once_with(
        'proj', 'p0', 'start_time < 1000050000 AND end_time > 1000000000'
    )


@mock.patch('cirq_google.engine.engine_client.EngineClient.list_time_slots_async')
def test_get_schedule_filter_by_time_slot(list_time_slots):
    results = [
        quantum.QuantumTimeSlot(
            processor_name='potofgold',
            start_time=Timestamp(seconds=1000020000),
            end_time=Timestamp(seconds=1000040000),
            time_slot_type=quantum.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
            maintenance_config=quantum.QuantumTimeSlot.MaintenanceConfig(
                title='Testing', description='Testing some new configuration.'
            ),
        )
    ]
    list_time_slots.return_value = results
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())

    assert (
        processor.get_schedule(
            datetime.datetime.fromtimestamp(1000000000),
            datetime.datetime.fromtimestamp(1000050000),
            quantum.QuantumTimeSlot.TimeSlotType.MAINTENANCE,
        )
        == results
    )
    list_time_slots.assert_called_once_with(
        'proj',
        'p0',
        'start_time < 1000050000 AND end_time > 1000000000 AND ' + 'time_slot_type = MAINTENANCE',
    )


def _allow_deprecated_freezegun(func):
    # a local hack, as freeze_time walks through all the sys.modules, and retrieves all the
    # attributes for all modules when it reaches deprecated module attributes, we throw an error
    # as the deprecation module thinks Cirq is using something deprecated. This hack SHOULD NOT be
    # used elsewhere, it is specific to freezegun functionality.
    def wrapper(*args, **kwargs):
        import os
        from cirq.testing.deprecation import ALLOW_DEPRECATION_IN_TEST

        orig_exist, orig_value = (
            ALLOW_DEPRECATION_IN_TEST in os.environ,
            os.environ.get(ALLOW_DEPRECATION_IN_TEST, None),
        )

        os.environ[ALLOW_DEPRECATION_IN_TEST] = 'True'
        try:
            return func(*args, **kwargs)
        finally:
            if orig_exist:
                # mypy can't resolve that orig_exist ensures that orig_value
                # of type Optional[str] can't be None
                os.environ[ALLOW_DEPRECATION_IN_TEST] = orig_value  # pragma: no cover
            else:
                del os.environ[ALLOW_DEPRECATION_IN_TEST]

    return wrapper


@_allow_deprecated_freezegun
@freezegun.freeze_time()
@mock.patch('cirq_google.engine.engine_client.EngineClient.list_time_slots_async')
def test_get_schedule_time_filter_behavior(list_time_slots):
    list_time_slots.return_value = []
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())

    now = int(datetime.datetime.now().timestamp())
    in_two_weeks = int((datetime.datetime.now() + datetime.timedelta(weeks=2)).timestamp())
    processor.get_schedule()
    list_time_slots.assert_called_with(
        'proj', 'p0', f'start_time < {in_two_weeks} AND end_time > {now}'
    )

    with pytest.raises(ValueError, match='from_time of type'):
        processor.get_schedule(from_time=object())

    with pytest.raises(ValueError, match='to_time of type'):
        processor.get_schedule(to_time=object())

    processor.get_schedule(from_time=None, to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', '')

    processor.get_schedule(from_time=datetime.timedelta(0), to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {now}')

    processor.get_schedule(from_time=datetime.timedelta(seconds=200), to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {now + 200}')

    test_timestamp = datetime.datetime.utcfromtimestamp(52)
    utc_ts = int(test_timestamp.timestamp())
    processor.get_schedule(from_time=test_timestamp, to_time=None)
    list_time_slots.assert_called_with('proj', 'p0', f'end_time > {utc_ts}')

    processor.get_schedule(from_time=None, to_time=datetime.timedelta(0))
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {now}')

    processor.get_schedule(from_time=None, to_time=datetime.timedelta(seconds=200))
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {now + 200}')

    processor.get_schedule(from_time=None, to_time=test_timestamp)
    list_time_slots.assert_called_with('proj', 'p0', f'start_time < {utc_ts}')


@_allow_deprecated_freezegun
@freezegun.freeze_time()
@mock.patch('cirq_google.engine.engine_client.EngineClient.list_reservations_async')
def test_list_reservations_time_filter_behavior(list_reservations):
    list_reservations.return_value = []
    processor = cg.EngineProcessor('proj', 'p0', EngineContext())

    now = int(datetime.datetime.now().timestamp())
    in_two_weeks = int((datetime.datetime.now() + datetime.timedelta(weeks=2)).timestamp())
    processor.list_reservations()
    list_reservations.assert_called_with(
        'proj', 'p0', f'start_time < {in_two_weeks} AND end_time > {now}'
    )

    with pytest.raises(ValueError, match='from_time of type'):
        processor.list_reservations(from_time=object())

    with pytest.raises(ValueError, match='to_time of type'):
        processor.list_reservations(to_time=object())

    processor.list_reservations(from_time=None, to_time=None)
    list_reservations.assert_called_with('proj', 'p0', '')

    processor.list_reservations(from_time=datetime.timedelta(0), to_time=None)
    list_reservations.assert_called_with('proj', 'p0', f'end_time > {now}')

    processor.list_reservations(from_time=datetime.timedelta(seconds=200), to_time=None)
    list_reservations.assert_called_with('proj', 'p0', f'end_time > {now + 200}')

    test_timestamp = datetime.datetime.utcfromtimestamp(52)
    utc_ts = int(test_timestamp.timestamp())
    processor.list_reservations(from_time=test_timestamp, to_time=None)
    list_reservations.assert_called_with('proj', 'p0', f'end_time > {utc_ts}')

    processor.list_reservations(from_time=None, to_time=datetime.timedelta(0))
    list_reservations.assert_called_with('proj', 'p0', f'start_time < {now}')

    processor.list_reservations(from_time=None, to_time=datetime.timedelta(seconds=200))
    list_reservations.assert_called_with('proj', 'p0', f'start_time < {now + 200}')

    processor.list_reservations(from_time=None, to_time=test_timestamp)
    list_reservations.assert_called_with('proj', 'p0', f'start_time < {utc_ts}')


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_sweep_params_with_unary_rpcs(client):
    client().create_program_async.return_value = (
        'prog',
        quantum.QuantumProgram(name='projects/proj/programs/prog'),
    )
    client().create_job_async.return_value = (
        'job-id',
        quantum.QuantumJob(
            name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'READY'}
        ),
    )
    client().get_job_async.return_value = quantum.QuantumJob(
        execution_status={'state': 'SUCCESS'}, update_time=_to_timestamp('2019-07-09T23:39:59Z')
    )
    client().get_job_results_async.return_value = quantum.QuantumResult(
        result=util.pack_any(_RESULTS_V2)
    )

    processor = cg.EngineProcessor('a', 'p', EngineContext(enable_streaming=False))
    job = processor.run_sweep(
        program=_CIRCUIT,
        params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})],
        run_name="run",
        device_config_name="config_alias",
    )
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    for result in results:
        assert result.job_id == job.id()
        assert result.job_finished_time is not None
    assert results == cirq.read_json(json_text=cirq.to_json(results))

    client().create_program_async.assert_called_once()
    client().create_job_async.assert_called_once()

    run_context = v2.run_context_pb2.RunContext()
    client().create_job_async.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 2
    for i, v in enumerate([1, 2]):
        assert sweeps[i].repetitions == 1
        assert sweeps[i].sweep.sweep_function.sweeps[0].single_sweep.const_value.int_value == v
    client().get_job_async.assert_called_once()
    client().get_job_results_async.assert_called_once()


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_run_sweep_params_with_stream_rpcs(client):
    client().get_job_async.return_value = quantum.QuantumJob(
        execution_status={'state': 'SUCCESS'}, update_time=_to_timestamp('2019-07-09T23:39:59Z')
    )
    expected_result = quantum.QuantumResult(result=util.pack_any(_RESULTS_V2))
    stream_future = duet.AwaitableFuture()
    stream_future.try_set_result(expected_result)
    client().run_job_over_stream.return_value = stream_future

    processor = cg.EngineProcessor('a', 'p', EngineContext(enable_streaming=True))
    job = processor.run_sweep(
        program=_CIRCUIT,
        params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})],
        run_name="run",
        device_config_name="config_alias",
    )
    results = job.results()
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    for result in results:
        assert result.job_id == job.id()
        assert result.job_finished_time is not None
    assert results == cirq.read_json(json_text=cirq.to_json(results))

    client().run_job_over_stream.assert_called_once()

    run_context = v2.run_context_pb2.RunContext()
    client().run_job_over_stream.call_args[1]['run_context'].Unpack(run_context)
    sweeps = run_context.parameter_sweeps
    assert len(sweeps) == 2
    for i, v in enumerate([1, 2]):
        assert sweeps[i].repetitions == 1
        assert sweeps[i].sweep.sweep_function.sweeps[0].single_sweep.const_value.int_value == v


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_sampler_with_unary_rpcs(client):
    client().create_program_async.return_value = (
        'prog',
        quantum.QuantumProgram(name='projects/proj/programs/prog'),
    )
    client().create_job_async.return_value = (
        'job-id',
        quantum.QuantumJob(
            name='projects/proj/programs/prog/jobs/job-id', execution_status={'state': 'READY'}
        ),
    )
    client().get_job_async.return_value = quantum.QuantumJob(
        execution_status={'state': 'SUCCESS'}, update_time=_to_timestamp('2019-07-09T23:39:59Z')
    )
    client().get_job_results_async.return_value = quantum.QuantumResult(
        result=util.pack_any(_RESULTS_V2)
    )
    processor = cg.EngineProcessor('proj', 'mysim', EngineContext(enable_streaming=False))
    sampler = processor.get_sampler()
    results = sampler.run_sweep(
        program=_CIRCUIT, params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})]
    )
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert client().create_program_async.call_args[0][0] == 'proj'


@mock.patch('cirq_google.engine.engine_client.EngineClient', autospec=True)
def test_sampler_with_stream_rpcs(client):
    client().get_job_async.return_value = quantum.QuantumJob(
        execution_status={'state': 'SUCCESS'}, update_time=_to_timestamp('2019-07-09T23:39:59Z')
    )
    expected_result = quantum.QuantumResult(result=util.pack_any(_RESULTS_V2))
    stream_future = duet.AwaitableFuture()
    stream_future.try_set_result(expected_result)
    client().run_job_over_stream.return_value = stream_future

    processor = cg.EngineProcessor('proj', 'mysim', EngineContext(enable_streaming=True))
    sampler = processor.get_sampler()
    results = sampler.run_sweep(
        program=_CIRCUIT, params=[cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 2})]
    )
    assert len(results) == 2
    for i, v in enumerate([1, 2]):
        assert results[i].repetitions == 1
        assert results[i].params.param_dict == {'a': v}
        assert results[i].measurements == {'q': np.array([[0]], dtype='uint8')}
    assert client().run_job_over_stream.call_args[1]['project_id'] == 'proj'


def test_str():
    processor = cg.EngineProcessor('a', 'p', EngineContext())
    assert str(processor) == 'EngineProcessor(project_id=\'a\', processor_id=\'p\')'
