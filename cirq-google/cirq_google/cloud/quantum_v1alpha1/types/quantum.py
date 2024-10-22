# -*- coding: utf-8 -*-
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import proto

from google.protobuf import any_pb2
from google.protobuf import duration_pb2
from google.protobuf import field_mask_pb2
from google.protobuf import timestamp_pb2


__protobuf__ = proto.module(
    package='google.cloud.quantum.v1alpha1',
    manifest={
        'QuantumProgram',
        'QuantumJob',
        'SchedulingConfig',
        'ExecutionStatus',
        'OutputConfig',
        'GcsLocation',
        'InlineData',
        'QuantumJobEvent',
        'QuantumResult',
        'QuantumProcessor',
        'QuantumCalibration',
        'QuantumReservationGrant',
        'QuantumReservationBudget',
        'QuantumTimeSlot',
        'QuantumReservation',
        'DeviceConfigSelector',
        'DeviceConfigKey',
    },
)


class QuantumProgram(proto.Message):
    r"""-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            -
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        labels (Sequence[google.cloud.quantum_v1alpha1.types.QuantumProgram.LabelsEntry]):
            -
        label_fingerprint (str):
            -
        description (str):
            -
        gcs_code_location (google.cloud.quantum_v1alpha1.types.GcsLocation):
            -

            This field is a member of `oneof`_ ``code_location``.
        code_inline_data (google.cloud.quantum_v1alpha1.types.InlineData):
            -

            This field is a member of `oneof`_ ``code_location``.
        code (google.protobuf.any_pb2.Any):
            -
    """

    name = proto.Field(proto.STRING, number=1)
    create_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    update_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    labels = proto.MapField(proto.STRING, proto.STRING, number=4)
    label_fingerprint = proto.Field(proto.STRING, number=5)
    description = proto.Field(proto.STRING, number=6)
    gcs_code_location = proto.Field(
        proto.MESSAGE, number=7, oneof='code_location', message='GcsLocation'
    )
    code_inline_data = proto.Field(
        proto.MESSAGE, number=9, oneof='code_location', message='InlineData'
    )
    code = proto.Field(proto.MESSAGE, number=8, message=any_pb2.Any)


class QuantumJob(proto.Message):
    r"""-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            -
        create_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        update_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        labels (Sequence[google.cloud.quantum_v1alpha1.types.QuantumJob.LabelsEntry]):
            -
        label_fingerprint (str):
            -
        description (str):
            -
        scheduling_config (google.cloud.quantum_v1alpha1.types.SchedulingConfig):
            -
        output_config (google.cloud.quantum_v1alpha1.types.OutputConfig):
            -
        execution_status (google.cloud.quantum_v1alpha1.types.ExecutionStatus):
            -
        gcs_run_context_location (google.cloud.quantum_v1alpha1.types.GcsLocation):
            -

            This field is a member of `oneof`_ ``run_context_location``.
        run_context_inline_data (google.cloud.quantum_v1alpha1.types.InlineData):
            -

            This field is a member of `oneof`_ ``run_context_location``.
        run_context (google.protobuf.any_pb2.Any):
            -
    """

    name = proto.Field(proto.STRING, number=1)
    create_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    update_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    labels = proto.MapField(proto.STRING, proto.STRING, number=4)
    label_fingerprint = proto.Field(proto.STRING, number=5)
    description = proto.Field(proto.STRING, number=6)
    scheduling_config = proto.Field(proto.MESSAGE, number=7, message='SchedulingConfig')
    output_config = proto.Field(proto.MESSAGE, number=8, message='OutputConfig')
    execution_status = proto.Field(proto.MESSAGE, number=9, message='ExecutionStatus')
    gcs_run_context_location = proto.Field(
        proto.MESSAGE, number=10, oneof='run_context_location', message='GcsLocation'
    )
    run_context_inline_data = proto.Field(
        proto.MESSAGE, number=12, oneof='run_context_location', message='InlineData'
    )
    run_context = proto.Field(proto.MESSAGE, number=11, message=any_pb2.Any)


class DeviceConfigSelector(proto.Message):
    r"""-
    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.
    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields
    Attributes:
        run_name (str):
            -
            This field is a member of `oneof`_ ``top_level_identifier``.
        snapshot_id (str):
            -
            This field is a member of `oneof`_ ``top_level_identifier``.
        config_alias (str):
            -
    """

    run_name: str = proto.Field(proto.STRING, number=1, oneof='top_level_identifier')
    snapshot_id: str = proto.Field(proto.STRING, number=3, oneof='top_level_identifier')
    config_alias: str = proto.Field(proto.STRING, number=2)


class DeviceConfigKey(proto.Message):
    r"""-
    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.
    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields
    Attributes:
        run (str):
            -
            This field is a member of `oneof`_ ``top_level_identifier``.
        snapshot_id (str):
            -
            This field is a member of `oneof`_ ``top_level_identifier``.
        config_alias (str):
            -
    """

    run: str = proto.Field(proto.STRING, number=1, oneof='top_level_identifier')
    snapshot_id: str = proto.Field(proto.STRING, number=3, oneof='top_level_identifier')
    config_alias: str = proto.Field(proto.STRING, number=2)


class SchedulingConfig(proto.Message):
    r"""-

    Attributes:
        target_route (str):
            -
        processor_selector (google.cloud.quantum_v1alpha1.types.SchedulingConfig.ProcessorSelector):
            -
        priority (int):
            -
    """

    class ProcessorSelector(proto.Message):
        r"""-

        Attributes:
            processor_names (Sequence[str]):
                -
            processor (str):
                -
            device_config_selector ((google.cloud.quantum_v1alpha1.types.DeviceConfigSelector):
                -
        """

        processor_names = proto.RepeatedField(proto.STRING, number=1)
        processor = proto.Field(proto.STRING, number=2)
        device_config_selector = proto.Field(proto.MESSAGE, number=3, message=DeviceConfigSelector)

    target_route = proto.Field(proto.STRING, number=1)
    processor_selector = proto.Field(proto.MESSAGE, number=3, message=ProcessorSelector)
    priority = proto.Field(proto.INT32, number=2)


class ExecutionStatus(proto.Message):
    r"""-

    Attributes:
        state (google.cloud.quantum_v1alpha1.types.ExecutionStatus.State):
            -
        processor_name (str):
            -
        calibration_name (str):
            -
        failure (google.cloud.quantum_v1alpha1.types.ExecutionStatus.Failure):
            -
        timing (google.cloud.quantum_v1alpha1.types.ExecutionStatus.Timing):
            -
    """

    class State(proto.Enum):
        r"""-"""

        STATE_UNSPECIFIED = 0
        READY = 1
        RUNNING = 2
        CANCELLING = 3
        CANCELLED = 4
        SUCCESS = 5
        FAILURE = 6

    class Failure(proto.Message):
        r"""-

        Attributes:
            error_code (google.cloud.quantum_v1alpha1.types.ExecutionStatus.Failure.Code):
                -
            error_message (str):
                -
        """

        class Code(proto.Enum):
            r"""-"""

            CODE_UNSPECIFIED = 0
            SYSTEM_ERROR = 1
            INVALID_PROGRAM = 2
            INVALID_RUN_CONTEXT = 3
            READ_PROGRAM_NOT_FOUND_IN_GCS = 4
            READ_PROGRAM_PERMISSION_DENIED = 5
            READ_PROGRAM_UNKNOWN_ERROR = 6
            READ_RUN_CONTEXT_NOT_FOUND_IN_GCS = 7
            READ_RUN_CONTEXT_PERMISSION_DENIED = 8
            READ_RUN_CONTEXT_UNKNOWN_ERROR = 9
            WRITE_RESULT_ALREADY_EXISTS_IN_GCS = 10
            WRITE_RESULT_GCS_PERMISSION_DENIED = 11
            SCHEDULING_EXPIRED = 14
            FAILED_PRECONDITION = 15

        error_code = proto.Field(proto.ENUM, number=1, enum='ExecutionStatus.Failure.Code')
        error_message = proto.Field(proto.STRING, number=2)

    class Timing(proto.Message):
        r"""-

        Attributes:
            started_time (google.protobuf.timestamp_pb2.Timestamp):
                -
            completed_time (google.protobuf.timestamp_pb2.Timestamp):
                -
        """

        started_time = proto.Field(proto.MESSAGE, number=1, message=timestamp_pb2.Timestamp)
        completed_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)

    state = proto.Field(proto.ENUM, number=1, enum=State)
    processor_name = proto.Field(proto.STRING, number=3)
    calibration_name = proto.Field(proto.STRING, number=4)
    failure = proto.Field(proto.MESSAGE, number=5, message=Failure)
    timing = proto.Field(proto.MESSAGE, number=6, message=Timing)


class OutputConfig(proto.Message):
    r"""-

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        gcs_results_location (google.cloud.quantum_v1alpha1.types.GcsLocation):
            -

            This field is a member of `oneof`_ ``output_destination``.
        overwrite_existing (bool):
            -
    """

    gcs_results_location = proto.Field(
        proto.MESSAGE, number=1, oneof='output_destination', message='GcsLocation'
    )
    overwrite_existing = proto.Field(proto.BOOL, number=2)


class GcsLocation(proto.Message):
    r"""-

    Attributes:
        uri (str):
            -
        type_url (str):
            -
    """

    uri = proto.Field(proto.STRING, number=1)
    type_url = proto.Field(proto.STRING, number=2)


class InlineData(proto.Message):
    r"""-

    Attributes:
        type_url (str):
            -
    """

    type_url = proto.Field(proto.STRING, number=1)


class QuantumJobEvent(proto.Message):
    r"""-

    Attributes:
        event_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        job (google.cloud.quantum_v1alpha1.types.QuantumJob):
            -
        modified_field_mask (google.protobuf.field_mask_pb2.FieldMask):
            -
    """

    event_time = proto.Field(proto.MESSAGE, number=1, message=timestamp_pb2.Timestamp)
    job = proto.Field(proto.MESSAGE, number=2, message='QuantumJob')
    modified_field_mask = proto.Field(proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask)


class QuantumResult(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        result (google.protobuf.any_pb2.Any):
            -
    """

    parent = proto.Field(proto.STRING, number=1)
    result = proto.Field(proto.MESSAGE, number=2, message=any_pb2.Any)


class QuantumProcessor(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        health (google.cloud.quantum_v1alpha1.types.QuantumProcessor.Health):
            -
        expected_down_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        expected_recovery_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        supported_languages (Sequence[str]):
            -
        device_spec (google.protobuf.any_pb2.Any):
            -
        schedule_horizon (google.protobuf.duration_pb2.Duration):
            -
        schedule_frozen_period (google.protobuf.duration_pb2.Duration):
            -
        current_calibration (str):
            Output only. -
        active_time_slot (google.cloud.quantum_v1alpha1.types.QuantumTimeSlot):
            Output only. -
        activity_stats (google.cloud.quantum_v1alpha1.types.QuantumProcessor.ActivityStats):
            -
        default_device_config_key (google.cloud.quantum_v1alpha1.types.DeviceConfigKey):
            -

    """

    class Health(proto.Enum):
        r"""-"""

        HEALTH_UNSPECIFIED = 0
        OK = 1
        DOWN = 2
        INACTIVE = 4
        UNAVAILABLE = 3

    class ActivityStats(proto.Message):
        r"""-

        Attributes:
            active_users_count (int):
                -
            active_jobs_count (int):
                -
        """

        active_users_count = proto.Field(proto.INT64, number=1)
        active_jobs_count = proto.Field(proto.INT64, number=2)

    name = proto.Field(proto.STRING, number=1)
    health = proto.Field(proto.ENUM, number=3, enum=Health)
    expected_down_time = proto.Field(proto.MESSAGE, number=7, message=timestamp_pb2.Timestamp)
    expected_recovery_time = proto.Field(proto.MESSAGE, number=4, message=timestamp_pb2.Timestamp)
    supported_languages = proto.RepeatedField(proto.STRING, number=5)
    device_spec = proto.Field(proto.MESSAGE, number=6, message=any_pb2.Any)
    schedule_horizon = proto.Field(proto.MESSAGE, number=8, message=duration_pb2.Duration)
    schedule_frozen_period = proto.Field(proto.MESSAGE, number=9, message=duration_pb2.Duration)
    current_calibration = proto.Field(proto.STRING, number=10)
    active_time_slot = proto.Field(proto.MESSAGE, number=11, message='QuantumTimeSlot')
    activity_stats = proto.Field(proto.MESSAGE, number=12, message=ActivityStats)
    default_device_config_key = proto.Field(proto.MESSAGE, number=13, message='DeviceConfigKey')


class QuantumCalibration(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        timestamp (google.protobuf.timestamp_pb2.Timestamp):
            -
        data (google.protobuf.any_pb2.Any):
            -
    """

    name = proto.Field(proto.STRING, number=1)
    timestamp = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    data = proto.Field(proto.MESSAGE, number=3, message=any_pb2.Any)


class QuantumReservationGrant(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        processor_names (Sequence[str]):
            -
        effective_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        granted_duration (google.protobuf.duration_pb2.Duration):
            -
        available_duration (google.protobuf.duration_pb2.Duration):
            -
        budgets (Sequence[google.cloud.quantum_v1alpha1.types.QuantumReservationGrant.Budget]):
            -
    """

    class Budget(proto.Message):
        r"""-

        Attributes:
            project_id (str):
                -
            granted_duration (google.protobuf.duration_pb2.Duration):
                -
            available_duration (google.protobuf.duration_pb2.Duration):
                -
        """

        project_id = proto.Field(proto.STRING, number=1)
        granted_duration = proto.Field(proto.MESSAGE, number=2, message=duration_pb2.Duration)
        available_duration = proto.Field(proto.MESSAGE, number=3, message=duration_pb2.Duration)

    name = proto.Field(proto.STRING, number=1)
    processor_names = proto.RepeatedField(proto.STRING, number=2)
    effective_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    expire_time = proto.Field(proto.MESSAGE, number=4, message=timestamp_pb2.Timestamp)
    granted_duration = proto.Field(proto.MESSAGE, number=5, message=duration_pb2.Duration)
    available_duration = proto.Field(proto.MESSAGE, number=6, message=duration_pb2.Duration)
    budgets = proto.RepeatedField(proto.MESSAGE, number=7, message=Budget)


class QuantumReservationBudget(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        processor_names (Sequence[str]):
            -
        effective_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        expire_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        granted_duration (google.protobuf.duration_pb2.Duration):
            -
        available_duration (google.protobuf.duration_pb2.Duration):
            -
    """

    name = proto.Field(proto.STRING, number=1)
    processor_names = proto.RepeatedField(proto.STRING, number=2)
    effective_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    expire_time = proto.Field(proto.MESSAGE, number=4, message=timestamp_pb2.Timestamp)
    granted_duration = proto.Field(proto.MESSAGE, number=5, message=duration_pb2.Duration)
    available_duration = proto.Field(proto.MESSAGE, number=6, message=duration_pb2.Duration)


class QuantumTimeSlot(proto.Message):
    r"""-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        processor_name (str):
            -
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        time_slot_type (google.cloud.quantum_v1alpha1.types.QuantumTimeSlot.TimeSlotType):
            -
        reservation_config (google.cloud.quantum_v1alpha1.types.QuantumTimeSlot.ReservationConfig):
            -

            This field is a member of `oneof`_ ``type_config``.
        maintenance_config (google.cloud.quantum_v1alpha1.types.QuantumTimeSlot.MaintenanceConfig):
            -

            This field is a member of `oneof`_ ``type_config``.
    """

    class TimeSlotType(proto.Enum):
        r"""-"""

        TIME_SLOT_TYPE_UNSPECIFIED = 0
        MAINTENANCE = 1
        OPEN_SWIM = 2
        RESERVATION = 3
        UNALLOCATED = 4

    class ReservationConfig(proto.Message):
        r"""-

        Attributes:
            reservation (str):
                -
            project_id (str):
                -
            whitelisted_users (Sequence[str]):
                -
        """

        reservation = proto.Field(proto.STRING, number=3)
        project_id = proto.Field(proto.STRING, number=1)
        whitelisted_users = proto.RepeatedField(proto.STRING, number=2)

    class MaintenanceConfig(proto.Message):
        r"""-

        Attributes:
            title (str):
                -
            description (str):
                -
        """

        title = proto.Field(proto.STRING, number=1)
        description = proto.Field(proto.STRING, number=2)

    processor_name = proto.Field(proto.STRING, number=1)
    start_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    end_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    time_slot_type = proto.Field(proto.ENUM, number=5, enum=TimeSlotType)
    reservation_config = proto.Field(
        proto.MESSAGE, number=6, oneof='type_config', message=ReservationConfig
    )
    maintenance_config = proto.Field(
        proto.MESSAGE, number=7, oneof='type_config', message=MaintenanceConfig
    )


class QuantumReservation(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        start_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        end_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        cancelled_time (google.protobuf.timestamp_pb2.Timestamp):
            -
        whitelisted_users (Sequence[str]):
            -
    """

    name = proto.Field(proto.STRING, number=1)
    start_time = proto.Field(proto.MESSAGE, number=2, message=timestamp_pb2.Timestamp)
    end_time = proto.Field(proto.MESSAGE, number=3, message=timestamp_pb2.Timestamp)
    cancelled_time = proto.Field(proto.MESSAGE, number=4, message=timestamp_pb2.Timestamp)
    whitelisted_users = proto.RepeatedField(proto.STRING, number=5)


__all__ = tuple(sorted(__protobuf__.manifest))
