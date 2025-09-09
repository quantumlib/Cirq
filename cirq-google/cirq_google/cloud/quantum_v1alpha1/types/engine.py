# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
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
from __future__ import annotations

from typing import MutableMapping, MutableSequence

import proto  # type: ignore
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore

from cirq_google.cloud.quantum_v1alpha1.types import quantum

__protobuf__ = proto.module(
    package='google.cloud.quantum.v1alpha1',
    manifest={
        'CreateQuantumJobRequest',
        'GetQuantumJobRequest',
        'ListQuantumJobsRequest',
        'ListQuantumJobsResponse',
        'DeleteQuantumJobRequest',
        'UpdateQuantumJobRequest',
        'CancelQuantumJobRequest',
        'ListQuantumJobEventsRequest',
        'ListQuantumJobEventsResponse',
        'GetQuantumResultRequest',
        'CreateQuantumProgramRequest',
        'GetQuantumProgramRequest',
        'ListQuantumProgramsRequest',
        'ListQuantumProgramsResponse',
        'DeleteQuantumProgramRequest',
        'UpdateQuantumProgramRequest',
        'ListQuantumProcessorsRequest',
        'ListQuantumProcessorsResponse',
        'GetQuantumProcessorRequest',
        'GetQuantumProcessorConfigRequest',
        'ListQuantumProcessorConfigsRequest',
        'ListQuantumProcessorConfigsResponse',
        'ListQuantumCalibrationsRequest',
        'ListQuantumCalibrationsResponse',
        'GetQuantumCalibrationRequest',
        'CreateQuantumReservationRequest',
        'CancelQuantumReservationRequest',
        'DeleteQuantumReservationRequest',
        'GetQuantumReservationRequest',
        'ListQuantumReservationsRequest',
        'ListQuantumReservationsResponse',
        'UpdateQuantumReservationRequest',
        'QuantumRunStreamRequest',
        'CreateQuantumProgramAndJobRequest',
        'QuantumRunStreamResponse',
        'StreamError',
        'ListQuantumReservationGrantsRequest',
        'ListQuantumReservationGrantsResponse',
        'ReallocateQuantumReservationGrantRequest',
        'ListQuantumReservationBudgetsRequest',
        'ListQuantumReservationBudgetsResponse',
        'ListQuantumTimeSlotsRequest',
        'ListQuantumTimeSlotsResponse',
    },
)


class CreateQuantumJobRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        quantum_job (cirq_google.cloud.quantum_v1alpha1.types.QuantumJob):
            -
        overwrite_existing_run_context (bool):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    quantum_job: quantum.QuantumJob = proto.Field(
        proto.MESSAGE, number=2, message=quantum.QuantumJob
    )
    overwrite_existing_run_context: bool = proto.Field(proto.BOOL, number=3)


class GetQuantumJobRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        return_run_context (bool):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)
    return_run_context: bool = proto.Field(proto.BOOL, number=2)


class ListQuantumJobsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)


class ListQuantumJobsResponse(proto.Message):
    r"""-

    Attributes:
        jobs (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumJob]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    jobs: MutableSequence[quantum.QuantumJob] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumJob
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class DeleteQuantumJobRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)


class UpdateQuantumJobRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        quantum_job (cirq_google.cloud.quantum_v1alpha1.types.QuantumJob):
            -
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)
    quantum_job: quantum.QuantumJob = proto.Field(
        proto.MESSAGE, number=2, message=quantum.QuantumJob
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask
    )


class CancelQuantumJobRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)


class ListQuantumJobEventsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)


class ListQuantumJobEventsResponse(proto.Message):
    r"""-

    Attributes:
        events (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumJobEvent]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    events: MutableSequence[quantum.QuantumJobEvent] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumJobEvent
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class GetQuantumResultRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)


class CreateQuantumProgramRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        quantum_program (cirq_google.cloud.quantum_v1alpha1.types.QuantumProgram):
            -
        overwrite_existing_source_code (bool):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    quantum_program: quantum.QuantumProgram = proto.Field(
        proto.MESSAGE, number=2, message=quantum.QuantumProgram
    )
    overwrite_existing_source_code: bool = proto.Field(proto.BOOL, number=3)


class GetQuantumProgramRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        return_code (bool):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)
    return_code: bool = proto.Field(proto.BOOL, number=2)


class ListQuantumProgramsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)


class ListQuantumProgramsResponse(proto.Message):
    r"""-

    Attributes:
        programs (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumProgram]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    programs: MutableSequence[quantum.QuantumProgram] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumProgram
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class DeleteQuantumProgramRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        delete_jobs (bool):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)
    delete_jobs: bool = proto.Field(proto.BOOL, number=2)


class UpdateQuantumProgramRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        quantum_program (cirq_google.cloud.quantum_v1alpha1.types.QuantumProgram):
            -
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)
    quantum_program: quantum.QuantumProgram = proto.Field(
        proto.MESSAGE, number=2, message=quantum.QuantumProgram
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask
    )


class ListQuantumProcessorsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)


class ListQuantumProcessorsResponse(proto.Message):
    r"""-

    Attributes:
        processors (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumProcessor]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    processors: MutableSequence[quantum.QuantumProcessor] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumProcessor
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class GetQuantumProcessorRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)


class GetQuantumProcessorConfigRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            Required. -
    """

    name: str = proto.Field(proto.STRING, number=1)


class ListQuantumProcessorConfigsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            Required. -
        page_size (int):
            Optional. -
        page_token (str):
            Optional. -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)


class ListQuantumProcessorConfigsResponse(proto.Message):
    r"""-

    Attributes:
        quantum_processor_configs (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumProcessorConfig]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    quantum_processor_configs: MutableSequence[quantum.QuantumProcessorConfig] = (
        proto.RepeatedField(proto.MESSAGE, number=1, message=quantum.QuantumProcessorConfig)
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class ListQuantumCalibrationsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        view (cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsRequest.QuantumCalibrationView):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    class QuantumCalibrationView(proto.Enum):
        r"""-

        Values:
            QUANTUM_CALIBRATION_VIEW_UNSPECIFIED (0):
                -
            BASIC (1):
                -
            FULL (2):
                -
        """

        QUANTUM_CALIBRATION_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2

    parent: str = proto.Field(proto.STRING, number=1)
    view: QuantumCalibrationView = proto.Field(proto.ENUM, number=5, enum=QuantumCalibrationView)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)


class ListQuantumCalibrationsResponse(proto.Message):
    r"""-

    Attributes:
        calibrations (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumCalibration]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    calibrations: MutableSequence[quantum.QuantumCalibration] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumCalibration
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class GetQuantumCalibrationRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)


class CreateQuantumReservationRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        quantum_reservation (cirq_google.cloud.quantum_v1alpha1.types.QuantumReservation):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    quantum_reservation: quantum.QuantumReservation = proto.Field(
        proto.MESSAGE, number=2, message=quantum.QuantumReservation
    )


class CancelQuantumReservationRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)


class DeleteQuantumReservationRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)


class GetQuantumReservationRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)


class ListQuantumReservationsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)


class ListQuantumReservationsResponse(proto.Message):
    r"""-

    Attributes:
        reservations (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumReservation]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    reservations: MutableSequence[quantum.QuantumReservation] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumReservation
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class UpdateQuantumReservationRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        quantum_reservation (cirq_google.cloud.quantum_v1alpha1.types.QuantumReservation):
            -
        update_mask (google.protobuf.field_mask_pb2.FieldMask):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)
    quantum_reservation: quantum.QuantumReservation = proto.Field(
        proto.MESSAGE, number=2, message=quantum.QuantumReservation
    )
    update_mask: field_mask_pb2.FieldMask = proto.Field(
        proto.MESSAGE, number=3, message=field_mask_pb2.FieldMask
    )


class QuantumRunStreamRequest(proto.Message):
    r"""-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        message_id (str):
            -
        parent (str):
            -
        create_quantum_program_and_job (cirq_google.cloud.quantum_v1alpha1.types.CreateQuantumProgramAndJobRequest):
            -

            This field is a member of `oneof`_ ``request``.
        create_quantum_job (cirq_google.cloud.quantum_v1alpha1.types.CreateQuantumJobRequest):
            -

            This field is a member of `oneof`_ ``request``.
        get_quantum_result (cirq_google.cloud.quantum_v1alpha1.types.GetQuantumResultRequest):
            -

            This field is a member of `oneof`_ ``request``.
    """

    message_id: str = proto.Field(proto.STRING, number=1)
    parent: str = proto.Field(proto.STRING, number=2)
    create_quantum_program_and_job: 'CreateQuantumProgramAndJobRequest' = proto.Field(
        proto.MESSAGE, number=3, oneof='request', message='CreateQuantumProgramAndJobRequest'
    )
    create_quantum_job: 'CreateQuantumJobRequest' = proto.Field(
        proto.MESSAGE, number=4, oneof='request', message='CreateQuantumJobRequest'
    )
    get_quantum_result: 'GetQuantumResultRequest' = proto.Field(
        proto.MESSAGE, number=5, oneof='request', message='GetQuantumResultRequest'
    )


class CreateQuantumProgramAndJobRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        quantum_program (cirq_google.cloud.quantum_v1alpha1.types.QuantumProgram):
            -
        quantum_job (cirq_google.cloud.quantum_v1alpha1.types.QuantumJob):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    quantum_program: quantum.QuantumProgram = proto.Field(
        proto.MESSAGE, number=2, message=quantum.QuantumProgram
    )
    quantum_job: quantum.QuantumJob = proto.Field(
        proto.MESSAGE, number=3, message=quantum.QuantumJob
    )


class QuantumRunStreamResponse(proto.Message):
    r"""-

    This message has `oneof`_ fields (mutually exclusive fields).
    For each oneof, at most one member field can be set at the same time.
    Setting any member of the oneof automatically clears all other
    members.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        message_id (str):
            -
        error (cirq_google.cloud.quantum_v1alpha1.types.StreamError):
            -

            This field is a member of `oneof`_ ``response``.
        job (cirq_google.cloud.quantum_v1alpha1.types.QuantumJob):
            -

            This field is a member of `oneof`_ ``response``.
        result (cirq_google.cloud.quantum_v1alpha1.types.QuantumResult):
            -

            This field is a member of `oneof`_ ``response``.
    """

    message_id: str = proto.Field(proto.STRING, number=1)
    error: 'StreamError' = proto.Field(
        proto.MESSAGE, number=2, oneof='response', message='StreamError'
    )
    job: quantum.QuantumJob = proto.Field(
        proto.MESSAGE, number=3, oneof='response', message=quantum.QuantumJob
    )
    result: quantum.QuantumResult = proto.Field(
        proto.MESSAGE, number=4, oneof='response', message=quantum.QuantumResult
    )


class StreamError(proto.Message):
    r"""-

    Attributes:
        code (cirq_google.cloud.quantum_v1alpha1.types.StreamError.Code):
            -
        message (str):
            -
    """

    class Code(proto.Enum):
        r"""-

        Values:
            CODE_UNSPECIFIED (0):
                -
            INTERNAL (1):
                -
            INVALID_ARGUMENT (2):
                -
            PERMISSION_DENIED (3):
                -
            PROGRAM_ALREADY_EXISTS (4):
                -
            JOB_ALREADY_EXISTS (5):
                -
            PROGRAM_DOES_NOT_EXIST (6):
                -
            JOB_DOES_NOT_EXIST (7):
                -
            PROCESSOR_DOES_NOT_EXIST (8):
                -
            INVALID_PROCESSOR_FOR_JOB (9):
                -
        """

        CODE_UNSPECIFIED = 0
        INTERNAL = 1
        INVALID_ARGUMENT = 2
        PERMISSION_DENIED = 3
        PROGRAM_ALREADY_EXISTS = 4
        JOB_ALREADY_EXISTS = 5
        PROGRAM_DOES_NOT_EXIST = 6
        JOB_DOES_NOT_EXIST = 7
        PROCESSOR_DOES_NOT_EXIST = 8
        INVALID_PROCESSOR_FOR_JOB = 9

    code: Code = proto.Field(proto.ENUM, number=1, enum=Code)
    message: str = proto.Field(proto.STRING, number=2)


class ListQuantumReservationGrantsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)


class ListQuantumReservationGrantsResponse(proto.Message):
    r"""-

    Attributes:
        reservation_grants (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumReservationGrant]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    reservation_grants: MutableSequence[quantum.QuantumReservationGrant] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumReservationGrant
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class ReallocateQuantumReservationGrantRequest(proto.Message):
    r"""-

    Attributes:
        name (str):
            -
        source_project_id (str):
            -
        target_project_id (str):
            -
        duration (google.protobuf.duration_pb2.Duration):
            -
    """

    name: str = proto.Field(proto.STRING, number=1)
    source_project_id: str = proto.Field(proto.STRING, number=2)
    target_project_id: str = proto.Field(proto.STRING, number=3)
    duration: duration_pb2.Duration = proto.Field(
        proto.MESSAGE, number=4, message=duration_pb2.Duration
    )


class ListQuantumReservationBudgetsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)


class ListQuantumReservationBudgetsResponse(proto.Message):
    r"""-

    Attributes:
        reservation_budgets (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumReservationBudget]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    reservation_budgets: MutableSequence[quantum.QuantumReservationBudget] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumReservationBudget
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


class ListQuantumTimeSlotsRequest(proto.Message):
    r"""-

    Attributes:
        parent (str):
            -
        page_size (int):
            -
        page_token (str):
            -
        filter (str):
            -
    """

    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    filter: str = proto.Field(proto.STRING, number=4)


class ListQuantumTimeSlotsResponse(proto.Message):
    r"""-

    Attributes:
        time_slots (MutableSequence[cirq_google.cloud.quantum_v1alpha1.types.QuantumTimeSlot]):
            -
        next_page_token (str):
            -
    """

    @property
    def raw_page(self):
        return self

    time_slots: MutableSequence[quantum.QuantumTimeSlot] = proto.RepeatedField(
        proto.MESSAGE, number=1, message=quantum.QuantumTimeSlot
    )
    next_page_token: str = proto.Field(proto.STRING, number=2)


__all__ = tuple(sorted(__protobuf__.manifest))
