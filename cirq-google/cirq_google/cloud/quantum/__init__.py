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

from cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.client import (
    QuantumEngineServiceClient as QuantumEngineServiceClient,
)
from cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.async_client import (
    QuantumEngineServiceAsyncClient as QuantumEngineServiceAsyncClient,
)

from cirq_google.cloud.quantum_v1alpha1.types.engine import (
    CancelQuantumJobRequest as CancelQuantumJobRequest,
    CancelQuantumReservationRequest as CancelQuantumReservationRequest,
    CreateQuantumJobRequest as CreateQuantumJobRequest,
    CreateQuantumProgramAndJobRequest as CreateQuantumProgramAndJobRequest,
    CreateQuantumProgramRequest as CreateQuantumProgramRequest,
    CreateQuantumReservationRequest as CreateQuantumReservationRequest,
    DeleteQuantumJobRequest as DeleteQuantumJobRequest,
    DeleteQuantumProgramRequest as DeleteQuantumProgramRequest,
    DeleteQuantumReservationRequest as DeleteQuantumReservationRequest,
    GetQuantumCalibrationRequest as GetQuantumCalibrationRequest,
    GetQuantumJobRequest as GetQuantumJobRequest,
    GetQuantumProcessorRequest as GetQuantumProcessorRequest,
    GetQuantumProgramRequest as GetQuantumProgramRequest,
    GetQuantumReservationRequest as GetQuantumReservationRequest,
    GetQuantumResultRequest as GetQuantumResultRequest,
    ListQuantumCalibrationsRequest as ListQuantumCalibrationsRequest,
    ListQuantumCalibrationsResponse as ListQuantumCalibrationsResponse,
    ListQuantumJobEventsRequest as ListQuantumJobEventsRequest,
    ListQuantumJobEventsResponse as ListQuantumJobEventsResponse,
    ListQuantumJobsRequest as ListQuantumJobsRequest,
    ListQuantumJobsResponse as ListQuantumJobsResponse,
    ListQuantumProcessorsRequest as ListQuantumProcessorsRequest,
    ListQuantumProcessorsResponse as ListQuantumProcessorsResponse,
    ListQuantumProgramsRequest as ListQuantumProgramsRequest,
    ListQuantumProgramsResponse as ListQuantumProgramsResponse,
    ListQuantumReservationBudgetsRequest as ListQuantumReservationBudgetsRequest,
    ListQuantumReservationBudgetsResponse as ListQuantumReservationBudgetsResponse,
    ListQuantumReservationGrantsRequest as ListQuantumReservationGrantsRequest,
    ListQuantumReservationGrantsResponse as ListQuantumReservationGrantsResponse,
    ListQuantumReservationsRequest as ListQuantumReservationsRequest,
    ListQuantumReservationsResponse as ListQuantumReservationsResponse,
    ListQuantumTimeSlotsRequest as ListQuantumTimeSlotsRequest,
    ListQuantumTimeSlotsResponse as ListQuantumTimeSlotsResponse,
    QuantumRunStreamRequest as QuantumRunStreamRequest,
    QuantumRunStreamResponse as QuantumRunStreamResponse,
    ReallocateQuantumReservationGrantRequest as ReallocateQuantumReservationGrantRequest,
    StreamError as StreamError,
    UpdateQuantumJobRequest as UpdateQuantumJobRequest,
    UpdateQuantumProgramRequest as UpdateQuantumProgramRequest,
    UpdateQuantumReservationRequest as UpdateQuantumReservationRequest,
)

from cirq_google.cloud.quantum_v1alpha1.types.quantum import (
    DeviceConfigSelector as DeviceConfigSelector,
    ExecutionStatus as ExecutionStatus,
    GcsLocation as GcsLocation,
    InlineData as InlineData,
    OutputConfig as OutputConfig,
    QuantumCalibration as QuantumCalibration,
    QuantumJob as QuantumJob,
    QuantumJobEvent as QuantumJobEvent,
    QuantumProcessor as QuantumProcessor,
    QuantumProgram as QuantumProgram,
    QuantumReservation as QuantumReservation,
    QuantumReservationBudget as QuantumReservationBudget,
    QuantumReservationGrant as QuantumReservationGrant,
    QuantumResult as QuantumResult,
    QuantumTimeSlot as QuantumTimeSlot,
    SchedulingConfig as SchedulingConfig,
    DeviceConfigKey as DeviceConfigKey,
)


__all__ = (
    'QuantumEngineServiceClient',
    'QuantumEngineServiceAsyncClient',
    'CancelQuantumJobRequest',
    'CancelQuantumReservationRequest',
    'CreateQuantumJobRequest',
    'CreateQuantumProgramAndJobRequest',
    'CreateQuantumProgramRequest',
    'CreateQuantumReservationRequest',
    'DeleteQuantumJobRequest',
    'DeleteQuantumProgramRequest',
    'DeleteQuantumReservationRequest',
    'GetQuantumCalibrationRequest',
    'GetQuantumJobRequest',
    'GetQuantumProcessorRequest',
    'GetQuantumProgramRequest',
    'GetQuantumReservationRequest',
    'GetQuantumResultRequest',
    'ListQuantumCalibrationsRequest',
    'ListQuantumCalibrationsResponse',
    'ListQuantumJobEventsRequest',
    'ListQuantumJobEventsResponse',
    'ListQuantumJobsRequest',
    'ListQuantumJobsResponse',
    'ListQuantumProcessorsRequest',
    'ListQuantumProcessorsResponse',
    'ListQuantumProgramsRequest',
    'ListQuantumProgramsResponse',
    'ListQuantumReservationBudgetsRequest',
    'ListQuantumReservationBudgetsResponse',
    'ListQuantumReservationGrantsRequest',
    'ListQuantumReservationGrantsResponse',
    'ListQuantumReservationsRequest',
    'ListQuantumReservationsResponse',
    'ListQuantumTimeSlotsRequest',
    'ListQuantumTimeSlotsResponse',
    'QuantumRunStreamRequest',
    'QuantumRunStreamResponse',
    'ReallocateQuantumReservationGrantRequest',
    'StreamError',
    'UpdateQuantumJobRequest',
    'UpdateQuantumProgramRequest',
    'UpdateQuantumReservationRequest',
    'DeviceConfigSelector',
    'ExecutionStatus',
    'GcsLocation',
    'InlineData',
    'OutputConfig',
    'QuantumCalibration',
    'QuantumJob',
    'QuantumJobEvent',
    'QuantumProcessor',
    'QuantumProgram',
    'QuantumReservation',
    'QuantumReservationBudget',
    'QuantumReservationGrant',
    'QuantumResult',
    'QuantumTimeSlot',
    'SchedulingConfig',
    'DeviceConfigKey',
)
