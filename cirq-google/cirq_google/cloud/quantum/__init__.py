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
from google.cloud.quantum import gapic_version as package_version

__version__ = package_version.__version__


from google.cloud.quantum_v1alpha1.services.quantum_engine_service.client import QuantumEngineServiceClient
from google.cloud.quantum_v1alpha1.services.quantum_engine_service.async_client import QuantumEngineServiceAsyncClient

from google.cloud.quantum_v1alpha1.types.engine import CancelQuantumJobRequest
from google.cloud.quantum_v1alpha1.types.engine import CancelQuantumReservationRequest
from google.cloud.quantum_v1alpha1.types.engine import CreateQuantumJobRequest
from google.cloud.quantum_v1alpha1.types.engine import CreateQuantumProgramAndJobRequest
from google.cloud.quantum_v1alpha1.types.engine import CreateQuantumProgramRequest
from google.cloud.quantum_v1alpha1.types.engine import CreateQuantumReservationRequest
from google.cloud.quantum_v1alpha1.types.engine import DeleteQuantumJobRequest
from google.cloud.quantum_v1alpha1.types.engine import DeleteQuantumProgramRequest
from google.cloud.quantum_v1alpha1.types.engine import DeleteQuantumReservationRequest
from google.cloud.quantum_v1alpha1.types.engine import GetQuantumCalibrationRequest
from google.cloud.quantum_v1alpha1.types.engine import GetQuantumJobRequest
from google.cloud.quantum_v1alpha1.types.engine import GetQuantumProcessorConfigRequest
from google.cloud.quantum_v1alpha1.types.engine import GetQuantumProcessorRequest
from google.cloud.quantum_v1alpha1.types.engine import GetQuantumProgramRequest
from google.cloud.quantum_v1alpha1.types.engine import GetQuantumReservationRequest
from google.cloud.quantum_v1alpha1.types.engine import GetQuantumResultRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumCalibrationsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumCalibrationsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumJobEventsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumJobEventsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumJobsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumJobsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumProcessorConfigsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumProcessorConfigsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumProcessorsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumProcessorsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumProgramsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumProgramsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumReservationBudgetsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumReservationBudgetsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumReservationGrantsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumReservationGrantsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumReservationsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumReservationsResponse
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumTimeSlotsRequest
from google.cloud.quantum_v1alpha1.types.engine import ListQuantumTimeSlotsResponse
from google.cloud.quantum_v1alpha1.types.engine import QuantumRunStreamRequest
from google.cloud.quantum_v1alpha1.types.engine import QuantumRunStreamResponse
from google.cloud.quantum_v1alpha1.types.engine import ReallocateQuantumReservationGrantRequest
from google.cloud.quantum_v1alpha1.types.engine import StreamError
from google.cloud.quantum_v1alpha1.types.engine import UpdateQuantumJobRequest
from google.cloud.quantum_v1alpha1.types.engine import UpdateQuantumProgramRequest
from google.cloud.quantum_v1alpha1.types.engine import UpdateQuantumReservationRequest
from google.cloud.quantum_v1alpha1.types.quantum import DeviceConfigKey
from google.cloud.quantum_v1alpha1.types.quantum import DeviceConfigSelector
from google.cloud.quantum_v1alpha1.types.quantum import ExecutionStatus
from google.cloud.quantum_v1alpha1.types.quantum import GcsLocation
from google.cloud.quantum_v1alpha1.types.quantum import InlineData
from google.cloud.quantum_v1alpha1.types.quantum import OutputConfig
from google.cloud.quantum_v1alpha1.types.quantum import QuantumCalibration
from google.cloud.quantum_v1alpha1.types.quantum import QuantumJob
from google.cloud.quantum_v1alpha1.types.quantum import QuantumJobEvent
from google.cloud.quantum_v1alpha1.types.quantum import QuantumProcessor
from google.cloud.quantum_v1alpha1.types.quantum import QuantumProcessorConfig
from google.cloud.quantum_v1alpha1.types.quantum import QuantumProgram
from google.cloud.quantum_v1alpha1.types.quantum import QuantumReservation
from google.cloud.quantum_v1alpha1.types.quantum import QuantumReservationBudget
from google.cloud.quantum_v1alpha1.types.quantum import QuantumReservationGrant
from google.cloud.quantum_v1alpha1.types.quantum import QuantumResult
from google.cloud.quantum_v1alpha1.types.quantum import QuantumTimeSlot
from google.cloud.quantum_v1alpha1.types.quantum import SchedulingConfig

__all__ = ('QuantumEngineServiceClient',
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
    'GetQuantumProcessorConfigRequest',
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
    'ListQuantumProcessorConfigsRequest',
    'ListQuantumProcessorConfigsResponse',
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
    'DeviceConfigKey',
    'DeviceConfigSelector',
    'ExecutionStatus',
    'GcsLocation',
    'InlineData',
    'OutputConfig',
    'QuantumCalibration',
    'QuantumJob',
    'QuantumJobEvent',
    'QuantumProcessor',
    'QuantumProcessorConfig',
    'QuantumProgram',
    'QuantumReservation',
    'QuantumReservationBudget',
    'QuantumReservationGrant',
    'QuantumResult',
    'QuantumTimeSlot',
    'SchedulingConfig',
)
