# -*- coding: utf-8 -*-
# Copyright 2024 Google LLC
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

from google.auth.transport.requests import AuthorizedSession  # type: ignore
import json  # type: ignore
import grpc  # type: ignore
from google.auth.transport.grpc import SslCredentials  # type: ignore
from google.auth import credentials as ga_credentials  # type: ignore
from google.api_core import exceptions as core_exceptions
from google.api_core import retry as retries
from google.api_core import rest_helpers
from google.api_core import rest_streaming
from google.api_core import path_template
from google.api_core import gapic_v1

from google.protobuf import json_format
from requests import __version__ as requests_version
import dataclasses
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore


from google.cloud.quantum_v1alpha1.types import engine
from google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import empty_pb2  # type: ignore

from .base import QuantumEngineServiceTransport, DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=requests_version,
)


class QuantumEngineServiceRestInterceptor:
    """Interceptor for QuantumEngineService.

    Interceptors are used to manipulate requests, request metadata, and responses
    in arbitrary ways.
    Example use cases include:
    * Logging
    * Verifying requests according to service or custom semantics
    * Stripping extraneous information from responses

    These use cases and more can be enabled by injecting an
    instance of a custom subclass when constructing the QuantumEngineServiceRestTransport.

    .. code-block:: python
        class MyCustomQuantumEngineServiceInterceptor(QuantumEngineServiceRestInterceptor):
            def pre_cancel_quantum_job(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_cancel_quantum_reservation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_cancel_quantum_reservation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_quantum_job(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_quantum_job(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_quantum_program(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_quantum_program(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_create_quantum_reservation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_create_quantum_reservation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_delete_quantum_job(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_quantum_program(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_delete_quantum_reservation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def pre_get_quantum_calibration(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_quantum_calibration(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_quantum_job(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_quantum_job(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_quantum_processor(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_quantum_processor(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_quantum_program(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_quantum_program(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_quantum_reservation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_quantum_reservation(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_get_quantum_result(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_quantum_result(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_calibrations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_calibrations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_job_events(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_job_events(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_jobs(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_jobs(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_processors(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_processors(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_programs(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_programs(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_reservation_budgets(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_reservation_budgets(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_reservation_grants(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_reservation_grants(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_reservations(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_reservations(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_list_quantum_time_slots(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_time_slots(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_reallocate_quantum_reservation_grant(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_reallocate_quantum_reservation_grant(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_quantum_job(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_quantum_job(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_quantum_program(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_quantum_program(self, response):
                logging.log(f"Received response: {response}")
                return response

            def pre_update_quantum_reservation(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_update_quantum_reservation(self, response):
                logging.log(f"Received response: {response}")
                return response

        transport = QuantumEngineServiceRestTransport(interceptor=MyCustomQuantumEngineServiceInterceptor())
        client = QuantumEngineServiceClient(transport=transport)


    """
    def pre_cancel_quantum_job(self, request: engine.CancelQuantumJobRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.CancelQuantumJobRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for cancel_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def pre_cancel_quantum_reservation(self, request: engine.CancelQuantumReservationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.CancelQuantumReservationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for cancel_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_cancel_quantum_reservation(self, response: quantum.QuantumReservation) -> quantum.QuantumReservation:
        """Post-rpc interceptor for cancel_quantum_reservation

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_create_quantum_job(self, request: engine.CreateQuantumJobRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.CreateQuantumJobRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_create_quantum_job(self, response: quantum.QuantumJob) -> quantum.QuantumJob:
        """Post-rpc interceptor for create_quantum_job

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_create_quantum_program(self, request: engine.CreateQuantumProgramRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.CreateQuantumProgramRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_quantum_program

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_create_quantum_program(self, response: quantum.QuantumProgram) -> quantum.QuantumProgram:
        """Post-rpc interceptor for create_quantum_program

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_create_quantum_reservation(self, request: engine.CreateQuantumReservationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.CreateQuantumReservationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for create_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_create_quantum_reservation(self, response: quantum.QuantumReservation) -> quantum.QuantumReservation:
        """Post-rpc interceptor for create_quantum_reservation

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_delete_quantum_job(self, request: engine.DeleteQuantumJobRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.DeleteQuantumJobRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def pre_delete_quantum_program(self, request: engine.DeleteQuantumProgramRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.DeleteQuantumProgramRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_quantum_program

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def pre_delete_quantum_reservation(self, request: engine.DeleteQuantumReservationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.DeleteQuantumReservationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for delete_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def pre_get_quantum_calibration(self, request: engine.GetQuantumCalibrationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.GetQuantumCalibrationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_quantum_calibration

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_calibration(self, response: quantum.QuantumCalibration) -> quantum.QuantumCalibration:
        """Post-rpc interceptor for get_quantum_calibration

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_get_quantum_job(self, request: engine.GetQuantumJobRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.GetQuantumJobRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_job(self, response: quantum.QuantumJob) -> quantum.QuantumJob:
        """Post-rpc interceptor for get_quantum_job

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_get_quantum_processor(self, request: engine.GetQuantumProcessorRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.GetQuantumProcessorRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_quantum_processor

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_processor(self, response: quantum.QuantumProcessor) -> quantum.QuantumProcessor:
        """Post-rpc interceptor for get_quantum_processor

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_get_quantum_program(self, request: engine.GetQuantumProgramRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.GetQuantumProgramRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_quantum_program

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_program(self, response: quantum.QuantumProgram) -> quantum.QuantumProgram:
        """Post-rpc interceptor for get_quantum_program

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_get_quantum_reservation(self, request: engine.GetQuantumReservationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.GetQuantumReservationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_reservation(self, response: quantum.QuantumReservation) -> quantum.QuantumReservation:
        """Post-rpc interceptor for get_quantum_reservation

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_get_quantum_result(self, request: engine.GetQuantumResultRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.GetQuantumResultRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for get_quantum_result

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_result(self, response: quantum.QuantumResult) -> quantum.QuantumResult:
        """Post-rpc interceptor for get_quantum_result

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_calibrations(self, request: engine.ListQuantumCalibrationsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumCalibrationsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_calibrations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_calibrations(self, response: engine.ListQuantumCalibrationsResponse) -> engine.ListQuantumCalibrationsResponse:
        """Post-rpc interceptor for list_quantum_calibrations

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_job_events(self, request: engine.ListQuantumJobEventsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumJobEventsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_job_events

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_job_events(self, response: engine.ListQuantumJobEventsResponse) -> engine.ListQuantumJobEventsResponse:
        """Post-rpc interceptor for list_quantum_job_events

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_jobs(self, request: engine.ListQuantumJobsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumJobsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_jobs

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_jobs(self, response: engine.ListQuantumJobsResponse) -> engine.ListQuantumJobsResponse:
        """Post-rpc interceptor for list_quantum_jobs

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_processors(self, request: engine.ListQuantumProcessorsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumProcessorsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_processors

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_processors(self, response: engine.ListQuantumProcessorsResponse) -> engine.ListQuantumProcessorsResponse:
        """Post-rpc interceptor for list_quantum_processors

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_programs(self, request: engine.ListQuantumProgramsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumProgramsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_programs

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_programs(self, response: engine.ListQuantumProgramsResponse) -> engine.ListQuantumProgramsResponse:
        """Post-rpc interceptor for list_quantum_programs

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_reservation_budgets(self, request: engine.ListQuantumReservationBudgetsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumReservationBudgetsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_reservation_budgets

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_reservation_budgets(self, response: engine.ListQuantumReservationBudgetsResponse) -> engine.ListQuantumReservationBudgetsResponse:
        """Post-rpc interceptor for list_quantum_reservation_budgets

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_reservation_grants(self, request: engine.ListQuantumReservationGrantsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumReservationGrantsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_reservation_grants

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_reservation_grants(self, response: engine.ListQuantumReservationGrantsResponse) -> engine.ListQuantumReservationGrantsResponse:
        """Post-rpc interceptor for list_quantum_reservation_grants

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_reservations(self, request: engine.ListQuantumReservationsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumReservationsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_reservations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_reservations(self, response: engine.ListQuantumReservationsResponse) -> engine.ListQuantumReservationsResponse:
        """Post-rpc interceptor for list_quantum_reservations

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_list_quantum_time_slots(self, request: engine.ListQuantumTimeSlotsRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ListQuantumTimeSlotsRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for list_quantum_time_slots

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_time_slots(self, response: engine.ListQuantumTimeSlotsResponse) -> engine.ListQuantumTimeSlotsResponse:
        """Post-rpc interceptor for list_quantum_time_slots

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_reallocate_quantum_reservation_grant(self, request: engine.ReallocateQuantumReservationGrantRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.ReallocateQuantumReservationGrantRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for reallocate_quantum_reservation_grant

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_reallocate_quantum_reservation_grant(self, response: quantum.QuantumReservationGrant) -> quantum.QuantumReservationGrant:
        """Post-rpc interceptor for reallocate_quantum_reservation_grant

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_update_quantum_job(self, request: engine.UpdateQuantumJobRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.UpdateQuantumJobRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_update_quantum_job(self, response: quantum.QuantumJob) -> quantum.QuantumJob:
        """Post-rpc interceptor for update_quantum_job

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_update_quantum_program(self, request: engine.UpdateQuantumProgramRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.UpdateQuantumProgramRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_quantum_program

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_update_quantum_program(self, response: quantum.QuantumProgram) -> quantum.QuantumProgram:
        """Post-rpc interceptor for update_quantum_program

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response
    def pre_update_quantum_reservation(self, request: engine.UpdateQuantumReservationRequest, metadata: Sequence[Tuple[str, str]]) -> Tuple[engine.UpdateQuantumReservationRequest, Sequence[Tuple[str, str]]]:
        """Pre-rpc interceptor for update_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_update_quantum_reservation(self, response: quantum.QuantumReservation) -> quantum.QuantumReservation:
        """Post-rpc interceptor for update_quantum_reservation

        Override in a subclass to manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code.
        """
        return response


@dataclasses.dataclass
class QuantumEngineServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: QuantumEngineServiceRestInterceptor


class QuantumEngineServiceRestTransport(QuantumEngineServiceTransport):
    """REST backend transport for QuantumEngineService.

    -

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1

    """

    def __init__(self, *,
            host: str = 'quantum.googleapis.com',
            credentials: Optional[ga_credentials.Credentials] = None,
            credentials_file: Optional[str] = None,
            scopes: Optional[Sequence[str]] = None,
            client_cert_source_for_mtls: Optional[Callable[[
                ], Tuple[bytes, bytes]]] = None,
            quota_project_id: Optional[str] = None,
            client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
            always_use_jwt_access: Optional[bool] = False,
            url_scheme: str = 'https',
            interceptor: Optional[QuantumEngineServiceRestInterceptor] = None,
            api_audience: Optional[str] = None,
            ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'quantum.googleapis.com').
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.

            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Callable[[], Tuple[bytes, bytes]]): Client
                certificate to configure mutual TLS HTTP channel. It is ignored
                if ``channel`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you are developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
            url_scheme: the protocol scheme for the API endpoint.  Normally
                "https", but for testing or local servers,
                "http" can be specified.
        """
        # Run the base constructor
        # TODO(yon-mg): resolve other ctor params i.e. scopes, quota, etc.
        # TODO: When custom host (api_endpoint) is set, `scopes` must *also* be set on the
        # credentials object
        maybe_url_match = re.match("^(?P<scheme>http(?:s)?://)?(?P<host>.*)$", host)
        if maybe_url_match is None:
            raise ValueError(f"Unexpected hostname structure: {host}")  # pragma: NO COVER

        url_match_items = maybe_url_match.groupdict()

        host = f"{url_scheme}://{host}" if not url_match_items["scheme"] else host

        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            api_audience=api_audience
        )
        self._session = AuthorizedSession(
            self._credentials, default_host=self.DEFAULT_HOST)
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or QuantumEngineServiceRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CancelQuantumJob(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("CancelQuantumJob")

        def __call__(self,
                request: engine.CancelQuantumJobRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ):
            r"""Call the cancel quantum job method over HTTP.

            Args:
                request (~.engine.CancelQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'post',
                'uri': '/v1alpha1/{name=projects/*/programs/*/jobs/*}:cancel',
                'body': '*',
            },
            ]
            request, metadata = self._interceptor.pre_cancel_quantum_job(request, metadata)
            pb_request = engine.CancelQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _CancelQuantumReservation(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("CancelQuantumReservation")

        def __call__(self,
                request: engine.CancelQuantumReservationRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumReservation:
            r"""Call the cancel quantum
        reservation method over HTTP.

            Args:
                request (~.engine.CancelQuantumReservationRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumReservation:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'post',
                'uri': '/v1alpha1/{name=projects/*/processors/*/reservations/*}:cancel',
                'body': '*',
            },
            ]
            request, metadata = self._interceptor.pre_cancel_quantum_reservation(request, metadata)
            pb_request = engine.CancelQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumReservation()
            pb_resp = quantum.QuantumReservation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_cancel_quantum_reservation(resp)
            return resp

    class _CreateQuantumJob(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("CreateQuantumJob")

        def __call__(self,
                request: engine.CreateQuantumJobRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumJob:
            r"""Call the create quantum job method over HTTP.

            Args:
                request (~.engine.CreateQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumJob:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'post',
                'uri': '/v1alpha1/{parent=projects/*/programs/*}/jobs',
                'body': 'quantum_job',
            },
            ]
            request, metadata = self._interceptor.pre_create_quantum_job(request, metadata)
            pb_request = engine.CreateQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumJob()
            pb_resp = quantum.QuantumJob.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_quantum_job(resp)
            return resp

    class _CreateQuantumProgram(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("CreateQuantumProgram")

        def __call__(self,
                request: engine.CreateQuantumProgramRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumProgram:
            r"""Call the create quantum program method over HTTP.

            Args:
                request (~.engine.CreateQuantumProgramRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumProgram:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'post',
                'uri': '/v1alpha1/{parent=projects/*}/programs',
                'body': 'quantum_program',
            },
            ]
            request, metadata = self._interceptor.pre_create_quantum_program(request, metadata)
            pb_request = engine.CreateQuantumProgramRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumProgram()
            pb_resp = quantum.QuantumProgram.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_quantum_program(resp)
            return resp

    class _CreateQuantumReservation(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("CreateQuantumReservation")

        def __call__(self,
                request: engine.CreateQuantumReservationRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumReservation:
            r"""Call the create quantum
        reservation method over HTTP.

            Args:
                request (~.engine.CreateQuantumReservationRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumReservation:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'post',
                'uri': '/v1alpha1/{parent=projects/*/processors/*}/reservations',
                'body': 'quantum_reservation',
            },
            ]
            request, metadata = self._interceptor.pre_create_quantum_reservation(request, metadata)
            pb_request = engine.CreateQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumReservation()
            pb_resp = quantum.QuantumReservation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_create_quantum_reservation(resp)
            return resp

    class _DeleteQuantumJob(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("DeleteQuantumJob")

        def __call__(self,
                request: engine.DeleteQuantumJobRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ):
            r"""Call the delete quantum job method over HTTP.

            Args:
                request (~.engine.DeleteQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'delete',
                'uri': '/v1alpha1/{name=projects/*/programs/*/jobs/*}',
            },
            ]
            request, metadata = self._interceptor.pre_delete_quantum_job(request, metadata)
            pb_request = engine.DeleteQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteQuantumProgram(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("DeleteQuantumProgram")

        def __call__(self,
                request: engine.DeleteQuantumProgramRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ):
            r"""Call the delete quantum program method over HTTP.

            Args:
                request (~.engine.DeleteQuantumProgramRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'delete',
                'uri': '/v1alpha1/{name=projects/*/programs/*}',
            },
            ]
            request, metadata = self._interceptor.pre_delete_quantum_program(request, metadata)
            pb_request = engine.DeleteQuantumProgramRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteQuantumReservation(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("DeleteQuantumReservation")

        def __call__(self,
                request: engine.DeleteQuantumReservationRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ):
            r"""Call the delete quantum
        reservation method over HTTP.

            Args:
                request (~.engine.DeleteQuantumReservationRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'delete',
                'uri': '/v1alpha1/{name=projects/*/processors/*/reservations/*}',
            },
            ]
            request, metadata = self._interceptor.pre_delete_quantum_reservation(request, metadata)
            pb_request = engine.DeleteQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _GetQuantumCalibration(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("GetQuantumCalibration")

        def __call__(self,
                request: engine.GetQuantumCalibrationRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumCalibration:
            r"""Call the get quantum calibration method over HTTP.

            Args:
                request (~.engine.GetQuantumCalibrationRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumCalibration:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{name=projects/*/processors/*/calibrations/*}',
            },
            ]
            request, metadata = self._interceptor.pre_get_quantum_calibration(request, metadata)
            pb_request = engine.GetQuantumCalibrationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumCalibration()
            pb_resp = quantum.QuantumCalibration.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_quantum_calibration(resp)
            return resp

    class _GetQuantumJob(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("GetQuantumJob")

        def __call__(self,
                request: engine.GetQuantumJobRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumJob:
            r"""Call the get quantum job method over HTTP.

            Args:
                request (~.engine.GetQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumJob:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{name=projects/*/programs/*/jobs/*}',
            },
            ]
            request, metadata = self._interceptor.pre_get_quantum_job(request, metadata)
            pb_request = engine.GetQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumJob()
            pb_resp = quantum.QuantumJob.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_quantum_job(resp)
            return resp

    class _GetQuantumProcessor(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("GetQuantumProcessor")

        def __call__(self,
                request: engine.GetQuantumProcessorRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumProcessor:
            r"""Call the get quantum processor method over HTTP.

            Args:
                request (~.engine.GetQuantumProcessorRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumProcessor:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{name=projects/*/processors/*}',
            },
            ]
            request, metadata = self._interceptor.pre_get_quantum_processor(request, metadata)
            pb_request = engine.GetQuantumProcessorRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumProcessor()
            pb_resp = quantum.QuantumProcessor.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_quantum_processor(resp)
            return resp

    class _GetQuantumProgram(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("GetQuantumProgram")

        def __call__(self,
                request: engine.GetQuantumProgramRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumProgram:
            r"""Call the get quantum program method over HTTP.

            Args:
                request (~.engine.GetQuantumProgramRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumProgram:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{name=projects/*/programs/*}',
            },
            ]
            request, metadata = self._interceptor.pre_get_quantum_program(request, metadata)
            pb_request = engine.GetQuantumProgramRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumProgram()
            pb_resp = quantum.QuantumProgram.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_quantum_program(resp)
            return resp

    class _GetQuantumReservation(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("GetQuantumReservation")

        def __call__(self,
                request: engine.GetQuantumReservationRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumReservation:
            r"""Call the get quantum reservation method over HTTP.

            Args:
                request (~.engine.GetQuantumReservationRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumReservation:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{name=projects/*/processors/*/reservations/*}',
            },
            ]
            request, metadata = self._interceptor.pre_get_quantum_reservation(request, metadata)
            pb_request = engine.GetQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumReservation()
            pb_resp = quantum.QuantumReservation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_quantum_reservation(resp)
            return resp

    class _GetQuantumResult(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("GetQuantumResult")

        def __call__(self,
                request: engine.GetQuantumResultRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumResult:
            r"""Call the get quantum result method over HTTP.

            Args:
                request (~.engine.GetQuantumResultRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumResult:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*/programs/*/jobs/*}/result',
            },
            ]
            request, metadata = self._interceptor.pre_get_quantum_result(request, metadata)
            pb_request = engine.GetQuantumResultRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumResult()
            pb_resp = quantum.QuantumResult.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_get_quantum_result(resp)
            return resp

    class _ListQuantumCalibrations(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumCalibrations")

        def __call__(self,
                request: engine.ListQuantumCalibrationsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumCalibrationsResponse:
            r"""Call the list quantum calibrations method over HTTP.

            Args:
                request (~.engine.ListQuantumCalibrationsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumCalibrationsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*/processors/*}/calibrations',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_calibrations(request, metadata)
            pb_request = engine.ListQuantumCalibrationsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumCalibrationsResponse()
            pb_resp = engine.ListQuantumCalibrationsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_calibrations(resp)
            return resp

    class _ListQuantumJobEvents(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumJobEvents")

        def __call__(self,
                request: engine.ListQuantumJobEventsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumJobEventsResponse:
            r"""Call the list quantum job events method over HTTP.

            Args:
                request (~.engine.ListQuantumJobEventsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumJobEventsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*/programs/*/jobs/*}/events',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_job_events(request, metadata)
            pb_request = engine.ListQuantumJobEventsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumJobEventsResponse()
            pb_resp = engine.ListQuantumJobEventsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_job_events(resp)
            return resp

    class _ListQuantumJobs(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumJobs")

        def __call__(self,
                request: engine.ListQuantumJobsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumJobsResponse:
            r"""Call the list quantum jobs method over HTTP.

            Args:
                request (~.engine.ListQuantumJobsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumJobsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*/programs/*}/jobs',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_jobs(request, metadata)
            pb_request = engine.ListQuantumJobsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumJobsResponse()
            pb_resp = engine.ListQuantumJobsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_jobs(resp)
            return resp

    class _ListQuantumProcessors(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumProcessors")

        def __call__(self,
                request: engine.ListQuantumProcessorsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumProcessorsResponse:
            r"""Call the list quantum processors method over HTTP.

            Args:
                request (~.engine.ListQuantumProcessorsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumProcessorsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*}/processors',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_processors(request, metadata)
            pb_request = engine.ListQuantumProcessorsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumProcessorsResponse()
            pb_resp = engine.ListQuantumProcessorsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_processors(resp)
            return resp

    class _ListQuantumPrograms(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumPrograms")

        def __call__(self,
                request: engine.ListQuantumProgramsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumProgramsResponse:
            r"""Call the list quantum programs method over HTTP.

            Args:
                request (~.engine.ListQuantumProgramsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumProgramsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*}/programs',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_programs(request, metadata)
            pb_request = engine.ListQuantumProgramsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumProgramsResponse()
            pb_resp = engine.ListQuantumProgramsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_programs(resp)
            return resp

    class _ListQuantumReservationBudgets(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumReservationBudgets")

        def __call__(self,
                request: engine.ListQuantumReservationBudgetsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumReservationBudgetsResponse:
            r"""Call the list quantum reservation
        budgets method over HTTP.

            Args:
                request (~.engine.ListQuantumReservationBudgetsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumReservationBudgetsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*}/reservationBudgets',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_reservation_budgets(request, metadata)
            pb_request = engine.ListQuantumReservationBudgetsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumReservationBudgetsResponse()
            pb_resp = engine.ListQuantumReservationBudgetsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_reservation_budgets(resp)
            return resp

    class _ListQuantumReservationGrants(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumReservationGrants")

        def __call__(self,
                request: engine.ListQuantumReservationGrantsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumReservationGrantsResponse:
            r"""Call the list quantum reservation
        grants method over HTTP.

            Args:
                request (~.engine.ListQuantumReservationGrantsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumReservationGrantsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*}/reservationGrant',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_reservation_grants(request, metadata)
            pb_request = engine.ListQuantumReservationGrantsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumReservationGrantsResponse()
            pb_resp = engine.ListQuantumReservationGrantsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_reservation_grants(resp)
            return resp

    class _ListQuantumReservations(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumReservations")

        def __call__(self,
                request: engine.ListQuantumReservationsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumReservationsResponse:
            r"""Call the list quantum reservations method over HTTP.

            Args:
                request (~.engine.ListQuantumReservationsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumReservationsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*/processors/*}/reservations',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_reservations(request, metadata)
            pb_request = engine.ListQuantumReservationsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumReservationsResponse()
            pb_resp = engine.ListQuantumReservationsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_reservations(resp)
            return resp

    class _ListQuantumTimeSlots(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ListQuantumTimeSlots")

        def __call__(self,
                request: engine.ListQuantumTimeSlotsRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> engine.ListQuantumTimeSlotsResponse:
            r"""Call the list quantum time slots method over HTTP.

            Args:
                request (~.engine.ListQuantumTimeSlotsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.engine.ListQuantumTimeSlotsResponse:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'get',
                'uri': '/v1alpha1/{parent=projects/*/processors/*}/timeSlots',
            },
            ]
            request, metadata = self._interceptor.pre_list_quantum_time_slots(request, metadata)
            pb_request = engine.ListQuantumTimeSlotsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumTimeSlotsResponse()
            pb_resp = engine.ListQuantumTimeSlotsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_list_quantum_time_slots(resp)
            return resp

    class _QuantumRunStream(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("QuantumRunStream")

        def __call__(self,
                request: engine.QuantumRunStreamRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> rest_streaming.ResponseIterator:
            raise NotImplementedError(
                "Method QuantumRunStream is not available over REST transport"
            )
    class _ReallocateQuantumReservationGrant(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("ReallocateQuantumReservationGrant")

        def __call__(self,
                request: engine.ReallocateQuantumReservationGrantRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumReservationGrant:
            r"""Call the reallocate quantum
        reservation grant method over HTTP.

            Args:
                request (~.engine.ReallocateQuantumReservationGrantRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumReservationGrant:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'post',
                'uri': '/v1alpha1/{name=projects/*/reservationGrant/*}:reallocate',
                'body': '*',
            },
            ]
            request, metadata = self._interceptor.pre_reallocate_quantum_reservation_grant(request, metadata)
            pb_request = engine.ReallocateQuantumReservationGrantRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumReservationGrant()
            pb_resp = quantum.QuantumReservationGrant.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_reallocate_quantum_reservation_grant(resp)
            return resp

    class _UpdateQuantumJob(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("UpdateQuantumJob")

        def __call__(self,
                request: engine.UpdateQuantumJobRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumJob:
            r"""Call the update quantum job method over HTTP.

            Args:
                request (~.engine.UpdateQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumJob:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'patch',
                'uri': '/v1alpha1/{name=projects/*/programs/*/jobs/*}',
                'body': 'quantum_job',
            },
            ]
            request, metadata = self._interceptor.pre_update_quantum_job(request, metadata)
            pb_request = engine.UpdateQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumJob()
            pb_resp = quantum.QuantumJob.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_quantum_job(resp)
            return resp

    class _UpdateQuantumProgram(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("UpdateQuantumProgram")

        def __call__(self,
                request: engine.UpdateQuantumProgramRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumProgram:
            r"""Call the update quantum program method over HTTP.

            Args:
                request (~.engine.UpdateQuantumProgramRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumProgram:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'patch',
                'uri': '/v1alpha1/{name=projects/*/programs/*}',
                'body': 'quantum_program',
            },
            ]
            request, metadata = self._interceptor.pre_update_quantum_program(request, metadata)
            pb_request = engine.UpdateQuantumProgramRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumProgram()
            pb_resp = quantum.QuantumProgram.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_quantum_program(resp)
            return resp

    class _UpdateQuantumReservation(QuantumEngineServiceRestStub):
        def __hash__(self):
            return hash("UpdateQuantumReservation")

        def __call__(self,
                request: engine.UpdateQuantumReservationRequest, *,
                retry: OptionalRetry=gapic_v1.method.DEFAULT,
                timeout: Optional[float]=None,
                metadata: Sequence[Tuple[str, str]]=(),
                ) -> quantum.QuantumReservation:
            r"""Call the update quantum
        reservation method over HTTP.

            Args:
                request (~.engine.UpdateQuantumReservationRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, str]]): Strings which should be
                    sent along with the request as metadata.

            Returns:
                ~.quantum.QuantumReservation:
                    -
            """

            http_options: List[Dict[str, str]] = [{
                'method': 'patch',
                'uri': '/v1alpha1/{name=projects/*/processors/*/reservations/*}',
                'body': 'quantum_reservation',
            },
            ]
            request, metadata = self._interceptor.pre_update_quantum_reservation(request, metadata)
            pb_request = engine.UpdateQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)

            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'],
                use_integers_for_enums=True
            )
            uri = transcoded_request['uri']
            method = transcoded_request['method']

            # Jsonify the query params
            query_params = json.loads(json_format.MessageToJson(
                transcoded_request['query_params'],
                use_integers_for_enums=True,
            ))

            query_params["$alt"] = "json;enum-encoding=int"

            # Send the request
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(self._session, method)(
                "{host}{uri}".format(host=self._host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
                )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumReservation()
            pb_resp = quantum.QuantumReservation.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)
            resp = self._interceptor.post_update_quantum_reservation(resp)
            return resp

    @property
    def cancel_quantum_job(self) -> Callable[
            [engine.CancelQuantumJobRequest],
            empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CancelQuantumJob(self._session, self._host, self._interceptor) # type: ignore

    @property
    def cancel_quantum_reservation(self) -> Callable[
            [engine.CancelQuantumReservationRequest],
            quantum.QuantumReservation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CancelQuantumReservation(self._session, self._host, self._interceptor) # type: ignore

    @property
    def create_quantum_job(self) -> Callable[
            [engine.CreateQuantumJobRequest],
            quantum.QuantumJob]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateQuantumJob(self._session, self._host, self._interceptor) # type: ignore

    @property
    def create_quantum_program(self) -> Callable[
            [engine.CreateQuantumProgramRequest],
            quantum.QuantumProgram]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateQuantumProgram(self._session, self._host, self._interceptor) # type: ignore

    @property
    def create_quantum_reservation(self) -> Callable[
            [engine.CreateQuantumReservationRequest],
            quantum.QuantumReservation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateQuantumReservation(self._session, self._host, self._interceptor) # type: ignore

    @property
    def delete_quantum_job(self) -> Callable[
            [engine.DeleteQuantumJobRequest],
            empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteQuantumJob(self._session, self._host, self._interceptor) # type: ignore

    @property
    def delete_quantum_program(self) -> Callable[
            [engine.DeleteQuantumProgramRequest],
            empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteQuantumProgram(self._session, self._host, self._interceptor) # type: ignore

    @property
    def delete_quantum_reservation(self) -> Callable[
            [engine.DeleteQuantumReservationRequest],
            empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteQuantumReservation(self._session, self._host, self._interceptor) # type: ignore

    @property
    def get_quantum_calibration(self) -> Callable[
            [engine.GetQuantumCalibrationRequest],
            quantum.QuantumCalibration]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumCalibration(self._session, self._host, self._interceptor) # type: ignore

    @property
    def get_quantum_job(self) -> Callable[
            [engine.GetQuantumJobRequest],
            quantum.QuantumJob]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumJob(self._session, self._host, self._interceptor) # type: ignore

    @property
    def get_quantum_processor(self) -> Callable[
            [engine.GetQuantumProcessorRequest],
            quantum.QuantumProcessor]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumProcessor(self._session, self._host, self._interceptor) # type: ignore

    @property
    def get_quantum_program(self) -> Callable[
            [engine.GetQuantumProgramRequest],
            quantum.QuantumProgram]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumProgram(self._session, self._host, self._interceptor) # type: ignore

    @property
    def get_quantum_reservation(self) -> Callable[
            [engine.GetQuantumReservationRequest],
            quantum.QuantumReservation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumReservation(self._session, self._host, self._interceptor) # type: ignore

    @property
    def get_quantum_result(self) -> Callable[
            [engine.GetQuantumResultRequest],
            quantum.QuantumResult]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumResult(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_calibrations(self) -> Callable[
            [engine.ListQuantumCalibrationsRequest],
            engine.ListQuantumCalibrationsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumCalibrations(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_job_events(self) -> Callable[
            [engine.ListQuantumJobEventsRequest],
            engine.ListQuantumJobEventsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumJobEvents(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_jobs(self) -> Callable[
            [engine.ListQuantumJobsRequest],
            engine.ListQuantumJobsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumJobs(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_processors(self) -> Callable[
            [engine.ListQuantumProcessorsRequest],
            engine.ListQuantumProcessorsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumProcessors(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_programs(self) -> Callable[
            [engine.ListQuantumProgramsRequest],
            engine.ListQuantumProgramsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumPrograms(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_reservation_budgets(self) -> Callable[
            [engine.ListQuantumReservationBudgetsRequest],
            engine.ListQuantumReservationBudgetsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumReservationBudgets(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_reservation_grants(self) -> Callable[
            [engine.ListQuantumReservationGrantsRequest],
            engine.ListQuantumReservationGrantsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumReservationGrants(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_reservations(self) -> Callable[
            [engine.ListQuantumReservationsRequest],
            engine.ListQuantumReservationsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumReservations(self._session, self._host, self._interceptor) # type: ignore

    @property
    def list_quantum_time_slots(self) -> Callable[
            [engine.ListQuantumTimeSlotsRequest],
            engine.ListQuantumTimeSlotsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumTimeSlots(self._session, self._host, self._interceptor) # type: ignore

    @property
    def quantum_run_stream(self) -> Callable[
            [engine.QuantumRunStreamRequest],
            engine.QuantumRunStreamResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._QuantumRunStream(self._session, self._host, self._interceptor) # type: ignore

    @property
    def reallocate_quantum_reservation_grant(self) -> Callable[
            [engine.ReallocateQuantumReservationGrantRequest],
            quantum.QuantumReservationGrant]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ReallocateQuantumReservationGrant(self._session, self._host, self._interceptor) # type: ignore

    @property
    def update_quantum_job(self) -> Callable[
            [engine.UpdateQuantumJobRequest],
            quantum.QuantumJob]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateQuantumJob(self._session, self._host, self._interceptor) # type: ignore

    @property
    def update_quantum_program(self) -> Callable[
            [engine.UpdateQuantumProgramRequest],
            quantum.QuantumProgram]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateQuantumProgram(self._session, self._host, self._interceptor) # type: ignore

    @property
    def update_quantum_reservation(self) -> Callable[
            [engine.UpdateQuantumReservationRequest],
            quantum.QuantumReservation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateQuantumReservation(self._session, self._host, self._interceptor) # type: ignore

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__=(
    'QuantumEngineServiceRestTransport',
)
