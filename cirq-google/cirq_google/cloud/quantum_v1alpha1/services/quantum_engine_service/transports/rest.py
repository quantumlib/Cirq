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
import dataclasses
import json  # type: ignore
import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import google.protobuf
from google.api_core import (
    exceptions as core_exceptions,
    gapic_v1,
    rest_helpers,
    rest_streaming,
    retry as retries,
)
from google.auth import credentials as ga_credentials  # type: ignore
from google.auth.transport.requests import AuthorizedSession  # type: ignore
from google.protobuf import empty_pb2  # type: ignore
from google.protobuf import json_format
from requests import __version__ as requests_version

from cirq_google.cloud.quantum_v1alpha1.types import engine, quantum

from .base import DEFAULT_CLIENT_INFO as BASE_DEFAULT_CLIENT_INFO
from .rest_base import _BaseQuantumEngineServiceRestTransport

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault, None]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object, None]  # type: ignore

try:
    from google.api_core import client_logging  # type: ignore

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = logging.getLogger(__name__)

DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
    gapic_version=BASE_DEFAULT_CLIENT_INFO.gapic_version,
    grpc_version=None,
    rest_version=f"requests@{requests_version}",
)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


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

            def pre_get_quantum_processor_config(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_get_quantum_processor_config(self, response):
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

            def pre_list_quantum_processor_configs(self, request, metadata):
                logging.log(f"Received request: {request}")
                return request, metadata

            def post_list_quantum_processor_configs(self, response):
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

    def pre_cancel_quantum_job(
        self,
        request: engine.CancelQuantumJobRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.CancelQuantumJobRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for cancel_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def pre_cancel_quantum_reservation(
        self,
        request: engine.CancelQuantumReservationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.CancelQuantumReservationRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for cancel_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_cancel_quantum_reservation(
        self, response: quantum.QuantumReservation
    ) -> quantum.QuantumReservation:
        """Post-rpc interceptor for cancel_quantum_reservation

        DEPRECATED. Please use the `post_cancel_quantum_reservation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_cancel_quantum_reservation` interceptor runs
        before the `post_cancel_quantum_reservation_with_metadata` interceptor.
        """
        return response

    def post_cancel_quantum_reservation_with_metadata(
        self,
        response: quantum.QuantumReservation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[quantum.QuantumReservation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for cancel_quantum_reservation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_cancel_quantum_reservation_with_metadata`
        interceptor in new development instead of the `post_cancel_quantum_reservation` interceptor.
        When both interceptors are used, this `post_cancel_quantum_reservation_with_metadata` interceptor runs after the
        `post_cancel_quantum_reservation` interceptor. The (possibly modified) response returned by
        `post_cancel_quantum_reservation` will be passed to
        `post_cancel_quantum_reservation_with_metadata`.
        """
        return response, metadata

    def pre_create_quantum_job(
        self,
        request: engine.CreateQuantumJobRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.CreateQuantumJobRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for create_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_create_quantum_job(self, response: quantum.QuantumJob) -> quantum.QuantumJob:
        """Post-rpc interceptor for create_quantum_job

        DEPRECATED. Please use the `post_create_quantum_job_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_create_quantum_job` interceptor runs
        before the `post_create_quantum_job_with_metadata` interceptor.
        """
        return response

    def post_create_quantum_job_with_metadata(
        self, response: quantum.QuantumJob, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[quantum.QuantumJob, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_quantum_job

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_create_quantum_job_with_metadata`
        interceptor in new development instead of the `post_create_quantum_job` interceptor.
        When both interceptors are used, this `post_create_quantum_job_with_metadata` interceptor runs after the
        `post_create_quantum_job` interceptor. The (possibly modified) response returned by
        `post_create_quantum_job` will be passed to
        `post_create_quantum_job_with_metadata`.
        """
        return response, metadata

    def pre_create_quantum_program(
        self,
        request: engine.CreateQuantumProgramRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.CreateQuantumProgramRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for create_quantum_program

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_create_quantum_program(
        self, response: quantum.QuantumProgram
    ) -> quantum.QuantumProgram:
        """Post-rpc interceptor for create_quantum_program

        DEPRECATED. Please use the `post_create_quantum_program_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_create_quantum_program` interceptor runs
        before the `post_create_quantum_program_with_metadata` interceptor.
        """
        return response

    def post_create_quantum_program_with_metadata(
        self, response: quantum.QuantumProgram, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[quantum.QuantumProgram, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_quantum_program

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_create_quantum_program_with_metadata`
        interceptor in new development instead of the `post_create_quantum_program` interceptor.
        When both interceptors are used, this `post_create_quantum_program_with_metadata` interceptor runs after the
        `post_create_quantum_program` interceptor. The (possibly modified) response returned by
        `post_create_quantum_program` will be passed to
        `post_create_quantum_program_with_metadata`.
        """
        return response, metadata

    def pre_create_quantum_reservation(
        self,
        request: engine.CreateQuantumReservationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.CreateQuantumReservationRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for create_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_create_quantum_reservation(
        self, response: quantum.QuantumReservation
    ) -> quantum.QuantumReservation:
        """Post-rpc interceptor for create_quantum_reservation

        DEPRECATED. Please use the `post_create_quantum_reservation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_create_quantum_reservation` interceptor runs
        before the `post_create_quantum_reservation_with_metadata` interceptor.
        """
        return response

    def post_create_quantum_reservation_with_metadata(
        self,
        response: quantum.QuantumReservation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[quantum.QuantumReservation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for create_quantum_reservation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_create_quantum_reservation_with_metadata`
        interceptor in new development instead of the `post_create_quantum_reservation` interceptor.
        When both interceptors are used, this `post_create_quantum_reservation_with_metadata` interceptor runs after the
        `post_create_quantum_reservation` interceptor. The (possibly modified) response returned by
        `post_create_quantum_reservation` will be passed to
        `post_create_quantum_reservation_with_metadata`.
        """
        return response, metadata

    def pre_delete_quantum_job(
        self,
        request: engine.DeleteQuantumJobRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.DeleteQuantumJobRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for delete_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def pre_delete_quantum_program(
        self,
        request: engine.DeleteQuantumProgramRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.DeleteQuantumProgramRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for delete_quantum_program

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def pre_delete_quantum_reservation(
        self,
        request: engine.DeleteQuantumReservationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.DeleteQuantumReservationRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for delete_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def pre_get_quantum_calibration(
        self,
        request: engine.GetQuantumCalibrationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.GetQuantumCalibrationRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_quantum_calibration

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_calibration(
        self, response: quantum.QuantumCalibration
    ) -> quantum.QuantumCalibration:
        """Post-rpc interceptor for get_quantum_calibration

        DEPRECATED. Please use the `post_get_quantum_calibration_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_get_quantum_calibration` interceptor runs
        before the `post_get_quantum_calibration_with_metadata` interceptor.
        """
        return response

    def post_get_quantum_calibration_with_metadata(
        self,
        response: quantum.QuantumCalibration,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[quantum.QuantumCalibration, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_quantum_calibration

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_get_quantum_calibration_with_metadata`
        interceptor in new development instead of the `post_get_quantum_calibration` interceptor.
        When both interceptors are used, this `post_get_quantum_calibration_with_metadata` interceptor runs after the
        `post_get_quantum_calibration` interceptor. The (possibly modified) response returned by
        `post_get_quantum_calibration` will be passed to
        `post_get_quantum_calibration_with_metadata`.
        """
        return response, metadata

    def pre_get_quantum_job(
        self,
        request: engine.GetQuantumJobRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.GetQuantumJobRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_job(self, response: quantum.QuantumJob) -> quantum.QuantumJob:
        """Post-rpc interceptor for get_quantum_job

        DEPRECATED. Please use the `post_get_quantum_job_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_get_quantum_job` interceptor runs
        before the `post_get_quantum_job_with_metadata` interceptor.
        """
        return response

    def post_get_quantum_job_with_metadata(
        self, response: quantum.QuantumJob, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[quantum.QuantumJob, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_quantum_job

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_get_quantum_job_with_metadata`
        interceptor in new development instead of the `post_get_quantum_job` interceptor.
        When both interceptors are used, this `post_get_quantum_job_with_metadata` interceptor runs after the
        `post_get_quantum_job` interceptor. The (possibly modified) response returned by
        `post_get_quantum_job` will be passed to
        `post_get_quantum_job_with_metadata`.
        """
        return response, metadata

    def pre_get_quantum_processor(
        self,
        request: engine.GetQuantumProcessorRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.GetQuantumProcessorRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_quantum_processor

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_processor(
        self, response: quantum.QuantumProcessor
    ) -> quantum.QuantumProcessor:
        """Post-rpc interceptor for get_quantum_processor

        DEPRECATED. Please use the `post_get_quantum_processor_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_get_quantum_processor` interceptor runs
        before the `post_get_quantum_processor_with_metadata` interceptor.
        """
        return response

    def post_get_quantum_processor_with_metadata(
        self, response: quantum.QuantumProcessor, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[quantum.QuantumProcessor, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_quantum_processor

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_get_quantum_processor_with_metadata`
        interceptor in new development instead of the `post_get_quantum_processor` interceptor.
        When both interceptors are used, this `post_get_quantum_processor_with_metadata` interceptor runs after the
        `post_get_quantum_processor` interceptor. The (possibly modified) response returned by
        `post_get_quantum_processor` will be passed to
        `post_get_quantum_processor_with_metadata`.
        """
        return response, metadata

    def pre_get_quantum_processor_config(
        self,
        request: engine.GetQuantumProcessorConfigRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.GetQuantumProcessorConfigRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_quantum_processor_config

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_processor_config(
        self, response: quantum.QuantumProcessorConfig
    ) -> quantum.QuantumProcessorConfig:
        """Post-rpc interceptor for get_quantum_processor_config

        DEPRECATED. Please use the `post_get_quantum_processor_config_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_get_quantum_processor_config` interceptor runs
        before the `post_get_quantum_processor_config_with_metadata` interceptor.
        """
        return response

    def post_get_quantum_processor_config_with_metadata(
        self,
        response: quantum.QuantumProcessorConfig,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[quantum.QuantumProcessorConfig, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_quantum_processor_config

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_get_quantum_processor_config_with_metadata`
        interceptor in new development instead of the `post_get_quantum_processor_config` interceptor.
        When both interceptors are used, this `post_get_quantum_processor_config_with_metadata` interceptor runs after the
        `post_get_quantum_processor_config` interceptor. The (possibly modified) response returned by
        `post_get_quantum_processor_config` will be passed to
        `post_get_quantum_processor_config_with_metadata`.
        """
        return response, metadata

    def pre_get_quantum_program(
        self,
        request: engine.GetQuantumProgramRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.GetQuantumProgramRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_quantum_program

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_program(self, response: quantum.QuantumProgram) -> quantum.QuantumProgram:
        """Post-rpc interceptor for get_quantum_program

        DEPRECATED. Please use the `post_get_quantum_program_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_get_quantum_program` interceptor runs
        before the `post_get_quantum_program_with_metadata` interceptor.
        """
        return response

    def post_get_quantum_program_with_metadata(
        self, response: quantum.QuantumProgram, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[quantum.QuantumProgram, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_quantum_program

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_get_quantum_program_with_metadata`
        interceptor in new development instead of the `post_get_quantum_program` interceptor.
        When both interceptors are used, this `post_get_quantum_program_with_metadata` interceptor runs after the
        `post_get_quantum_program` interceptor. The (possibly modified) response returned by
        `post_get_quantum_program` will be passed to
        `post_get_quantum_program_with_metadata`.
        """
        return response, metadata

    def pre_get_quantum_reservation(
        self,
        request: engine.GetQuantumReservationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.GetQuantumReservationRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_reservation(
        self, response: quantum.QuantumReservation
    ) -> quantum.QuantumReservation:
        """Post-rpc interceptor for get_quantum_reservation

        DEPRECATED. Please use the `post_get_quantum_reservation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_get_quantum_reservation` interceptor runs
        before the `post_get_quantum_reservation_with_metadata` interceptor.
        """
        return response

    def post_get_quantum_reservation_with_metadata(
        self,
        response: quantum.QuantumReservation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[quantum.QuantumReservation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_quantum_reservation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_get_quantum_reservation_with_metadata`
        interceptor in new development instead of the `post_get_quantum_reservation` interceptor.
        When both interceptors are used, this `post_get_quantum_reservation_with_metadata` interceptor runs after the
        `post_get_quantum_reservation` interceptor. The (possibly modified) response returned by
        `post_get_quantum_reservation` will be passed to
        `post_get_quantum_reservation_with_metadata`.
        """
        return response, metadata

    def pre_get_quantum_result(
        self,
        request: engine.GetQuantumResultRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.GetQuantumResultRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for get_quantum_result

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_get_quantum_result(self, response: quantum.QuantumResult) -> quantum.QuantumResult:
        """Post-rpc interceptor for get_quantum_result

        DEPRECATED. Please use the `post_get_quantum_result_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_get_quantum_result` interceptor runs
        before the `post_get_quantum_result_with_metadata` interceptor.
        """
        return response

    def post_get_quantum_result_with_metadata(
        self, response: quantum.QuantumResult, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[quantum.QuantumResult, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for get_quantum_result

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_get_quantum_result_with_metadata`
        interceptor in new development instead of the `post_get_quantum_result` interceptor.
        When both interceptors are used, this `post_get_quantum_result_with_metadata` interceptor runs after the
        `post_get_quantum_result` interceptor. The (possibly modified) response returned by
        `post_get_quantum_result` will be passed to
        `post_get_quantum_result_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_calibrations(
        self,
        request: engine.ListQuantumCalibrationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumCalibrationsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_calibrations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_calibrations(
        self, response: engine.ListQuantumCalibrationsResponse
    ) -> engine.ListQuantumCalibrationsResponse:
        """Post-rpc interceptor for list_quantum_calibrations

        DEPRECATED. Please use the `post_list_quantum_calibrations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_calibrations` interceptor runs
        before the `post_list_quantum_calibrations_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_calibrations_with_metadata(
        self,
        response: engine.ListQuantumCalibrationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumCalibrationsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_quantum_calibrations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_calibrations_with_metadata`
        interceptor in new development instead of the `post_list_quantum_calibrations` interceptor.
        When both interceptors are used, this `post_list_quantum_calibrations_with_metadata` interceptor runs after the
        `post_list_quantum_calibrations` interceptor. The (possibly modified) response returned by
        `post_list_quantum_calibrations` will be passed to
        `post_list_quantum_calibrations_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_job_events(
        self,
        request: engine.ListQuantumJobEventsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumJobEventsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_job_events

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_job_events(
        self, response: engine.ListQuantumJobEventsResponse
    ) -> engine.ListQuantumJobEventsResponse:
        """Post-rpc interceptor for list_quantum_job_events

        DEPRECATED. Please use the `post_list_quantum_job_events_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_job_events` interceptor runs
        before the `post_list_quantum_job_events_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_job_events_with_metadata(
        self,
        response: engine.ListQuantumJobEventsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumJobEventsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_quantum_job_events

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_job_events_with_metadata`
        interceptor in new development instead of the `post_list_quantum_job_events` interceptor.
        When both interceptors are used, this `post_list_quantum_job_events_with_metadata` interceptor runs after the
        `post_list_quantum_job_events` interceptor. The (possibly modified) response returned by
        `post_list_quantum_job_events` will be passed to
        `post_list_quantum_job_events_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_jobs(
        self,
        request: engine.ListQuantumJobsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumJobsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_jobs

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_jobs(
        self, response: engine.ListQuantumJobsResponse
    ) -> engine.ListQuantumJobsResponse:
        """Post-rpc interceptor for list_quantum_jobs

        DEPRECATED. Please use the `post_list_quantum_jobs_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_jobs` interceptor runs
        before the `post_list_quantum_jobs_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_jobs_with_metadata(
        self,
        response: engine.ListQuantumJobsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumJobsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_quantum_jobs

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_jobs_with_metadata`
        interceptor in new development instead of the `post_list_quantum_jobs` interceptor.
        When both interceptors are used, this `post_list_quantum_jobs_with_metadata` interceptor runs after the
        `post_list_quantum_jobs` interceptor. The (possibly modified) response returned by
        `post_list_quantum_jobs` will be passed to
        `post_list_quantum_jobs_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_processor_configs(
        self,
        request: engine.ListQuantumProcessorConfigsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumProcessorConfigsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_processor_configs

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_processor_configs(
        self, response: engine.ListQuantumProcessorConfigsResponse
    ) -> engine.ListQuantumProcessorConfigsResponse:
        """Post-rpc interceptor for list_quantum_processor_configs

        DEPRECATED. Please use the `post_list_quantum_processor_configs_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_processor_configs` interceptor runs
        before the `post_list_quantum_processor_configs_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_processor_configs_with_metadata(
        self,
        response: engine.ListQuantumProcessorConfigsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumProcessorConfigsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_quantum_processor_configs

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_processor_configs_with_metadata`
        interceptor in new development instead of the `post_list_quantum_processor_configs` interceptor.
        When both interceptors are used, this `post_list_quantum_processor_configs_with_metadata` interceptor runs after the
        `post_list_quantum_processor_configs` interceptor. The (possibly modified) response returned by
        `post_list_quantum_processor_configs` will be passed to
        `post_list_quantum_processor_configs_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_processors(
        self,
        request: engine.ListQuantumProcessorsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumProcessorsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_processors

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_processors(
        self, response: engine.ListQuantumProcessorsResponse
    ) -> engine.ListQuantumProcessorsResponse:
        """Post-rpc interceptor for list_quantum_processors

        DEPRECATED. Please use the `post_list_quantum_processors_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_processors` interceptor runs
        before the `post_list_quantum_processors_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_processors_with_metadata(
        self,
        response: engine.ListQuantumProcessorsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumProcessorsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_quantum_processors

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_processors_with_metadata`
        interceptor in new development instead of the `post_list_quantum_processors` interceptor.
        When both interceptors are used, this `post_list_quantum_processors_with_metadata` interceptor runs after the
        `post_list_quantum_processors` interceptor. The (possibly modified) response returned by
        `post_list_quantum_processors` will be passed to
        `post_list_quantum_processors_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_programs(
        self,
        request: engine.ListQuantumProgramsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumProgramsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_programs

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_programs(
        self, response: engine.ListQuantumProgramsResponse
    ) -> engine.ListQuantumProgramsResponse:
        """Post-rpc interceptor for list_quantum_programs

        DEPRECATED. Please use the `post_list_quantum_programs_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_programs` interceptor runs
        before the `post_list_quantum_programs_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_programs_with_metadata(
        self,
        response: engine.ListQuantumProgramsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumProgramsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_quantum_programs

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_programs_with_metadata`
        interceptor in new development instead of the `post_list_quantum_programs` interceptor.
        When both interceptors are used, this `post_list_quantum_programs_with_metadata` interceptor runs after the
        `post_list_quantum_programs` interceptor. The (possibly modified) response returned by
        `post_list_quantum_programs` will be passed to
        `post_list_quantum_programs_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_reservation_budgets(
        self,
        request: engine.ListQuantumReservationBudgetsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        engine.ListQuantumReservationBudgetsRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for list_quantum_reservation_budgets

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_reservation_budgets(
        self, response: engine.ListQuantumReservationBudgetsResponse
    ) -> engine.ListQuantumReservationBudgetsResponse:
        """Post-rpc interceptor for list_quantum_reservation_budgets

        DEPRECATED. Please use the `post_list_quantum_reservation_budgets_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_reservation_budgets` interceptor runs
        before the `post_list_quantum_reservation_budgets_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_reservation_budgets_with_metadata(
        self,
        response: engine.ListQuantumReservationBudgetsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        engine.ListQuantumReservationBudgetsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_quantum_reservation_budgets

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_reservation_budgets_with_metadata`
        interceptor in new development instead of the `post_list_quantum_reservation_budgets` interceptor.
        When both interceptors are used, this `post_list_quantum_reservation_budgets_with_metadata` interceptor runs after the
        `post_list_quantum_reservation_budgets` interceptor. The (possibly modified) response returned by
        `post_list_quantum_reservation_budgets` will be passed to
        `post_list_quantum_reservation_budgets_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_reservation_grants(
        self,
        request: engine.ListQuantumReservationGrantsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumReservationGrantsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_reservation_grants

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_reservation_grants(
        self, response: engine.ListQuantumReservationGrantsResponse
    ) -> engine.ListQuantumReservationGrantsResponse:
        """Post-rpc interceptor for list_quantum_reservation_grants

        DEPRECATED. Please use the `post_list_quantum_reservation_grants_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_reservation_grants` interceptor runs
        before the `post_list_quantum_reservation_grants_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_reservation_grants_with_metadata(
        self,
        response: engine.ListQuantumReservationGrantsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        engine.ListQuantumReservationGrantsResponse, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Post-rpc interceptor for list_quantum_reservation_grants

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_reservation_grants_with_metadata`
        interceptor in new development instead of the `post_list_quantum_reservation_grants` interceptor.
        When both interceptors are used, this `post_list_quantum_reservation_grants_with_metadata` interceptor runs after the
        `post_list_quantum_reservation_grants` interceptor. The (possibly modified) response returned by
        `post_list_quantum_reservation_grants` will be passed to
        `post_list_quantum_reservation_grants_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_reservations(
        self,
        request: engine.ListQuantumReservationsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumReservationsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_reservations

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_reservations(
        self, response: engine.ListQuantumReservationsResponse
    ) -> engine.ListQuantumReservationsResponse:
        """Post-rpc interceptor for list_quantum_reservations

        DEPRECATED. Please use the `post_list_quantum_reservations_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_reservations` interceptor runs
        before the `post_list_quantum_reservations_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_reservations_with_metadata(
        self,
        response: engine.ListQuantumReservationsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumReservationsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_quantum_reservations

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_reservations_with_metadata`
        interceptor in new development instead of the `post_list_quantum_reservations` interceptor.
        When both interceptors are used, this `post_list_quantum_reservations_with_metadata` interceptor runs after the
        `post_list_quantum_reservations` interceptor. The (possibly modified) response returned by
        `post_list_quantum_reservations` will be passed to
        `post_list_quantum_reservations_with_metadata`.
        """
        return response, metadata

    def pre_list_quantum_time_slots(
        self,
        request: engine.ListQuantumTimeSlotsRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumTimeSlotsRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for list_quantum_time_slots

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_list_quantum_time_slots(
        self, response: engine.ListQuantumTimeSlotsResponse
    ) -> engine.ListQuantumTimeSlotsResponse:
        """Post-rpc interceptor for list_quantum_time_slots

        DEPRECATED. Please use the `post_list_quantum_time_slots_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_list_quantum_time_slots` interceptor runs
        before the `post_list_quantum_time_slots_with_metadata` interceptor.
        """
        return response

    def post_list_quantum_time_slots_with_metadata(
        self,
        response: engine.ListQuantumTimeSlotsResponse,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.ListQuantumTimeSlotsResponse, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for list_quantum_time_slots

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_list_quantum_time_slots_with_metadata`
        interceptor in new development instead of the `post_list_quantum_time_slots` interceptor.
        When both interceptors are used, this `post_list_quantum_time_slots_with_metadata` interceptor runs after the
        `post_list_quantum_time_slots` interceptor. The (possibly modified) response returned by
        `post_list_quantum_time_slots` will be passed to
        `post_list_quantum_time_slots_with_metadata`.
        """
        return response, metadata

    def pre_reallocate_quantum_reservation_grant(
        self,
        request: engine.ReallocateQuantumReservationGrantRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[
        engine.ReallocateQuantumReservationGrantRequest, Sequence[Tuple[str, Union[str, bytes]]]
    ]:
        """Pre-rpc interceptor for reallocate_quantum_reservation_grant

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_reallocate_quantum_reservation_grant(
        self, response: quantum.QuantumReservationGrant
    ) -> quantum.QuantumReservationGrant:
        """Post-rpc interceptor for reallocate_quantum_reservation_grant

        DEPRECATED. Please use the `post_reallocate_quantum_reservation_grant_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_reallocate_quantum_reservation_grant` interceptor runs
        before the `post_reallocate_quantum_reservation_grant_with_metadata` interceptor.
        """
        return response

    def post_reallocate_quantum_reservation_grant_with_metadata(
        self,
        response: quantum.QuantumReservationGrant,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[quantum.QuantumReservationGrant, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for reallocate_quantum_reservation_grant

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_reallocate_quantum_reservation_grant_with_metadata`
        interceptor in new development instead of the `post_reallocate_quantum_reservation_grant` interceptor.
        When both interceptors are used, this `post_reallocate_quantum_reservation_grant_with_metadata` interceptor runs after the
        `post_reallocate_quantum_reservation_grant` interceptor. The (possibly modified) response returned by
        `post_reallocate_quantum_reservation_grant` will be passed to
        `post_reallocate_quantum_reservation_grant_with_metadata`.
        """
        return response, metadata

    def pre_update_quantum_job(
        self,
        request: engine.UpdateQuantumJobRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.UpdateQuantumJobRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for update_quantum_job

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_update_quantum_job(self, response: quantum.QuantumJob) -> quantum.QuantumJob:
        """Post-rpc interceptor for update_quantum_job

        DEPRECATED. Please use the `post_update_quantum_job_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_update_quantum_job` interceptor runs
        before the `post_update_quantum_job_with_metadata` interceptor.
        """
        return response

    def post_update_quantum_job_with_metadata(
        self, response: quantum.QuantumJob, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[quantum.QuantumJob, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_quantum_job

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_update_quantum_job_with_metadata`
        interceptor in new development instead of the `post_update_quantum_job` interceptor.
        When both interceptors are used, this `post_update_quantum_job_with_metadata` interceptor runs after the
        `post_update_quantum_job` interceptor. The (possibly modified) response returned by
        `post_update_quantum_job` will be passed to
        `post_update_quantum_job_with_metadata`.
        """
        return response, metadata

    def pre_update_quantum_program(
        self,
        request: engine.UpdateQuantumProgramRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.UpdateQuantumProgramRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for update_quantum_program

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_update_quantum_program(
        self, response: quantum.QuantumProgram
    ) -> quantum.QuantumProgram:
        """Post-rpc interceptor for update_quantum_program

        DEPRECATED. Please use the `post_update_quantum_program_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_update_quantum_program` interceptor runs
        before the `post_update_quantum_program_with_metadata` interceptor.
        """
        return response

    def post_update_quantum_program_with_metadata(
        self, response: quantum.QuantumProgram, metadata: Sequence[Tuple[str, Union[str, bytes]]]
    ) -> Tuple[quantum.QuantumProgram, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_quantum_program

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_update_quantum_program_with_metadata`
        interceptor in new development instead of the `post_update_quantum_program` interceptor.
        When both interceptors are used, this `post_update_quantum_program_with_metadata` interceptor runs after the
        `post_update_quantum_program` interceptor. The (possibly modified) response returned by
        `post_update_quantum_program` will be passed to
        `post_update_quantum_program_with_metadata`.
        """
        return response, metadata

    def pre_update_quantum_reservation(
        self,
        request: engine.UpdateQuantumReservationRequest,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[engine.UpdateQuantumReservationRequest, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Pre-rpc interceptor for update_quantum_reservation

        Override in a subclass to manipulate the request or metadata
        before they are sent to the QuantumEngineService server.
        """
        return request, metadata

    def post_update_quantum_reservation(
        self, response: quantum.QuantumReservation
    ) -> quantum.QuantumReservation:
        """Post-rpc interceptor for update_quantum_reservation

        DEPRECATED. Please use the `post_update_quantum_reservation_with_metadata`
        interceptor instead.

        Override in a subclass to read or manipulate the response
        after it is returned by the QuantumEngineService server but before
        it is returned to user code. This `post_update_quantum_reservation` interceptor runs
        before the `post_update_quantum_reservation_with_metadata` interceptor.
        """
        return response

    def post_update_quantum_reservation_with_metadata(
        self,
        response: quantum.QuantumReservation,
        metadata: Sequence[Tuple[str, Union[str, bytes]]],
    ) -> Tuple[quantum.QuantumReservation, Sequence[Tuple[str, Union[str, bytes]]]]:
        """Post-rpc interceptor for update_quantum_reservation

        Override in a subclass to read or manipulate the response or metadata after it
        is returned by the QuantumEngineService server but before it is returned to user code.

        We recommend only using this `post_update_quantum_reservation_with_metadata`
        interceptor in new development instead of the `post_update_quantum_reservation` interceptor.
        When both interceptors are used, this `post_update_quantum_reservation_with_metadata` interceptor runs after the
        `post_update_quantum_reservation` interceptor. The (possibly modified) response returned by
        `post_update_quantum_reservation` will be passed to
        `post_update_quantum_reservation_with_metadata`.
        """
        return response, metadata


@dataclasses.dataclass
class QuantumEngineServiceRestStub:
    _session: AuthorizedSession
    _host: str
    _interceptor: QuantumEngineServiceRestInterceptor


class QuantumEngineServiceRestTransport(_BaseQuantumEngineServiceRestTransport):
    """REST backend synchronous transport for QuantumEngineService.

    -

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = 'quantum.googleapis.com',
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
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
        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            url_scheme=url_scheme,
            api_audience=api_audience,
        )
        self._session = AuthorizedSession(self._credentials, default_host=self.DEFAULT_HOST)
        if client_cert_source_for_mtls:
            self._session.configure_mtls_channel(client_cert_source_for_mtls)
        self._interceptor = interceptor or QuantumEngineServiceRestInterceptor()
        self._prep_wrapped_messages(client_info)

    class _CancelQuantumJob(
        _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumJob, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.CancelQuantumJob")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.CancelQuantumJobRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the cancel quantum job method over HTTP.

            Args:
                request (~.engine.CancelQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumJob._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_quantum_job(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumJob._get_transcoded_request(
                http_options, request
            )

            body = (
                _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumJob._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumJob._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.CancelQuantumJob",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CancelQuantumJob",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._CancelQuantumJob._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request, body
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _CancelQuantumReservation(
        _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumReservation,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.CancelQuantumReservation")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.CancelQuantumReservationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumReservation:
            r"""Call the cancel quantum
            reservation method over HTTP.

                Args:
                    request (~.engine.CancelQuantumReservationRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.quantum.QuantumReservation:
                        -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumReservation._get_http_options()
            )

            request, metadata = self._interceptor.pre_cancel_quantum_reservation(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumReservation._get_transcoded_request(
                http_options, request
            )

            body = _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumReservation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseCancelQuantumReservation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.CancelQuantumReservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CancelQuantumReservation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._CancelQuantumReservation._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request, body
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_cancel_quantum_reservation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumReservation.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.cancel_quantum_reservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CancelQuantumReservation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateQuantumJob(
        _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumJob, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.CreateQuantumJob")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.CreateQuantumJobRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumJob:
            r"""Call the create quantum job method over HTTP.

            Args:
                request (~.engine.CreateQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumJob:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumJob._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_quantum_job(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumJob._get_transcoded_request(
                http_options, request
            )

            body = (
                _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumJob._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumJob._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.CreateQuantumJob",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CreateQuantumJob",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._CreateQuantumJob._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request, body
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_quantum_job_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumJob.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.create_quantum_job",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CreateQuantumJob",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateQuantumProgram(
        _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumProgram,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.CreateQuantumProgram")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.CreateQuantumProgramRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumProgram:
            r"""Call the create quantum program method over HTTP.

            Args:
                request (~.engine.CreateQuantumProgramRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumProgram:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumProgram._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_quantum_program(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumProgram._get_transcoded_request(
                http_options, request
            )

            body = _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumProgram._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumProgram._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.CreateQuantumProgram",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CreateQuantumProgram",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._CreateQuantumProgram._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request, body
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_quantum_program_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumProgram.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.create_quantum_program",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CreateQuantumProgram",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _CreateQuantumReservation(
        _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumReservation,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.CreateQuantumReservation")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.CreateQuantumReservationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumReservation:
            r"""Call the create quantum
            reservation method over HTTP.

                Args:
                    request (~.engine.CreateQuantumReservationRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.quantum.QuantumReservation:
                        -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumReservation._get_http_options()
            )

            request, metadata = self._interceptor.pre_create_quantum_reservation(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumReservation._get_transcoded_request(
                http_options, request
            )

            body = _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumReservation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseCreateQuantumReservation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.CreateQuantumReservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CreateQuantumReservation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._CreateQuantumReservation._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request, body
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_create_quantum_reservation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumReservation.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.create_quantum_reservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "CreateQuantumReservation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _DeleteQuantumJob(
        _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumJob, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.DeleteQuantumJob")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.DeleteQuantumJobRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete quantum job method over HTTP.

            Args:
                request (~.engine.DeleteQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumJob._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_quantum_job(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumJob._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumJob._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.DeleteQuantumJob",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "DeleteQuantumJob",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._DeleteQuantumJob._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteQuantumProgram(
        _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumProgram,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.DeleteQuantumProgram")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.DeleteQuantumProgramRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete quantum program method over HTTP.

            Args:
                request (~.engine.DeleteQuantumProgramRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumProgram._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_quantum_program(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumProgram._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumProgram._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.DeleteQuantumProgram",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "DeleteQuantumProgram",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._DeleteQuantumProgram._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _DeleteQuantumReservation(
        _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumReservation,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.DeleteQuantumReservation")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.DeleteQuantumReservationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ):
            r"""Call the delete quantum
            reservation method over HTTP.

                Args:
                    request (~.engine.DeleteQuantumReservationRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumReservation._get_http_options()
            )

            request, metadata = self._interceptor.pre_delete_quantum_reservation(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumReservation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseDeleteQuantumReservation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = json_format.MessageToJson(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.DeleteQuantumReservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "DeleteQuantumReservation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._DeleteQuantumReservation._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

    class _GetQuantumCalibration(
        _BaseQuantumEngineServiceRestTransport._BaseGetQuantumCalibration,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.GetQuantumCalibration")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.GetQuantumCalibrationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumCalibration:
            r"""Call the get quantum calibration method over HTTP.

            Args:
                request (~.engine.GetQuantumCalibrationRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumCalibration:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumCalibration._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_quantum_calibration(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumCalibration._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumCalibration._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.GetQuantumCalibration",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumCalibration",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._GetQuantumCalibration._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_quantum_calibration_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumCalibration.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.get_quantum_calibration",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumCalibration",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetQuantumJob(
        _BaseQuantumEngineServiceRestTransport._BaseGetQuantumJob, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.GetQuantumJob")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.GetQuantumJobRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumJob:
            r"""Call the get quantum job method over HTTP.

            Args:
                request (~.engine.GetQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumJob:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumJob._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_quantum_job(request, metadata)
            transcoded_request = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumJob._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumJob._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.GetQuantumJob",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumJob",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._GetQuantumJob._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_quantum_job_with_metadata(resp, response_metadata)
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumJob.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.get_quantum_job",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumJob",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetQuantumProcessor(
        _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessor,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.GetQuantumProcessor")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.GetQuantumProcessorRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumProcessor:
            r"""Call the get quantum processor method over HTTP.

            Args:
                request (~.engine.GetQuantumProcessorRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumProcessor:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessor._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_quantum_processor(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessor._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessor._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.GetQuantumProcessor",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumProcessor",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._GetQuantumProcessor._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_quantum_processor_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumProcessor.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.get_quantum_processor",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumProcessor",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetQuantumProcessorConfig(
        _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessorConfig,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.GetQuantumProcessorConfig")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.GetQuantumProcessorConfigRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumProcessorConfig:
            r"""Call the get quantum processor
            config method over HTTP.

                Args:
                    request (~.engine.GetQuantumProcessorConfigRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.quantum.QuantumProcessorConfig:
                        -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessorConfig._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_quantum_processor_config(
                request, metadata
            )
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessorConfig._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessorConfig._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.GetQuantumProcessorConfig",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumProcessorConfig",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._GetQuantumProcessorConfig._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = quantum.QuantumProcessorConfig()
            pb_resp = quantum.QuantumProcessorConfig.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_get_quantum_processor_config(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_quantum_processor_config_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumProcessorConfig.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.get_quantum_processor_config",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumProcessorConfig",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetQuantumProgram(
        _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProgram, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.GetQuantumProgram")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.GetQuantumProgramRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumProgram:
            r"""Call the get quantum program method over HTTP.

            Args:
                request (~.engine.GetQuantumProgramRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumProgram:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProgram._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_quantum_program(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProgram._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProgram._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.GetQuantumProgram",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumProgram",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._GetQuantumProgram._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_quantum_program_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumProgram.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.get_quantum_program",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumProgram",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetQuantumReservation(
        _BaseQuantumEngineServiceRestTransport._BaseGetQuantumReservation,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.GetQuantumReservation")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.GetQuantumReservationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumReservation:
            r"""Call the get quantum reservation method over HTTP.

            Args:
                request (~.engine.GetQuantumReservationRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumReservation:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumReservation._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_quantum_reservation(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumReservation._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumReservation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.GetQuantumReservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumReservation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._GetQuantumReservation._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_quantum_reservation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumReservation.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.get_quantum_reservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumReservation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _GetQuantumResult(
        _BaseQuantumEngineServiceRestTransport._BaseGetQuantumResult, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.GetQuantumResult")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.GetQuantumResultRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumResult:
            r"""Call the get quantum result method over HTTP.

            Args:
                request (~.engine.GetQuantumResultRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumResult:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumResult._get_http_options()
            )

            request, metadata = self._interceptor.pre_get_quantum_result(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseGetQuantumResult._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = (
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumResult._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.GetQuantumResult",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumResult",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._GetQuantumResult._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_get_quantum_result_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumResult.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.get_quantum_result",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "GetQuantumResult",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumCalibrations(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumCalibrations,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumCalibrations")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumCalibrationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumCalibrationsResponse:
            r"""Call the list quantum calibrations method over HTTP.

            Args:
                request (~.engine.ListQuantumCalibrationsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.engine.ListQuantumCalibrationsResponse:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumCalibrations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_calibrations(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumCalibrations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumCalibrations._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumCalibrations",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumCalibrations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._ListQuantumCalibrations._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_calibrations_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumCalibrationsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_calibrations",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumCalibrations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumJobEvents(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumJobEvents,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumJobEvents")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumJobEventsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumJobEventsResponse:
            r"""Call the list quantum job events method over HTTP.

            Args:
                request (~.engine.ListQuantumJobEventsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.engine.ListQuantumJobEventsResponse:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumJobEvents._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_job_events(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumJobEvents._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumJobEvents._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumJobEvents",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumJobEvents",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._ListQuantumJobEvents._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_job_events_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumJobEventsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_job_events",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumJobEvents",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumJobs(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumJobs, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumJobs")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumJobsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumJobsResponse:
            r"""Call the list quantum jobs method over HTTP.

            Args:
                request (~.engine.ListQuantumJobsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.engine.ListQuantumJobsResponse:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumJobs._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_jobs(request, metadata)
            transcoded_request = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumJobs._get_transcoded_request(
                    http_options, request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumJobs._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumJobs",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumJobs",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._ListQuantumJobs._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_jobs_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumJobsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_jobs",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumJobs",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumProcessorConfigs(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumProcessorConfigs,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumProcessorConfigs")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumProcessorConfigsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumProcessorConfigsResponse:
            r"""Call the list quantum processor
            configs method over HTTP.

                Args:
                    request (~.engine.ListQuantumProcessorConfigsRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.engine.ListQuantumProcessorConfigsResponse:
                        -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumProcessorConfigs._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_processor_configs(
                request, metadata
            )
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumProcessorConfigs._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumProcessorConfigs._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumProcessorConfigs",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumProcessorConfigs",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._ListQuantumProcessorConfigs._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
            )

            # In case of error, raise the appropriate core_exceptions.GoogleAPICallError exception
            # subclass.
            if response.status_code >= 400:
                raise core_exceptions.from_http_response(response)

            # Return the response
            resp = engine.ListQuantumProcessorConfigsResponse()
            pb_resp = engine.ListQuantumProcessorConfigsResponse.pb(resp)

            json_format.Parse(response.content, pb_resp, ignore_unknown_fields=True)

            resp = self._interceptor.post_list_quantum_processor_configs(resp)
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_processor_configs_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumProcessorConfigsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_processor_configs",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumProcessorConfigs",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumProcessors(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumProcessors,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumProcessors")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumProcessorsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumProcessorsResponse:
            r"""Call the list quantum processors method over HTTP.

            Args:
                request (~.engine.ListQuantumProcessorsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.engine.ListQuantumProcessorsResponse:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumProcessors._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_processors(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumProcessors._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumProcessors._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumProcessors",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumProcessors",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._ListQuantumProcessors._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_processors_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumProcessorsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_processors",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumProcessors",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumPrograms(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumPrograms,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumPrograms")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumProgramsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumProgramsResponse:
            r"""Call the list quantum programs method over HTTP.

            Args:
                request (~.engine.ListQuantumProgramsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.engine.ListQuantumProgramsResponse:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumPrograms._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_programs(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumPrograms._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumPrograms._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumPrograms",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumPrograms",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._ListQuantumPrograms._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_programs_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumProgramsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_programs",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumPrograms",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumReservationBudgets(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservationBudgets,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumReservationBudgets")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumReservationBudgetsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumReservationBudgetsResponse:
            r"""Call the list quantum reservation
            budgets method over HTTP.

                Args:
                    request (~.engine.ListQuantumReservationBudgetsRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.engine.ListQuantumReservationBudgetsResponse:
                        -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservationBudgets._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_reservation_budgets(
                request, metadata
            )
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservationBudgets._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservationBudgets._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumReservationBudgets",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumReservationBudgets",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                QuantumEngineServiceRestTransport._ListQuantumReservationBudgets._get_response(
                    self._host, metadata, query_params, self._session, timeout, transcoded_request
                )
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_reservation_budgets_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumReservationBudgetsResponse.to_json(
                        response
                    )
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_reservation_budgets",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumReservationBudgets",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumReservationGrants(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservationGrants,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumReservationGrants")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumReservationGrantsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumReservationGrantsResponse:
            r"""Call the list quantum reservation
            grants method over HTTP.

                Args:
                    request (~.engine.ListQuantumReservationGrantsRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.engine.ListQuantumReservationGrantsResponse:
                        -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservationGrants._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_reservation_grants(
                request, metadata
            )
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservationGrants._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservationGrants._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumReservationGrants",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumReservationGrants",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                QuantumEngineServiceRestTransport._ListQuantumReservationGrants._get_response(
                    self._host, metadata, query_params, self._session, timeout, transcoded_request
                )
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_reservation_grants_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumReservationGrantsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_reservation_grants",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumReservationGrants",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumReservations(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservations,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumReservations")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumReservationsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumReservationsResponse:
            r"""Call the list quantum reservations method over HTTP.

            Args:
                request (~.engine.ListQuantumReservationsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.engine.ListQuantumReservationsResponse:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservations._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_reservations(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservations._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumReservations._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumReservations",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumReservations",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._ListQuantumReservations._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_reservations_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumReservationsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_reservations",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumReservations",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _ListQuantumTimeSlots(
        _BaseQuantumEngineServiceRestTransport._BaseListQuantumTimeSlots,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ListQuantumTimeSlots")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
            )
            return response

        def __call__(
            self,
            request: engine.ListQuantumTimeSlotsRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> engine.ListQuantumTimeSlotsResponse:
            r"""Call the list quantum time slots method over HTTP.

            Args:
                request (~.engine.ListQuantumTimeSlotsRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.engine.ListQuantumTimeSlotsResponse:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseListQuantumTimeSlots._get_http_options()
            )

            request, metadata = self._interceptor.pre_list_quantum_time_slots(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseListQuantumTimeSlots._get_transcoded_request(
                http_options, request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseListQuantumTimeSlots._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ListQuantumTimeSlots",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumTimeSlots",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._ListQuantumTimeSlots._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_list_quantum_time_slots_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = engine.ListQuantumTimeSlotsResponse.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.list_quantum_time_slots",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ListQuantumTimeSlots",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _QuantumRunStream(
        _BaseQuantumEngineServiceRestTransport._BaseQuantumRunStream, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.QuantumRunStream")

        def __call__(
            self,
            request: engine.QuantumRunStreamRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> rest_streaming.ResponseIterator:
            raise NotImplementedError(
                "Method QuantumRunStream is not available over REST transport"
            )

    class _ReallocateQuantumReservationGrant(
        _BaseQuantumEngineServiceRestTransport._BaseReallocateQuantumReservationGrant,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.ReallocateQuantumReservationGrant")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.ReallocateQuantumReservationGrantRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumReservationGrant:
            r"""Call the reallocate quantum
            reservation grant method over HTTP.

                Args:
                    request (~.engine.ReallocateQuantumReservationGrantRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.quantum.QuantumReservationGrant:
                        -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseReallocateQuantumReservationGrant._get_http_options()
            )

            request, metadata = self._interceptor.pre_reallocate_quantum_reservation_grant(
                request, metadata
            )
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseReallocateQuantumReservationGrant._get_transcoded_request(
                http_options, request
            )

            body = _BaseQuantumEngineServiceRestTransport._BaseReallocateQuantumReservationGrant._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseReallocateQuantumReservationGrant._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.ReallocateQuantumReservationGrant",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ReallocateQuantumReservationGrant",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = (
                QuantumEngineServiceRestTransport._ReallocateQuantumReservationGrant._get_response(
                    self._host,
                    metadata,
                    query_params,
                    self._session,
                    timeout,
                    transcoded_request,
                    body,
                )
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_reallocate_quantum_reservation_grant_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumReservationGrant.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.reallocate_quantum_reservation_grant",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "ReallocateQuantumReservationGrant",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateQuantumJob(
        _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumJob, QuantumEngineServiceRestStub
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.UpdateQuantumJob")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.UpdateQuantumJobRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumJob:
            r"""Call the update quantum job method over HTTP.

            Args:
                request (~.engine.UpdateQuantumJobRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumJob:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumJob._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_quantum_job(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumJob._get_transcoded_request(
                http_options, request
            )

            body = (
                _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumJob._get_request_body_json(
                    transcoded_request
                )
            )

            # Jsonify the query params
            query_params = (
                _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumJob._get_query_params_json(
                    transcoded_request
                )
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.UpdateQuantumJob",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "UpdateQuantumJob",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._UpdateQuantumJob._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request, body
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_quantum_job_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumJob.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.update_quantum_job",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "UpdateQuantumJob",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateQuantumProgram(
        _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumProgram,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.UpdateQuantumProgram")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.UpdateQuantumProgramRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumProgram:
            r"""Call the update quantum program method over HTTP.

            Args:
                request (~.engine.UpdateQuantumProgramRequest):
                    The request object. -
                retry (google.api_core.retry.Retry): Designation of what errors, if any,
                    should be retried.
                timeout (float): The timeout for this request.
                metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                    sent along with the request as metadata. Normally, each value must be of type `str`,
                    but for metadata keys ending with the suffix `-bin`, the corresponding values must
                    be of type `bytes`.

            Returns:
                ~.quantum.QuantumProgram:
                    -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumProgram._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_quantum_program(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumProgram._get_transcoded_request(
                http_options, request
            )

            body = _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumProgram._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumProgram._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.UpdateQuantumProgram",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "UpdateQuantumProgram",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._UpdateQuantumProgram._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request, body
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_quantum_program_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumProgram.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.update_quantum_program",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "UpdateQuantumProgram",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    class _UpdateQuantumReservation(
        _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumReservation,
        QuantumEngineServiceRestStub,
    ):
        def __hash__(self):
            return hash("QuantumEngineServiceRestTransport.UpdateQuantumReservation")

        @staticmethod
        def _get_response(
            host, metadata, query_params, session, timeout, transcoded_request, body=None
        ):

            uri = transcoded_request['uri']
            method = transcoded_request['method']
            headers = dict(metadata)
            headers['Content-Type'] = 'application/json'
            response = getattr(session, method)(
                "{host}{uri}".format(host=host, uri=uri),
                timeout=timeout,
                headers=headers,
                params=rest_helpers.flatten_query_params(query_params, strict=True),
                data=body,
            )
            return response

        def __call__(
            self,
            request: engine.UpdateQuantumReservationRequest,
            *,
            retry: OptionalRetry = gapic_v1.method.DEFAULT,
            timeout: Optional[float] = None,
            metadata: Sequence[Tuple[str, Union[str, bytes]]] = (),
        ) -> quantum.QuantumReservation:
            r"""Call the update quantum
            reservation method over HTTP.

                Args:
                    request (~.engine.UpdateQuantumReservationRequest):
                        The request object. -
                    retry (google.api_core.retry.Retry): Designation of what errors, if any,
                        should be retried.
                    timeout (float): The timeout for this request.
                    metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                        sent along with the request as metadata. Normally, each value must be of type `str`,
                        but for metadata keys ending with the suffix `-bin`, the corresponding values must
                        be of type `bytes`.

                Returns:
                    ~.quantum.QuantumReservation:
                        -
            """

            http_options = (
                _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumReservation._get_http_options()
            )

            request, metadata = self._interceptor.pre_update_quantum_reservation(request, metadata)
            transcoded_request = _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumReservation._get_transcoded_request(
                http_options, request
            )

            body = _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumReservation._get_request_body_json(
                transcoded_request
            )

            # Jsonify the query params
            query_params = _BaseQuantumEngineServiceRestTransport._BaseUpdateQuantumReservation._get_query_params_json(
                transcoded_request
            )

            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                request_url = "{host}{uri}".format(host=self._host, uri=transcoded_request['uri'])
                method = transcoded_request['method']
                try:
                    request_payload = type(request).to_json(request)
                except:
                    request_payload = None
                http_request = {
                    "payload": request_payload,
                    "requestMethod": method,
                    "requestUrl": request_url,
                    "headers": dict(metadata),
                }
                _LOGGER.debug(
                    f"Sending request for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.UpdateQuantumReservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "UpdateQuantumReservation",
                        "httpRequest": http_request,
                        "metadata": http_request["headers"],
                    },
                )

            # Send the request
            response = QuantumEngineServiceRestTransport._UpdateQuantumReservation._get_response(
                self._host, metadata, query_params, self._session, timeout, transcoded_request, body
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
            response_metadata = [(k, str(v)) for k, v in response.headers.items()]
            resp, _ = self._interceptor.post_update_quantum_reservation_with_metadata(
                resp, response_metadata
            )
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(logging.DEBUG):  # pragma: NO COVER
                try:
                    response_payload = quantum.QuantumReservation.to_json(response)
                except:
                    response_payload = None
                http_response = {
                    "payload": response_payload,
                    "headers": dict(response.headers),
                    "status": response.status_code,
                }
                _LOGGER.debug(
                    "Received response for cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient.update_quantum_reservation",
                    extra={
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "rpcName": "UpdateQuantumReservation",
                        "metadata": http_response["headers"],
                        "httpResponse": http_response,
                    },
                )
            return resp

    @property
    def cancel_quantum_job(self) -> Callable[[engine.CancelQuantumJobRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CancelQuantumJob(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def cancel_quantum_reservation(
        self,
    ) -> Callable[[engine.CancelQuantumReservationRequest], quantum.QuantumReservation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CancelQuantumReservation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_quantum_job(self) -> Callable[[engine.CreateQuantumJobRequest], quantum.QuantumJob]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateQuantumJob(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_quantum_program(
        self,
    ) -> Callable[[engine.CreateQuantumProgramRequest], quantum.QuantumProgram]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateQuantumProgram(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def create_quantum_reservation(
        self,
    ) -> Callable[[engine.CreateQuantumReservationRequest], quantum.QuantumReservation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._CreateQuantumReservation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_quantum_job(self) -> Callable[[engine.DeleteQuantumJobRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteQuantumJob(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_quantum_program(
        self,
    ) -> Callable[[engine.DeleteQuantumProgramRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteQuantumProgram(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def delete_quantum_reservation(
        self,
    ) -> Callable[[engine.DeleteQuantumReservationRequest], empty_pb2.Empty]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._DeleteQuantumReservation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_quantum_calibration(
        self,
    ) -> Callable[[engine.GetQuantumCalibrationRequest], quantum.QuantumCalibration]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumCalibration(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_quantum_job(self) -> Callable[[engine.GetQuantumJobRequest], quantum.QuantumJob]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumJob(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_quantum_processor(
        self,
    ) -> Callable[[engine.GetQuantumProcessorRequest], quantum.QuantumProcessor]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumProcessor(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_quantum_processor_config(
        self,
    ) -> Callable[[engine.GetQuantumProcessorConfigRequest], quantum.QuantumProcessorConfig]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumProcessorConfig(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_quantum_program(
        self,
    ) -> Callable[[engine.GetQuantumProgramRequest], quantum.QuantumProgram]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumProgram(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_quantum_reservation(
        self,
    ) -> Callable[[engine.GetQuantumReservationRequest], quantum.QuantumReservation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumReservation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def get_quantum_result(
        self,
    ) -> Callable[[engine.GetQuantumResultRequest], quantum.QuantumResult]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._GetQuantumResult(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_calibrations(
        self,
    ) -> Callable[[engine.ListQuantumCalibrationsRequest], engine.ListQuantumCalibrationsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumCalibrations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_job_events(
        self,
    ) -> Callable[[engine.ListQuantumJobEventsRequest], engine.ListQuantumJobEventsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumJobEvents(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_jobs(
        self,
    ) -> Callable[[engine.ListQuantumJobsRequest], engine.ListQuantumJobsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumJobs(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_processor_configs(
        self,
    ) -> Callable[
        [engine.ListQuantumProcessorConfigsRequest], engine.ListQuantumProcessorConfigsResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumProcessorConfigs(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_processors(
        self,
    ) -> Callable[[engine.ListQuantumProcessorsRequest], engine.ListQuantumProcessorsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumProcessors(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_programs(
        self,
    ) -> Callable[[engine.ListQuantumProgramsRequest], engine.ListQuantumProgramsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumPrograms(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_reservation_budgets(
        self,
    ) -> Callable[
        [engine.ListQuantumReservationBudgetsRequest], engine.ListQuantumReservationBudgetsResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumReservationBudgets(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_reservation_grants(
        self,
    ) -> Callable[
        [engine.ListQuantumReservationGrantsRequest], engine.ListQuantumReservationGrantsResponse
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumReservationGrants(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_reservations(
        self,
    ) -> Callable[[engine.ListQuantumReservationsRequest], engine.ListQuantumReservationsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumReservations(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def list_quantum_time_slots(
        self,
    ) -> Callable[[engine.ListQuantumTimeSlotsRequest], engine.ListQuantumTimeSlotsResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ListQuantumTimeSlots(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def quantum_run_stream(
        self,
    ) -> Callable[[engine.QuantumRunStreamRequest], engine.QuantumRunStreamResponse]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._QuantumRunStream(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def reallocate_quantum_reservation_grant(
        self,
    ) -> Callable[
        [engine.ReallocateQuantumReservationGrantRequest], quantum.QuantumReservationGrant
    ]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._ReallocateQuantumReservationGrant(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_quantum_job(self) -> Callable[[engine.UpdateQuantumJobRequest], quantum.QuantumJob]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateQuantumJob(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_quantum_program(
        self,
    ) -> Callable[[engine.UpdateQuantumProgramRequest], quantum.QuantumProgram]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateQuantumProgram(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def update_quantum_reservation(
        self,
    ) -> Callable[[engine.UpdateQuantumReservationRequest], quantum.QuantumReservation]:
        # The return type is fine, but mypy isn't sophisticated enough to determine what's going on here.
        # In C++ this would require a dynamic_cast
        return self._UpdateQuantumReservation(self._session, self._host, self._interceptor)  # type: ignore

    @property
    def kind(self) -> str:
        return "rest"

    def close(self):
        self._session.close()


__all__ = ('QuantumEngineServiceRestTransport',)
