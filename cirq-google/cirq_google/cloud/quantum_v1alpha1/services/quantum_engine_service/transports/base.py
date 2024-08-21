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
import abc
import importlib.metadata
from typing import Awaitable, Callable, Dict, Optional, Sequence, Union

import google.auth
import google.api_core
from google.api_core import exceptions as core_exceptions
from google.api_core import gapic_v1
from google.auth import credentials as ga_credentials
from google.oauth2 import service_account

from cirq_google.cloud.quantum_v1alpha1.types import engine
from cirq_google.cloud.quantum_v1alpha1.types import quantum
from google.protobuf import empty_pb2

try:
    DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
        gapic_version=importlib.metadata.version("google-cloud-quantum")
    )
except ModuleNotFoundError:
    DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo()


class QuantumEngineServiceTransport(abc.ABC):
    """Abstract transport class for QuantumEngineService."""

    AUTH_SCOPES = ('https://www.googleapis.com/auth/cloud-platform',)

    DEFAULT_HOST: str = 'quantum.googleapis.com'

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        **kwargs,
    ) -> None:
        """Instantiate the transport.

        Args:
            host (Optional[str]):
                 The hostname to connect to.
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A list of scopes.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.
        """
        # Save the hostname. Default to port 443 (HTTPS) if none is specified.
        if ':' not in host:
            host += ':443'
        self._host = host

        scopes_kwargs = {"scopes": scopes, "default_scopes": self.AUTH_SCOPES}

        # Save the scopes.
        self._scopes = scopes

        # If no credentials are provided, then determine the appropriate
        # defaults.
        if credentials and credentials_file:
            raise core_exceptions.DuplicateCredentialArgs(
                "'credentials_file' and 'credentials' are mutually exclusive"
            )

        if credentials_file is not None:
            credentials, _ = google.auth.load_credentials_from_file(
                credentials_file, **scopes_kwargs, quota_project_id=quota_project_id
            )
        elif credentials is None:
            credentials, _ = google.auth.default(**scopes_kwargs, quota_project_id=quota_project_id)

        # If the credentials are service account credentials, then always try to use self signed JWT.
        if (
            always_use_jwt_access
            and isinstance(credentials, service_account.Credentials)
            and hasattr(service_account.Credentials, "with_always_use_jwt_access")
        ):
            credentials = credentials.with_always_use_jwt_access(True)

        # Save the credentials.
        self._credentials = credentials

    def _prep_wrapped_messages(self, client_info):
        # Precompute the wrapped methods.
        self._wrapped_methods = {
            self.create_quantum_program: gapic_v1.method.wrap_method(
                self.create_quantum_program, default_timeout=60.0, client_info=client_info
            ),
            self.get_quantum_program: gapic_v1.method.wrap_method(
                self.get_quantum_program, default_timeout=60.0, client_info=client_info
            ),
            self.list_quantum_programs: gapic_v1.method.wrap_method(
                self.list_quantum_programs, default_timeout=60.0, client_info=client_info
            ),
            self.delete_quantum_program: gapic_v1.method.wrap_method(
                self.delete_quantum_program, default_timeout=60.0, client_info=client_info
            ),
            self.update_quantum_program: gapic_v1.method.wrap_method(
                self.update_quantum_program, default_timeout=60.0, client_info=client_info
            ),
            self.create_quantum_job: gapic_v1.method.wrap_method(
                self.create_quantum_job, default_timeout=60.0, client_info=client_info
            ),
            self.get_quantum_job: gapic_v1.method.wrap_method(
                self.get_quantum_job, default_timeout=60.0, client_info=client_info
            ),
            self.list_quantum_jobs: gapic_v1.method.wrap_method(
                self.list_quantum_jobs, default_timeout=60.0, client_info=client_info
            ),
            self.delete_quantum_job: gapic_v1.method.wrap_method(
                self.delete_quantum_job, default_timeout=60.0, client_info=client_info
            ),
            self.update_quantum_job: gapic_v1.method.wrap_method(
                self.update_quantum_job, default_timeout=60.0, client_info=client_info
            ),
            self.cancel_quantum_job: gapic_v1.method.wrap_method(
                self.cancel_quantum_job, default_timeout=None, client_info=client_info
            ),
            self.list_quantum_job_events: gapic_v1.method.wrap_method(
                self.list_quantum_job_events, default_timeout=60.0, client_info=client_info
            ),
            self.get_quantum_result: gapic_v1.method.wrap_method(
                self.get_quantum_result, default_timeout=60.0, client_info=client_info
            ),
            self.list_quantum_processors: gapic_v1.method.wrap_method(
                self.list_quantum_processors, default_timeout=60.0, client_info=client_info
            ),
            self.get_quantum_processor: gapic_v1.method.wrap_method(
                self.get_quantum_processor, default_timeout=60.0, client_info=client_info
            ),
            self.list_quantum_calibrations: gapic_v1.method.wrap_method(
                self.list_quantum_calibrations, default_timeout=60.0, client_info=client_info
            ),
            self.get_quantum_calibration: gapic_v1.method.wrap_method(
                self.get_quantum_calibration, default_timeout=60.0, client_info=client_info
            ),
            self.create_quantum_reservation: gapic_v1.method.wrap_method(
                self.create_quantum_reservation, default_timeout=60.0, client_info=client_info
            ),
            self.cancel_quantum_reservation: gapic_v1.method.wrap_method(
                self.cancel_quantum_reservation, default_timeout=60.0, client_info=client_info
            ),
            self.delete_quantum_reservation: gapic_v1.method.wrap_method(
                self.delete_quantum_reservation, default_timeout=60.0, client_info=client_info
            ),
            self.get_quantum_reservation: gapic_v1.method.wrap_method(
                self.get_quantum_reservation, default_timeout=60.0, client_info=client_info
            ),
            self.list_quantum_reservations: gapic_v1.method.wrap_method(
                self.list_quantum_reservations, default_timeout=60.0, client_info=client_info
            ),
            self.update_quantum_reservation: gapic_v1.method.wrap_method(
                self.update_quantum_reservation, default_timeout=60.0, client_info=client_info
            ),
            self.quantum_run_stream: gapic_v1.method.wrap_method(
                self.quantum_run_stream, default_timeout=60.0, client_info=client_info
            ),
            self.list_quantum_reservation_grants: gapic_v1.method.wrap_method(
                self.list_quantum_reservation_grants, default_timeout=60.0, client_info=client_info
            ),
            self.reallocate_quantum_reservation_grant: gapic_v1.method.wrap_method(
                self.reallocate_quantum_reservation_grant,
                default_timeout=60.0,
                client_info=client_info,
            ),
            self.list_quantum_reservation_budgets: gapic_v1.method.wrap_method(
                self.list_quantum_reservation_budgets, default_timeout=60.0, client_info=client_info
            ),
            self.list_quantum_time_slots: gapic_v1.method.wrap_method(
                self.list_quantum_time_slots, default_timeout=60.0, client_info=client_info
            ),
        }

    def close(self):
        """Closes resources associated with the transport.

        .. warning::
             Only call this method if the transport is NOT shared
             with other clients - this may cause errors in other clients!
        """
        raise NotImplementedError()

    @property
    def create_quantum_program(
        self,
    ) -> Callable[
        [engine.CreateQuantumProgramRequest],
        Union[quantum.QuantumProgram, Awaitable[quantum.QuantumProgram]],
    ]:
        raise NotImplementedError()

    @property
    def get_quantum_program(
        self,
    ) -> Callable[
        [engine.GetQuantumProgramRequest],
        Union[quantum.QuantumProgram, Awaitable[quantum.QuantumProgram]],
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_programs(
        self,
    ) -> Callable[
        [engine.ListQuantumProgramsRequest],
        Union[engine.ListQuantumProgramsResponse, Awaitable[engine.ListQuantumProgramsResponse]],
    ]:
        raise NotImplementedError()

    @property
    def delete_quantum_program(
        self,
    ) -> Callable[
        [engine.DeleteQuantumProgramRequest], Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]]
    ]:
        raise NotImplementedError()

    @property
    def update_quantum_program(
        self,
    ) -> Callable[
        [engine.UpdateQuantumProgramRequest],
        Union[quantum.QuantumProgram, Awaitable[quantum.QuantumProgram]],
    ]:
        raise NotImplementedError()

    @property
    def create_quantum_job(
        self,
    ) -> Callable[
        [engine.CreateQuantumJobRequest], Union[quantum.QuantumJob, Awaitable[quantum.QuantumJob]]
    ]:
        raise NotImplementedError()

    @property
    def get_quantum_job(
        self,
    ) -> Callable[
        [engine.GetQuantumJobRequest], Union[quantum.QuantumJob, Awaitable[quantum.QuantumJob]]
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_jobs(
        self,
    ) -> Callable[
        [engine.ListQuantumJobsRequest],
        Union[engine.ListQuantumJobsResponse, Awaitable[engine.ListQuantumJobsResponse]],
    ]:
        raise NotImplementedError()

    @property
    def delete_quantum_job(
        self,
    ) -> Callable[
        [engine.DeleteQuantumJobRequest], Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]]
    ]:
        raise NotImplementedError()

    @property
    def update_quantum_job(
        self,
    ) -> Callable[
        [engine.UpdateQuantumJobRequest], Union[quantum.QuantumJob, Awaitable[quantum.QuantumJob]]
    ]:
        raise NotImplementedError()

    @property
    def cancel_quantum_job(
        self,
    ) -> Callable[
        [engine.CancelQuantumJobRequest], Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]]
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_job_events(
        self,
    ) -> Callable[
        [engine.ListQuantumJobEventsRequest],
        Union[engine.ListQuantumJobEventsResponse, Awaitable[engine.ListQuantumJobEventsResponse]],
    ]:
        raise NotImplementedError()

    @property
    def get_quantum_result(
        self,
    ) -> Callable[
        [engine.GetQuantumResultRequest],
        Union[quantum.QuantumResult, Awaitable[quantum.QuantumResult]],
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_processors(
        self,
    ) -> Callable[
        [engine.ListQuantumProcessorsRequest],
        Union[
            engine.ListQuantumProcessorsResponse, Awaitable[engine.ListQuantumProcessorsResponse]
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_quantum_processor(
        self,
    ) -> Callable[
        [engine.GetQuantumProcessorRequest],
        Union[quantum.QuantumProcessor, Awaitable[quantum.QuantumProcessor]],
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_calibrations(
        self,
    ) -> Callable[
        [engine.ListQuantumCalibrationsRequest],
        Union[
            engine.ListQuantumCalibrationsResponse,
            Awaitable[engine.ListQuantumCalibrationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def get_quantum_calibration(
        self,
    ) -> Callable[
        [engine.GetQuantumCalibrationRequest],
        Union[quantum.QuantumCalibration, Awaitable[quantum.QuantumCalibration]],
    ]:
        raise NotImplementedError()

    @property
    def create_quantum_reservation(
        self,
    ) -> Callable[
        [engine.CreateQuantumReservationRequest],
        Union[quantum.QuantumReservation, Awaitable[quantum.QuantumReservation]],
    ]:
        raise NotImplementedError()

    @property
    def cancel_quantum_reservation(
        self,
    ) -> Callable[
        [engine.CancelQuantumReservationRequest],
        Union[quantum.QuantumReservation, Awaitable[quantum.QuantumReservation]],
    ]:
        raise NotImplementedError()

    @property
    def delete_quantum_reservation(
        self,
    ) -> Callable[
        [engine.DeleteQuantumReservationRequest], Union[empty_pb2.Empty, Awaitable[empty_pb2.Empty]]
    ]:
        raise NotImplementedError()

    @property
    def get_quantum_reservation(
        self,
    ) -> Callable[
        [engine.GetQuantumReservationRequest],
        Union[quantum.QuantumReservation, Awaitable[quantum.QuantumReservation]],
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_reservations(
        self,
    ) -> Callable[
        [engine.ListQuantumReservationsRequest],
        Union[
            engine.ListQuantumReservationsResponse,
            Awaitable[engine.ListQuantumReservationsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def update_quantum_reservation(
        self,
    ) -> Callable[
        [engine.UpdateQuantumReservationRequest],
        Union[quantum.QuantumReservation, Awaitable[quantum.QuantumReservation]],
    ]:
        raise NotImplementedError()

    @property
    def quantum_run_stream(
        self,
    ) -> Callable[
        [engine.QuantumRunStreamRequest],
        Union[engine.QuantumRunStreamResponse, Awaitable[engine.QuantumRunStreamResponse]],
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_reservation_grants(
        self,
    ) -> Callable[
        [engine.ListQuantumReservationGrantsRequest],
        Union[
            engine.ListQuantumReservationGrantsResponse,
            Awaitable[engine.ListQuantumReservationGrantsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def reallocate_quantum_reservation_grant(
        self,
    ) -> Callable[
        [engine.ReallocateQuantumReservationGrantRequest],
        Union[quantum.QuantumReservationGrant, Awaitable[quantum.QuantumReservationGrant]],
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_reservation_budgets(
        self,
    ) -> Callable[
        [engine.ListQuantumReservationBudgetsRequest],
        Union[
            engine.ListQuantumReservationBudgetsResponse,
            Awaitable[engine.ListQuantumReservationBudgetsResponse],
        ],
    ]:
        raise NotImplementedError()

    @property
    def list_quantum_time_slots(
        self,
    ) -> Callable[
        [engine.ListQuantumTimeSlotsRequest],
        Union[engine.ListQuantumTimeSlotsResponse, Awaitable[engine.ListQuantumTimeSlotsResponse]],
    ]:
        raise NotImplementedError()


__all__ = ('QuantumEngineServiceTransport',)
