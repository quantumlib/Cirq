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
import warnings
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import google.auth
import grpc  # type: ignore
from google.api_core import gapic_v1, grpc_helpers
from google.auth import credentials as ga_credentials
from google.auth.transport.grpc import SslCredentials
from google.protobuf import empty_pb2

from cirq_google.cloud.quantum_v1alpha1.types import engine, quantum

from .base import DEFAULT_CLIENT_INFO, QuantumEngineServiceTransport


class QuantumEngineServiceGrpcTransport(QuantumEngineServiceTransport):
    """gRPC backend transport for QuantumEngineService.

    -

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends protocol buffers over the wire using gRPC (which is built on
    top of HTTP/2); the ``grpcio`` package must be installed.
    """

    _stubs: Dict[str, Callable]

    def __init__(
        self,
        *,
        host: str = 'quantum.googleapis.com',
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        channel: Optional[grpc.Channel] = None,
        api_mtls_endpoint: Optional[str] = None,
        client_cert_source: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        ssl_channel_credentials: Optional[grpc.ChannelCredentials] = None,
        client_cert_source_for_mtls: Optional[Callable[[], Tuple[bytes, bytes]]] = None,
        quota_project_id: Optional[str] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
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
                This argument is ignored if ``channel`` is provided.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is ignored if ``channel`` is provided.
            scopes (Optional(Sequence[str])): A list of scopes. This argument is
                ignored if ``channel`` is provided.
            channel (Optional[grpc.Channel]): A ``Channel`` instance through
                which to make calls.
            api_mtls_endpoint (Optional[str]): Deprecated. The mutual TLS endpoint.
                If provided, it overrides the ``host`` argument and tries to create
                a mutual TLS channel with client SSL credentials from
                ``client_cert_source`` or application default SSL credentials.
            client_cert_source (Optional[Callable[[], Tuple[bytes, bytes]]]):
                Deprecated. A callback to provide client SSL certificate bytes and
                private key bytes, both in PEM format. It is ignored if
                ``api_mtls_endpoint`` is None.
            ssl_channel_credentials (grpc.ChannelCredentials): SSL credentials
                for the grpc channel. It is ignored if ``channel`` is provided.
            client_cert_source_for_mtls (Optional[Callable[[], Tuple[bytes, bytes]]]):
                A callback to provide client certificate bytes and private key bytes,
                both in PEM format. It is used to configure a mutual TLS channel. It is
                ignored if ``channel`` or ``ssl_channel_credentials`` is provided.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            always_use_jwt_access (Optional[bool]): Whether self signed JWT should
                be used for service account credentials.

        Raises:
          google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
              creation failed for any reason.
          google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """
        self._grpc_channel = None
        self._ssl_channel_credentials = ssl_channel_credentials
        self._stubs: Dict[str, Callable] = {}

        if api_mtls_endpoint:
            warnings.warn("api_mtls_endpoint is deprecated", DeprecationWarning)
        if client_cert_source:
            warnings.warn("client_cert_source is deprecated", DeprecationWarning)

        if channel:
            # Ignore credentials if a channel was passed.
            credentials = None
            # If a channel was explicitly provided, set it.
            self._grpc_channel = channel
            self._ssl_channel_credentials = None

        else:
            if api_mtls_endpoint:
                host = api_mtls_endpoint

                # Create SSL credentials with client_cert_source or application
                # default SSL credentials.
                if client_cert_source:
                    cert, key = client_cert_source()
                    self._ssl_channel_credentials = grpc.ssl_channel_credentials(
                        certificate_chain=cert, private_key=key
                    )
                else:
                    self._ssl_channel_credentials = SslCredentials().ssl_credentials

            else:
                if client_cert_source_for_mtls and not ssl_channel_credentials:
                    cert, key = client_cert_source_for_mtls()
                    self._ssl_channel_credentials = grpc.ssl_channel_credentials(
                        certificate_chain=cert, private_key=key
                    )

        # The base transport sets the host, credentials and scopes
        super().__init__(
            host=host,
            credentials=credentials,
            credentials_file=credentials_file,
            scopes=scopes,
            quota_project_id=quota_project_id,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
        )

        if not self._grpc_channel:
            self._grpc_channel = type(self).create_channel(
                self._host,
                # use the credentials which are saved
                credentials=self._credentials,
                # Set ``credentials_file`` to ``None`` here as
                # the credentials that we saved earlier should be used.
                credentials_file=None,
                scopes=self._scopes,
                ssl_credentials=self._ssl_channel_credentials,
                quota_project_id=quota_project_id,
                options=[
                    ('grpc.max_send_message_length', 20 * 1024 * 1024),  # 20MiB
                    ('grpc.max_receive_message_length', -1),  # unlimited
                    ('grpc.max_metadata_length', 10 * 1024 * 1024),  # 10MiB
                ],
            )

        # Wrap messages. This must be done after self._grpc_channel exists
        self._prep_wrapped_messages(client_info)

    @classmethod
    def create_channel(
        cls,
        host: str = 'quantum.googleapis.com',
        credentials: Optional[ga_credentials.Credentials] = None,
        credentials_file: Optional[str] = None,
        scopes: Optional[Sequence[str]] = None,
        quota_project_id: Optional[str] = None,
        **kwargs,
    ) -> grpc.Channel:
        """Create and return a gRPC channel object.
        Args:
            host (Optional[str]): The host for the channel to use.
            credentials (Optional[~.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If
                none are specified, the client will attempt to ascertain
                the credentials from the environment.
            credentials_file (Optional[str]): A file with credentials that can
                be loaded with :func:`google.auth.load_credentials_from_file`.
                This argument is mutually exclusive with credentials.
            scopes (Optional[Sequence[str]]): A optional list of scopes needed for this
                service. These are only used when credentials are not specified and
                are passed to :func:`google.auth.default`.
            quota_project_id (Optional[str]): An optional project to use for billing
                and quota.
            kwargs (Optional[dict]): Keyword arguments, which are passed to the
                channel creation.
        Returns:
            grpc.Channel: A gRPC channel object.

        Raises:
            google.api_core.exceptions.DuplicateCredentialArgs: If both ``credentials``
              and ``credentials_file`` are passed.
        """

        return grpc_helpers.create_channel(
            host,
            credentials=credentials,
            credentials_file=credentials_file,
            quota_project_id=quota_project_id,
            default_scopes=cls.AUTH_SCOPES,
            scopes=scopes,
            default_host=cls.DEFAULT_HOST,
            **kwargs,
        )

    @property
    def grpc_channel(self) -> grpc.Channel:
        """Return the channel designed to connect to this service."""
        return self._grpc_channel

    @property
    def create_quantum_program(
        self,
    ) -> Callable[[engine.CreateQuantumProgramRequest], quantum.QuantumProgram]:
        r"""Return a callable for the create quantum program method over gRPC.

        -

        Returns:
            Callable[[~.CreateQuantumProgramRequest],
                    ~.QuantumProgram]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'create_quantum_program' not in self._stubs:
            self._stubs['create_quantum_program'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/CreateQuantumProgram',
                request_serializer=engine.CreateQuantumProgramRequest.serialize,
                response_deserializer=quantum.QuantumProgram.deserialize,
            )
        return self._stubs['create_quantum_program']

    @property
    def get_quantum_program(
        self,
    ) -> Callable[[engine.GetQuantumProgramRequest], quantum.QuantumProgram]:
        r"""Return a callable for the get quantum program method over gRPC.

        -

        Returns:
            Callable[[~.GetQuantumProgramRequest],
                    ~.QuantumProgram]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'get_quantum_program' not in self._stubs:
            self._stubs['get_quantum_program'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/GetQuantumProgram',
                request_serializer=engine.GetQuantumProgramRequest.serialize,
                response_deserializer=quantum.QuantumProgram.deserialize,
            )
        return self._stubs['get_quantum_program']

    @property
    def list_quantum_programs(
        self,
    ) -> Callable[[engine.ListQuantumProgramsRequest], engine.ListQuantumProgramsResponse]:
        r"""Return a callable for the list quantum programs method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumProgramsRequest],
                    ~.ListQuantumProgramsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_programs' not in self._stubs:
            self._stubs['list_quantum_programs'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumPrograms',
                request_serializer=engine.ListQuantumProgramsRequest.serialize,
                response_deserializer=engine.ListQuantumProgramsResponse.deserialize,
            )
        return self._stubs['list_quantum_programs']

    @property
    def delete_quantum_program(
        self,
    ) -> Callable[[engine.DeleteQuantumProgramRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete quantum program method over gRPC.

        -

        Returns:
            Callable[[~.DeleteQuantumProgramRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'delete_quantum_program' not in self._stubs:
            self._stubs['delete_quantum_program'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/DeleteQuantumProgram',
                request_serializer=engine.DeleteQuantumProgramRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs['delete_quantum_program']

    @property
    def update_quantum_program(
        self,
    ) -> Callable[[engine.UpdateQuantumProgramRequest], quantum.QuantumProgram]:
        r"""Return a callable for the update quantum program method over gRPC.

        -

        Returns:
            Callable[[~.UpdateQuantumProgramRequest],
                    ~.QuantumProgram]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'update_quantum_program' not in self._stubs:
            self._stubs['update_quantum_program'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/UpdateQuantumProgram',
                request_serializer=engine.UpdateQuantumProgramRequest.serialize,
                response_deserializer=quantum.QuantumProgram.deserialize,
            )
        return self._stubs['update_quantum_program']

    @property
    def create_quantum_job(self) -> Callable[[engine.CreateQuantumJobRequest], quantum.QuantumJob]:
        r"""Return a callable for the create quantum job method over gRPC.

        -

        Returns:
            Callable[[~.CreateQuantumJobRequest],
                    ~.QuantumJob]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'create_quantum_job' not in self._stubs:
            self._stubs['create_quantum_job'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/CreateQuantumJob',
                request_serializer=engine.CreateQuantumJobRequest.serialize,
                response_deserializer=quantum.QuantumJob.deserialize,
            )
        return self._stubs['create_quantum_job']

    @property
    def get_quantum_job(self) -> Callable[[engine.GetQuantumJobRequest], quantum.QuantumJob]:
        r"""Return a callable for the get quantum job method over gRPC.

        -

        Returns:
            Callable[[~.GetQuantumJobRequest],
                    ~.QuantumJob]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'get_quantum_job' not in self._stubs:
            self._stubs['get_quantum_job'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/GetQuantumJob',
                request_serializer=engine.GetQuantumJobRequest.serialize,
                response_deserializer=quantum.QuantumJob.deserialize,
            )
        return self._stubs['get_quantum_job']

    @property
    def list_quantum_jobs(
        self,
    ) -> Callable[[engine.ListQuantumJobsRequest], engine.ListQuantumJobsResponse]:
        r"""Return a callable for the list quantum jobs method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumJobsRequest],
                    ~.ListQuantumJobsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_jobs' not in self._stubs:
            self._stubs['list_quantum_jobs'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumJobs',
                request_serializer=engine.ListQuantumJobsRequest.serialize,
                response_deserializer=engine.ListQuantumJobsResponse.deserialize,
            )
        return self._stubs['list_quantum_jobs']

    @property
    def delete_quantum_job(self) -> Callable[[engine.DeleteQuantumJobRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete quantum job method over gRPC.

        -

        Returns:
            Callable[[~.DeleteQuantumJobRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'delete_quantum_job' not in self._stubs:
            self._stubs['delete_quantum_job'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/DeleteQuantumJob',
                request_serializer=engine.DeleteQuantumJobRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs['delete_quantum_job']

    @property
    def update_quantum_job(self) -> Callable[[engine.UpdateQuantumJobRequest], quantum.QuantumJob]:
        r"""Return a callable for the update quantum job method over gRPC.

        -

        Returns:
            Callable[[~.UpdateQuantumJobRequest],
                    ~.QuantumJob]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'update_quantum_job' not in self._stubs:
            self._stubs['update_quantum_job'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/UpdateQuantumJob',
                request_serializer=engine.UpdateQuantumJobRequest.serialize,
                response_deserializer=quantum.QuantumJob.deserialize,
            )
        return self._stubs['update_quantum_job']

    @property
    def cancel_quantum_job(self) -> Callable[[engine.CancelQuantumJobRequest], empty_pb2.Empty]:
        r"""Return a callable for the cancel quantum job method over gRPC.

        -

        Returns:
            Callable[[~.CancelQuantumJobRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'cancel_quantum_job' not in self._stubs:
            self._stubs['cancel_quantum_job'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/CancelQuantumJob',
                request_serializer=engine.CancelQuantumJobRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs['cancel_quantum_job']

    @property
    def list_quantum_job_events(
        self,
    ) -> Callable[[engine.ListQuantumJobEventsRequest], engine.ListQuantumJobEventsResponse]:
        r"""Return a callable for the list quantum job events method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumJobEventsRequest],
                    ~.ListQuantumJobEventsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_job_events' not in self._stubs:
            self._stubs['list_quantum_job_events'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumJobEvents',
                request_serializer=engine.ListQuantumJobEventsRequest.serialize,
                response_deserializer=engine.ListQuantumJobEventsResponse.deserialize,
            )
        return self._stubs['list_quantum_job_events']

    @property
    def get_quantum_result(
        self,
    ) -> Callable[[engine.GetQuantumResultRequest], quantum.QuantumResult]:
        r"""Return a callable for the get quantum result method over gRPC.

        -

        Returns:
            Callable[[~.GetQuantumResultRequest],
                    ~.QuantumResult]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'get_quantum_result' not in self._stubs:
            self._stubs['get_quantum_result'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/GetQuantumResult',
                request_serializer=engine.GetQuantumResultRequest.serialize,
                response_deserializer=quantum.QuantumResult.deserialize,
            )
        return self._stubs['get_quantum_result']

    @property
    def list_quantum_processors(
        self,
    ) -> Callable[[engine.ListQuantumProcessorsRequest], engine.ListQuantumProcessorsResponse]:
        r"""Return a callable for the list quantum processors method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumProcessorsRequest],
                    ~.ListQuantumProcessorsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_processors' not in self._stubs:
            self._stubs['list_quantum_processors'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumProcessors',
                request_serializer=engine.ListQuantumProcessorsRequest.serialize,
                response_deserializer=engine.ListQuantumProcessorsResponse.deserialize,
            )
        return self._stubs['list_quantum_processors']

    @property
    def get_quantum_processor(
        self,
    ) -> Callable[[engine.GetQuantumProcessorRequest], quantum.QuantumProcessor]:
        r"""Return a callable for the get quantum processor method over gRPC.

        -

        Returns:
            Callable[[~.GetQuantumProcessorRequest],
                    ~.QuantumProcessor]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'get_quantum_processor' not in self._stubs:
            self._stubs['get_quantum_processor'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/GetQuantumProcessor',
                request_serializer=engine.GetQuantumProcessorRequest.serialize,
                response_deserializer=quantum.QuantumProcessor.deserialize,
            )
        return self._stubs['get_quantum_processor']

    @property
    def list_quantum_calibrations(
        self,
    ) -> Callable[[engine.ListQuantumCalibrationsRequest], engine.ListQuantumCalibrationsResponse]:
        r"""Return a callable for the list quantum calibrations method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumCalibrationsRequest],
                    ~.ListQuantumCalibrationsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_calibrations' not in self._stubs:
            self._stubs['list_quantum_calibrations'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumCalibrations',
                request_serializer=engine.ListQuantumCalibrationsRequest.serialize,
                response_deserializer=engine.ListQuantumCalibrationsResponse.deserialize,
            )
        return self._stubs['list_quantum_calibrations']

    @property
    def get_quantum_calibration(
        self,
    ) -> Callable[[engine.GetQuantumCalibrationRequest], quantum.QuantumCalibration]:
        r"""Return a callable for the get quantum calibration method over gRPC.

        -

        Returns:
            Callable[[~.GetQuantumCalibrationRequest],
                    ~.QuantumCalibration]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'get_quantum_calibration' not in self._stubs:
            self._stubs['get_quantum_calibration'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/GetQuantumCalibration',
                request_serializer=engine.GetQuantumCalibrationRequest.serialize,
                response_deserializer=quantum.QuantumCalibration.deserialize,
            )
        return self._stubs['get_quantum_calibration']

    @property
    def create_quantum_reservation(
        self,
    ) -> Callable[[engine.CreateQuantumReservationRequest], quantum.QuantumReservation]:
        r"""Return a callable for the create quantum reservation method over gRPC.

        -

        Returns:
            Callable[[~.CreateQuantumReservationRequest],
                    ~.QuantumReservation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'create_quantum_reservation' not in self._stubs:
            self._stubs['create_quantum_reservation'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/CreateQuantumReservation',
                request_serializer=engine.CreateQuantumReservationRequest.serialize,
                response_deserializer=quantum.QuantumReservation.deserialize,
            )
        return self._stubs['create_quantum_reservation']

    @property
    def cancel_quantum_reservation(
        self,
    ) -> Callable[[engine.CancelQuantumReservationRequest], quantum.QuantumReservation]:
        r"""Return a callable for the cancel quantum reservation method over gRPC.

        -

        Returns:
            Callable[[~.CancelQuantumReservationRequest],
                    ~.QuantumReservation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'cancel_quantum_reservation' not in self._stubs:
            self._stubs['cancel_quantum_reservation'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/CancelQuantumReservation',
                request_serializer=engine.CancelQuantumReservationRequest.serialize,
                response_deserializer=quantum.QuantumReservation.deserialize,
            )
        return self._stubs['cancel_quantum_reservation']

    @property
    def delete_quantum_reservation(
        self,
    ) -> Callable[[engine.DeleteQuantumReservationRequest], empty_pb2.Empty]:
        r"""Return a callable for the delete quantum reservation method over gRPC.

        -

        Returns:
            Callable[[~.DeleteQuantumReservationRequest],
                    ~.Empty]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'delete_quantum_reservation' not in self._stubs:
            self._stubs['delete_quantum_reservation'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/DeleteQuantumReservation',
                request_serializer=engine.DeleteQuantumReservationRequest.serialize,
                response_deserializer=empty_pb2.Empty.FromString,
            )
        return self._stubs['delete_quantum_reservation']

    @property
    def get_quantum_reservation(
        self,
    ) -> Callable[[engine.GetQuantumReservationRequest], quantum.QuantumReservation]:
        r"""Return a callable for the get quantum reservation method over gRPC.

        -

        Returns:
            Callable[[~.GetQuantumReservationRequest],
                    ~.QuantumReservation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'get_quantum_reservation' not in self._stubs:
            self._stubs['get_quantum_reservation'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/GetQuantumReservation',
                request_serializer=engine.GetQuantumReservationRequest.serialize,
                response_deserializer=quantum.QuantumReservation.deserialize,
            )
        return self._stubs['get_quantum_reservation']

    @property
    def list_quantum_reservations(
        self,
    ) -> Callable[[engine.ListQuantumReservationsRequest], engine.ListQuantumReservationsResponse]:
        r"""Return a callable for the list quantum reservations method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumReservationsRequest],
                    ~.ListQuantumReservationsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_reservations' not in self._stubs:
            self._stubs['list_quantum_reservations'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumReservations',
                request_serializer=engine.ListQuantumReservationsRequest.serialize,
                response_deserializer=engine.ListQuantumReservationsResponse.deserialize,
            )
        return self._stubs['list_quantum_reservations']

    @property
    def update_quantum_reservation(
        self,
    ) -> Callable[[engine.UpdateQuantumReservationRequest], quantum.QuantumReservation]:
        r"""Return a callable for the update quantum reservation method over gRPC.

        -

        Returns:
            Callable[[~.UpdateQuantumReservationRequest],
                    ~.QuantumReservation]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'update_quantum_reservation' not in self._stubs:
            self._stubs['update_quantum_reservation'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/UpdateQuantumReservation',
                request_serializer=engine.UpdateQuantumReservationRequest.serialize,
                response_deserializer=quantum.QuantumReservation.deserialize,
            )
        return self._stubs['update_quantum_reservation']

    @property
    def quantum_run_stream(
        self,
    ) -> Callable[[engine.QuantumRunStreamRequest], engine.QuantumRunStreamResponse]:
        r"""Return a callable for the quantum run stream method over gRPC.

        -

        Returns:
            Callable[[~.QuantumRunStreamRequest],
                    ~.QuantumRunStreamResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'quantum_run_stream' not in self._stubs:
            self._stubs['quantum_run_stream'] = self.grpc_channel.stream_stream(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/QuantumRunStream',
                request_serializer=engine.QuantumRunStreamRequest.serialize,
                response_deserializer=engine.QuantumRunStreamResponse.deserialize,
            )
        return self._stubs['quantum_run_stream']

    @property
    def list_quantum_reservation_grants(
        self,
    ) -> Callable[
        [engine.ListQuantumReservationGrantsRequest], engine.ListQuantumReservationGrantsResponse
    ]:
        r"""Return a callable for the list quantum reservation
        grants method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumReservationGrantsRequest],
                    ~.ListQuantumReservationGrantsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_reservation_grants' not in self._stubs:
            self._stubs['list_quantum_reservation_grants'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumReservationGrants',
                request_serializer=engine.ListQuantumReservationGrantsRequest.serialize,
                response_deserializer=engine.ListQuantumReservationGrantsResponse.deserialize,
            )
        return self._stubs['list_quantum_reservation_grants']

    @property
    def reallocate_quantum_reservation_grant(
        self,
    ) -> Callable[
        [engine.ReallocateQuantumReservationGrantRequest], quantum.QuantumReservationGrant
    ]:
        r"""Return a callable for the reallocate quantum reservation
        grant method over gRPC.

        -

        Returns:
            Callable[[~.ReallocateQuantumReservationGrantRequest],
                    ~.QuantumReservationGrant]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'reallocate_quantum_reservation_grant' not in self._stubs:
            self._stubs['reallocate_quantum_reservation_grant'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ReallocateQuantumReservationGrant',
                request_serializer=engine.ReallocateQuantumReservationGrantRequest.serialize,
                response_deserializer=quantum.QuantumReservationGrant.deserialize,
            )
        return self._stubs['reallocate_quantum_reservation_grant']

    @property
    def list_quantum_reservation_budgets(
        self,
    ) -> Callable[
        [engine.ListQuantumReservationBudgetsRequest], engine.ListQuantumReservationBudgetsResponse
    ]:
        r"""Return a callable for the list quantum reservation
        budgets method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumReservationBudgetsRequest],
                    ~.ListQuantumReservationBudgetsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_reservation_budgets' not in self._stubs:
            self._stubs['list_quantum_reservation_budgets'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumReservationBudgets',
                request_serializer=engine.ListQuantumReservationBudgetsRequest.serialize,
                response_deserializer=engine.ListQuantumReservationBudgetsResponse.deserialize,
            )
        return self._stubs['list_quantum_reservation_budgets']

    @property
    def list_quantum_time_slots(
        self,
    ) -> Callable[[engine.ListQuantumTimeSlotsRequest], engine.ListQuantumTimeSlotsResponse]:
        r"""Return a callable for the list quantum time slots method over gRPC.

        -

        Returns:
            Callable[[~.ListQuantumTimeSlotsRequest],
                    ~.ListQuantumTimeSlotsResponse]:
                A function that, when called, will call the underlying RPC
                on the server.
        """
        # Generate a "stub function" on-the-fly which will actually make
        # the request.
        # gRPC handles serialization and deserialization, so we just need
        # to pass in the functions for each.
        if 'list_quantum_time_slots' not in self._stubs:
            self._stubs['list_quantum_time_slots'] = self.grpc_channel.unary_unary(
                '/google.cloud.quantum.v1alpha1.QuantumEngineService/ListQuantumTimeSlots',
                request_serializer=engine.ListQuantumTimeSlotsRequest.serialize,
                response_deserializer=engine.ListQuantumTimeSlotsResponse.deserialize,
            )
        return self._stubs['list_quantum_time_slots']

    def close(self):
        self.grpc_channel.close()


__all__ = ('QuantumEngineServiceGrpcTransport',)
