# -*- coding: utf-8 -*-
#
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


import google.api_core.grpc_helpers

from cirq.google.engine.client.quantum_v1alpha1.proto import engine_pb2_grpc


class QuantumEngineServiceGrpcTransport(object):
    """gRPC transport class providing stubs for
    google.cloud.quantum.v1alpha1 QuantumEngineService API.

    The transport provides access to the raw gRPC stubs,
    which can be used to take advantage of advanced
    features of gRPC.
    """

    # The scopes needed to make gRPC calls to all of the methods defined
    # in this service.
    _OAUTH_SCOPES = ('https://www.googleapis.com/auth/cloud-platform',)

    def __init__(self, channel=None, credentials=None, address='quantum.googleapis.com:443'):
        """Instantiate the transport class.

        Args:
            channel (grpc.Channel): A ``Channel`` instance through
                which to make calls. This argument is mutually exclusive
                with ``credentials``; providing both will raise an exception.
            credentials (google.auth.credentials.Credentials): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            address (str): The address where the service is hosted.
        """
        # If both `channel` and `credentials` are specified, raise an
        # exception (channels come with credentials baked in already).
        if channel is not None and credentials is not None:
            raise ValueError(
                'The `channel` and `credentials` arguments are mutually exclusive.',
            )

        # Create the channel.
        if channel is None:
            channel = self.create_channel(
                address=address,
                credentials=credentials,
                options={
                    'grpc.max_send_message_length': -1,
                    'grpc.max_receive_message_length': -1,
                }.items(),
            )

        self._channel = channel

        # gRPC uses objects called "stubs" that are bound to the
        # channel and provide a basic method for each RPC.
        self._stubs = {
            'quantum_engine_service_stub': engine_pb2_grpc.QuantumEngineServiceStub(channel),
        }

    @classmethod
    def create_channel(cls, address='quantum.googleapis.com:443', credentials=None, **kwargs):
        """Create and return a gRPC channel object.

        Args:
            address (str): The host for the channel to use.
            credentials (~.Credentials): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If
                none are specified, the client will attempt to ascertain
                the credentials from the environment.
            kwargs (dict): Keyword arguments, which are passed to the
                channel creation.

        Returns:
            grpc.Channel: A gRPC channel object.
        """
        return google.api_core.grpc_helpers.create_channel(
            address, credentials=credentials, scopes=cls._OAUTH_SCOPES, **kwargs
        )

    @property
    def channel(self):
        """The gRPC channel used by the transport.

        Returns:
            grpc.Channel: A gRPC channel object.
        """
        return self._channel

    @property
    def create_quantum_program(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.create_quantum_program`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].CreateQuantumProgram

    @property
    def get_quantum_program(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.get_quantum_program`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].GetQuantumProgram

    @property
    def list_quantum_programs(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_programs`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumPrograms

    @property
    def delete_quantum_program(self):
        """Return the gRPC stub
        for :meth:`QuantumEngineServiceClient.delete_quantum_program`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].DeleteQuantumProgram

    @property
    def update_quantum_program(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.update_quantum_program`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].UpdateQuantumProgram

    @property
    def create_quantum_job(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.create_quantum_job`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].CreateQuantumJob

    @property
    def get_quantum_job(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.get_quantum_job`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].GetQuantumJob

    @property
    def list_quantum_jobs(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_jobs`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumJobs

    @property
    def delete_quantum_job(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.delete_quantum_job`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].DeleteQuantumJob

    @property
    def update_quantum_job(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.update_quantum_job`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].UpdateQuantumJob

    @property
    def cancel_quantum_job(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.cancel_quantum_job`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].CancelQuantumJob

    @property
    def list_quantum_job_events(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_job_events`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumJobEvents

    @property
    def get_quantum_result(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.get_quantum_result`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].GetQuantumResult

    @property
    def list_quantum_processors(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_processors`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumProcessors

    @property
    def get_quantum_processor(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.get_quantum_processor`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].GetQuantumProcessor

    @property
    def list_quantum_calibrations(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_calibrations`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumCalibrations

    @property
    def get_quantum_calibration(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.get_quantum_calibration`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].GetQuantumCalibration

    @property
    def create_quantum_reservation(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.create_quantum_reservation`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].CreateQuantumReservation

    @property
    def cancel_quantum_reservation(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.cancel_quantum_reservation`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].CancelQuantumReservation

    @property
    def delete_quantum_reservation(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.delete_quantum_reservation`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].DeleteQuantumReservation

    @property
    def get_quantum_reservation(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.get_quantum_reservation`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].GetQuantumReservation

    @property
    def list_quantum_reservations(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_reservations`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumReservations

    @property
    def update_quantum_reservation(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.update_quantum_reservation`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].UpdateQuantumReservation

    @property
    def quantum_run_stream(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.quantum_run_stream`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].QuantumRunStream

    @property
    def list_quantum_reservation_grants(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_reservation_grants`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumReservationGrants

    @property
    def reallocate_quantum_reservation_grant(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.reallocate_quantum_reservation_grant`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ReallocateQuantumReservationGrant

    @property
    def list_quantum_reservation_budgets(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_reservation_budgets`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumReservationBudgets

    @property
    def list_quantum_time_slots(self):
        """Return the gRPC stub for
        :meth:`QuantumEngineServiceClient.list_quantum_time_slots`.

        -

        Returns:
            Callable: A callable which accepts the appropriate
                deserialized request object and returns a
                deserialized response object.
        """
        return self._stubs['quantum_engine_service_stub'].ListQuantumTimeSlots
