# -*- coding: utf-8 -*-
#
# Copyright 2019 Google LLC
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
"""Accesses the cirq.google.engine.client.quantum.v1alpha1 QuantumEngineService
API."""

import functools
import warnings

from google.oauth2 import service_account
import google.api_core.client_options
import google.api_core.gapic_v1.client_info
import google.api_core.gapic_v1.config
import google.api_core.gapic_v1.method
import google.api_core.gapic_v1.routing_header
import google.api_core.grpc_helpers
import google.api_core.page_iterator
import google.api_core.path_template
import google.api_core.protobuf_helpers

from cirq.google.engine.client.quantum_v1alpha1.proto import engine_pb2
from cirq.google.engine.client.quantum_v1alpha1.gapic import \
    quantum_engine_service_client_config
from cirq.google.engine.client.quantum_v1alpha1.gapic.transports import \
    quantum_engine_service_grpc_transport

_GAPIC_LIBRARY_VERSION = 0.1


class QuantumEngineServiceClient(object):
    """-"""

    SERVICE_ADDRESS = 'quantum.googleapis.com:443'
    """The default address of the service."""

    # The name of the interface for this client. This is the key used to
    # find the method configuration in the client_config dictionary.
    _INTERFACE_NAME = \
        'cirq.google.engine.client.quantum.v1alpha1.QuantumEngineService'

    @classmethod
    def from_service_account_file(cls, filename, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
        file.

        Args:
            filename (str): The path to the service account private key json
                file.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            QuantumEngineServiceClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_file(
            filename)
        kwargs['credentials'] = credentials
        return cls(*args, **kwargs)

    from_service_account_json = from_service_account_file

    @classmethod
    def calibration_path(cls, project, processor, calibration):
        """DEPRECATED. Return a fully-qualified calibration string."""
        warnings.warn('Resource name helper functions are deprecated.',
                      PendingDeprecationWarning,
                      stacklevel=1)
        return google.api_core.path_template.expand(
            'projects/{project}/processors/{processor}/calibrations/'
            '{calibration}',
            project=project,
            processor=processor,
            calibration=calibration,
        )

    @classmethod
    def job_path(cls, project, program, job):
        """DEPRECATED. Return a fully-qualified job string."""
        warnings.warn('Resource name helper functions are deprecated.',
                      PendingDeprecationWarning,
                      stacklevel=1)
        return google.api_core.path_template.expand(
            'projects/{project}/programs/{program}/jobs/{job}',
            project=project,
            program=program,
            job=job,
        )

    @classmethod
    def processor_path(cls, project, processor):
        """DEPRECATED. Return a fully-qualified processor string."""
        warnings.warn('Resource name helper functions are deprecated.',
                      PendingDeprecationWarning,
                      stacklevel=1)
        return google.api_core.path_template.expand(
            'projects/{project}/processors/{processor}',
            project=project,
            processor=processor,
        )

    @classmethod
    def program_path(cls, project, program):
        """DEPRECATED. Return a fully-qualified program string."""
        warnings.warn('Resource name helper functions are deprecated.',
                      PendingDeprecationWarning,
                      stacklevel=1)
        return google.api_core.path_template.expand(
            'projects/{project}/programs/{program}',
            project=project,
            program=program,
        )

    @classmethod
    def project_path(cls, project):
        """DEPRECATED. Return a fully-qualified project string."""
        warnings.warn('Resource name helper functions are deprecated.',
                      PendingDeprecationWarning,
                      stacklevel=1)
        return google.api_core.path_template.expand(
            'projects/{project}',
            project=project,
        )

    def __init__(self,
                 transport=None,
                 channel=None,
                 credentials=None,
                 client_config=None,
                 client_info=None,
                 client_options=None):
        """Constructor.

        Args:
            transport (Union[~.QuantumEngineServiceGrpcTransport,
                    Callable[[~.Credentials, type],
                    ~.QuantumEngineServiceGrpcTransport]): A transport
                instance, responsible for actually making the API calls.
                The default transport uses the gRPC protocol.
                This argument may also be a callable which returns a
                transport instance. Callables will be sent the credentials
                as the first argument and the default transport class as
                the second argument.
            channel (grpc.Channel): DEPRECATED. A ``Channel`` instance
                through which to make calls. This argument is mutually exclusive
                with ``credentials``; providing both will raise an exception.
            credentials (google.auth.credentials.Credentials): The
                authorization credentials to attach to requests. These
                credentials identify this application to the service. If none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
                This argument is mutually exclusive with providing a
                transport instance to ``transport``; doing so will raise
                an exception.
            client_config (dict): DEPRECATED. A dictionary of call options for
                each method. If not specified, the default configuration is
                used.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            client_options (Union[dict,
             google.api_core.client_options.ClientOptions]):
                Client options used to set user options on the client. API
                Endpoint should be set through client_options.
        """
        # Raise deprecation warnings for things we want to go away.
        if client_config is not None:
            warnings.warn('The `client_config` argument is deprecated.',
                          PendingDeprecationWarning,
                          stacklevel=2)
        else:
            client_config = quantum_engine_service_client_config.config

        if channel:
            warnings.warn(
                'The `channel` argument is deprecated; use '
                '`transport` instead.',
                PendingDeprecationWarning,
                stacklevel=2)

        api_endpoint = self.SERVICE_ADDRESS
        if client_options:
            if type(client_options) == dict:
                client_options = google.api_core.client_options.from_dict(
                    client_options)
            if client_options.api_endpoint:
                api_endpoint = client_options.api_endpoint

        # Instantiate the transport.
        # The transport is responsible for handling serialization and
        # deserialization and actually sending data to the service.
        if transport:
            if callable(transport):
                self.transport = transport(
                    credentials=credentials,
                    default_class=quantum_engine_service_grpc_transport.
                    QuantumEngineServiceGrpcTransport,
                    address=api_endpoint,
                )
            else:
                if credentials:
                    raise ValueError(
                        'Received both a transport instance and '
                        'credentials; these are mutually exclusive.')
                self.transport = transport
        else:
            self.transport = quantum_engine_service_grpc_transport.\
                QuantumEngineServiceGrpcTransport(
                address=api_endpoint,
                channel=channel,
                credentials=credentials,
            )

        if client_info is None:
            client_info = google.api_core.gapic_v1.client_info.ClientInfo(
                gapic_version=_GAPIC_LIBRARY_VERSION,)
        else:
            client_info.gapic_version = _GAPIC_LIBRARY_VERSION
        self._client_info = client_info

        # Parse out the default settings for retry and timeout for each RPC
        # from the client configuration.
        # (Ordinarily, these are the defaults specified in the `*_config.py`
        # file next to this one.)
        self._method_configs = google.api_core.gapic_v1.config.parse_method_configs(
            client_config['interfaces'][self._INTERFACE_NAME],)

        # Save a dictionary of cached API call functions.
        # These are the actual callables which invoke the proper
        # transport methods, wrapped with `wrap_method` to add retry,
        # timeout, and the like.
        self._inner_api_calls = {}

    # Service calls
    def create_quantum_program(self,
                               parent,
                               quantum_program,
                               overwrite_existing_source_code,
                               retry=google.api_core.gapic_v1.method.DEFAULT,
                               timeout=google.api_core.gapic_v1.method.DEFAULT,
                               metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> parent = client.project_path('[PROJECT]')
            >>>
            >>> # TODO: Initialize `quantum_program`:
            >>> quantum_program = {}
            >>>
            >>> # TODO: Initialize `overwrite_existing_source_code`:
            >>> overwrite_existing_source_code = False
            >>>
            >>> response = client.create_quantum_program(parent,
            >>>     quantum_program, overwrite_existing_source_code)

        Args:
            parent (str): -
            quantum_program (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram`
            overwrite_existing_source_code (bool): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'create_quantum_program' not in self._inner_api_calls:
            self._inner_api_calls[
                'create_quantum_program'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.create_quantum_program,
                    default_retry=self._method_configs['CreateQuantumProgram'].
                    retry,
                    default_timeout=self.
                    _method_configs['CreateQuantumProgram'].timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.CreateQuantumProgramRequest(
            parent=parent,
            quantum_program=quantum_program,
            overwrite_existing_source_code=overwrite_existing_source_code,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('parent', parent)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['create_quantum_program'](
            request, retry=retry, timeout=timeout, metadata=metadata)

    def get_quantum_program(self,
                            name,
                            return_code,
                            retry=google.api_core.gapic_v1.method.DEFAULT,
                            timeout=google.api_core.gapic_v1.method.DEFAULT,
                            metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.program_path('[PROJECT]', '[PROGRAM]')
            >>>
            >>> # TODO: Initialize `return_code`:
            >>> return_code = False
            >>>
            >>> response = client.get_quantum_program(name, return_code)

        Args:
            name (str): -
            return_code (bool): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'get_quantum_program' not in self._inner_api_calls:
            self._inner_api_calls[
                'get_quantum_program'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.get_quantum_program,
                    default_retry=self._method_configs['GetQuantumProgram'].
                    retry,
                    default_timeout=self._method_configs['GetQuantumProgram'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.GetQuantumProgramRequest(
            name=name,
            return_code=return_code,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_program'](request,
                                                            retry=retry,
                                                            timeout=timeout,
                                                            metadata=metadata)

    def list_quantum_programs(self,
                              parent,
                              filter_,
                              page_size=None,
                              retry=google.api_core.gapic_v1.method.DEFAULT,
                              timeout=google.api_core.gapic_v1.method.DEFAULT,
                              metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> parent = client.project_path('[PROJECT]')
            >>>
            >>> # TODO: Initialize `filter_`:
            >>> filter_ = ''
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_programs(parent, filter_):
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_programs(parent, filter_).pages:
            ...     for element in page:
            ...         # process element
            ...         pass

        Args:
            parent (str): -
            filter_ (str): -
            page_size (int): The maximum number of resources contained in the
                underlying API response. If page streaming is performed per-
                resource, this parameter does not affect the return value. If page
                streaming is performed per-page, this determines the maximum number
                of resources in a page.
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~google.api_core.page_iterator.PageIterator` instance.
            An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram` instances.
            You can also iterate over the pages of the response
            using its `pages` property.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'list_quantum_programs' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_programs'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.list_quantum_programs,
                    default_retry=self._method_configs['ListQuantumPrograms'].
                    retry,
                    default_timeout=self._method_configs['ListQuantumPrograms'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.ListQuantumProgramsRequest(
            parent=parent,
            filter=filter_,
            page_size=page_size,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('parent', parent)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_programs'],
                retry=retry,
                timeout=timeout,
                metadata=metadata),
            request=request,
            items_field='programs',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def delete_quantum_program(self,
                               name,
                               delete_jobs,
                               retry=google.api_core.gapic_v1.method.DEFAULT,
                               timeout=google.api_core.gapic_v1.method.DEFAULT,
                               metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.program_path('[PROJECT]', '[PROGRAM]')
            >>>
            >>> # TODO: Initialize `delete_jobs`:
            >>> delete_jobs = False
            >>>
            >>> client.delete_quantum_program(name, delete_jobs)

        Args:
            name (str): -
            delete_jobs (bool): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'delete_quantum_program' not in self._inner_api_calls:
            self._inner_api_calls[
                'delete_quantum_program'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.delete_quantum_program,
                    default_retry=self._method_configs['DeleteQuantumProgram'].
                    retry,
                    default_timeout=self.
                    _method_configs['DeleteQuantumProgram'].timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.DeleteQuantumProgramRequest(
            name=name,
            delete_jobs=delete_jobs,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        self._inner_api_calls['delete_quantum_program'](request,
                                                        retry=retry,
                                                        timeout=timeout,
                                                        metadata=metadata)

    def update_quantum_program(self,
                               name,
                               quantum_program,
                               update_mask,
                               retry=google.api_core.gapic_v1.method.DEFAULT,
                               timeout=google.api_core.gapic_v1.method.DEFAULT,
                               metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.program_path('[PROJECT]', '[PROGRAM]')
            >>>
            >>> # TODO: Initialize `quantum_program`:
            >>> quantum_program = {}
            >>>
            >>> # TODO: Initialize `update_mask`:
            >>> update_mask = {}
            >>>
            >>> response = client.update_quantum_program(name, quantum_program, update_mask)

        Args:
            name (str): -
            quantum_program (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram`
            update_mask (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'update_quantum_program' not in self._inner_api_calls:
            self._inner_api_calls[
                'update_quantum_program'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.update_quantum_program,
                    default_retry=self._method_configs['UpdateQuantumProgram'].
                    retry,
                    default_timeout=self.
                    _method_configs['UpdateQuantumProgram'].timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.UpdateQuantumProgramRequest(
            name=name,
            quantum_program=quantum_program,
            update_mask=update_mask,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['update_quantum_program'](
            request, retry=retry, timeout=timeout, metadata=metadata)

    def create_quantum_job(self,
                           parent,
                           quantum_job,
                           overwrite_existing_run_context,
                           retry=google.api_core.gapic_v1.method.DEFAULT,
                           timeout=google.api_core.gapic_v1.method.DEFAULT,
                           metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> parent = client.program_path('[PROJECT]', '[PROGRAM]')
            >>>
            >>> # TODO: Initialize `quantum_job`:
            >>> quantum_job = {}
            >>>
            >>> # TODO: Initialize `overwrite_existing_run_context`:
            >>> overwrite_existing_run_context = False
            >>>
            >>> response = client.create_quantum_job(parent, quantum_job, overwrite_existing_run_context)

        Args:
            parent (str): -
            quantum_job (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob`
            overwrite_existing_run_context (bool): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'create_quantum_job' not in self._inner_api_calls:
            self._inner_api_calls[
                'create_quantum_job'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.create_quantum_job,
                    default_retry=self._method_configs['CreateQuantumJob'].
                    retry,
                    default_timeout=self._method_configs['CreateQuantumJob'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.CreateQuantumJobRequest(
            parent=parent,
            quantum_job=quantum_job,
            overwrite_existing_run_context=overwrite_existing_run_context,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('parent', parent)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['create_quantum_job'](request,
                                                           retry=retry,
                                                           timeout=timeout,
                                                           metadata=metadata)

    def get_quantum_job(self,
                        name,
                        return_run_context,
                        retry=google.api_core.gapic_v1.method.DEFAULT,
                        timeout=google.api_core.gapic_v1.method.DEFAULT,
                        metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.job_path('[PROJECT]', '[PROGRAM]', '[JOB]')
            >>>
            >>> # TODO: Initialize `return_run_context`:
            >>> return_run_context = False
            >>>
            >>> response = client.get_quantum_job(name, return_run_context)

        Args:
            name (str): -
            return_run_context (bool): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'get_quantum_job' not in self._inner_api_calls:
            self._inner_api_calls[
                'get_quantum_job'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.get_quantum_job,
                    default_retry=self._method_configs['GetQuantumJob'].retry,
                    default_timeout=self._method_configs['GetQuantumJob'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.GetQuantumJobRequest(
            name=name,
            return_run_context=return_run_context,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_job'](request,
                                                        retry=retry,
                                                        timeout=timeout,
                                                        metadata=metadata)

    def list_quantum_jobs(self,
                          parent,
                          filter_,
                          page_size=None,
                          retry=google.api_core.gapic_v1.method.DEFAULT,
                          timeout=google.api_core.gapic_v1.method.DEFAULT,
                          metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> parent = client.program_path('[PROJECT]', '[PROGRAM]')
            >>>
            >>> # TODO: Initialize `filter_`:
            >>> filter_ = ''
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_jobs(parent, filter_):
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_jobs(parent, filter_).pages:
            ...     for element in page:
            ...         # process element
            ...         pass

        Args:
            parent (str): -
            filter_ (str): -
            page_size (int): The maximum number of resources contained in the
                underlying API response. If page streaming is performed per-
                resource, this parameter does not affect the return value. If page
                streaming is performed per-page, this determines the maximum number
                of resources in a page.
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~google.api_core.page_iterator.PageIterator` instance.
            An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob` instances.
            You can also iterate over the pages of the response
            using its `pages` property.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'list_quantum_jobs' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_jobs'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.list_quantum_jobs,
                    default_retry=self._method_configs['ListQuantumJobs'].retry,
                    default_timeout=self._method_configs['ListQuantumJobs'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.ListQuantumJobsRequest(
            parent=parent,
            filter=filter_,
            page_size=page_size,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('parent', parent)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(self._inner_api_calls['list_quantum_jobs'],
                                     retry=retry,
                                     timeout=timeout,
                                     metadata=metadata),
            request=request,
            items_field='jobs',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def delete_quantum_job(self,
                           name,
                           retry=google.api_core.gapic_v1.method.DEFAULT,
                           timeout=google.api_core.gapic_v1.method.DEFAULT,
                           metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.job_path('[PROJECT]', '[PROGRAM]', '[JOB]')
            >>>
            >>> client.delete_quantum_job(name)

        Args:
            name (str): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'delete_quantum_job' not in self._inner_api_calls:
            self._inner_api_calls[
                'delete_quantum_job'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.delete_quantum_job,
                    default_retry=self._method_configs['DeleteQuantumJob'].
                    retry,
                    default_timeout=self._method_configs['DeleteQuantumJob'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.DeleteQuantumJobRequest(name=name,)
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        self._inner_api_calls['delete_quantum_job'](request,
                                                    retry=retry,
                                                    timeout=timeout,
                                                    metadata=metadata)

    def update_quantum_job(self,
                           name,
                           quantum_job,
                           update_mask,
                           retry=google.api_core.gapic_v1.method.DEFAULT,
                           timeout=google.api_core.gapic_v1.method.DEFAULT,
                           metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.job_path('[PROJECT]', '[PROGRAM]', '[JOB]')
            >>>
            >>> # TODO: Initialize `quantum_job`:
            >>> quantum_job = {}
            >>>
            >>> # TODO: Initialize `update_mask`:
            >>> update_mask = {}
            >>>
            >>> response = client.update_quantum_job(name, quantum_job, update_mask)

        Args:
            name (str): -
            quantum_job (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob`
            update_mask (Union[dict, ~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.FieldMask`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'update_quantum_job' not in self._inner_api_calls:
            self._inner_api_calls[
                'update_quantum_job'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.update_quantum_job,
                    default_retry=self._method_configs['UpdateQuantumJob'].
                    retry,
                    default_timeout=self._method_configs['UpdateQuantumJob'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.UpdateQuantumJobRequest(
            name=name,
            quantum_job=quantum_job,
            update_mask=update_mask,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['update_quantum_job'](request,
                                                           retry=retry,
                                                           timeout=timeout,
                                                           metadata=metadata)

    def cancel_quantum_job(self,
                           name,
                           retry=google.api_core.gapic_v1.method.DEFAULT,
                           timeout=google.api_core.gapic_v1.method.DEFAULT,
                           metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.job_path('[PROJECT]', '[PROGRAM]', '[JOB]')
            >>>
            >>> client.cancel_quantum_job(name)

        Args:
            name (str): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'cancel_quantum_job' not in self._inner_api_calls:
            self._inner_api_calls[
                'cancel_quantum_job'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.cancel_quantum_job,
                    default_retry=self._method_configs['CancelQuantumJob'].
                    retry,
                    default_timeout=self._method_configs['CancelQuantumJob'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.CancelQuantumJobRequest(name=name,)
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        self._inner_api_calls['cancel_quantum_job'](request,
                                                    retry=retry,
                                                    timeout=timeout,
                                                    metadata=metadata)

    def list_quantum_job_events(self,
                                parent,
                                page_size=None,
                                retry=google.api_core.gapic_v1.method.DEFAULT,
                                timeout=google.api_core.gapic_v1.method.DEFAULT,
                                metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> parent = client.job_path('[PROJECT]', '[PROGRAM]', '[JOB]')
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_job_events(parent):
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_job_events(parent).pages:
            ...     for element in page:
            ...         # process element
            ...         pass

        Args:
            parent (str): -
            page_size (int): The maximum number of resources contained in the
                underlying API response. If page streaming is performed per-
                resource, this parameter does not affect the return value. If page
                streaming is performed per-page, this determines the maximum number
                of resources in a page.
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~google.api_core.page_iterator.PageIterator` instance.
            An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumJobEvent` instances.
            You can also iterate over the pages of the response
            using its `pages` property.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'list_quantum_job_events' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_job_events'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.list_quantum_job_events,
                    default_retry=self._method_configs['ListQuantumJobEvents'].
                    retry,
                    default_timeout=self.
                    _method_configs['ListQuantumJobEvents'].timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.ListQuantumJobEventsRequest(
            parent=parent,
            page_size=page_size,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('parent', parent)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_job_events'],
                retry=retry,
                timeout=timeout,
                metadata=metadata),
            request=request,
            items_field='events',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def get_quantum_result(self,
                           parent,
                           retry=google.api_core.gapic_v1.method.DEFAULT,
                           timeout=google.api_core.gapic_v1.method.DEFAULT,
                           metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> parent = client.job_path('[PROJECT]', '[PROGRAM]', '[JOB]')
            >>>
            >>> response = client.get_quantum_result(parent)

        Args:
            parent (str): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumResult` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'get_quantum_result' not in self._inner_api_calls:
            self._inner_api_calls[
                'get_quantum_result'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.get_quantum_result,
                    default_retry=self._method_configs['GetQuantumResult'].
                    retry,
                    default_timeout=self._method_configs['GetQuantumResult'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.GetQuantumResultRequest(parent=parent,)
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('parent', parent)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_result'](request,
                                                           retry=retry,
                                                           timeout=timeout,
                                                           metadata=metadata)

    def list_quantum_processors(self,
                                parent,
                                filter_,
                                page_size=None,
                                retry=google.api_core.gapic_v1.method.DEFAULT,
                                timeout=google.api_core.gapic_v1.method.DEFAULT,
                                metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> parent = client.project_path('[PROJECT]')
            >>>
            >>> # TODO: Initialize `filter_`:
            >>> filter_ = ''
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_processors(parent, filter_):
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_processors(parent, filter_).pages:
            ...     for element in page:
            ...         # process element
            ...         pass

        Args:
            parent (str): -
            filter_ (str): -
            page_size (int): The maximum number of resources contained in the
                underlying API response. If page streaming is performed per-
                resource, this parameter does not affect the return value. If page
                streaming is performed per-page, this determines the maximum number
                of resources in a page.
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~google.api_core.page_iterator.PageIterator` instance.
            An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProcessor` instances.
            You can also iterate over the pages of the response
            using its `pages` property.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'list_quantum_processors' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_processors'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.list_quantum_processors,
                    default_retry=self._method_configs['ListQuantumProcessors'].
                    retry,
                    default_timeout=self.
                    _method_configs['ListQuantumProcessors'].timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.ListQuantumProcessorsRequest(
            parent=parent,
            filter=filter_,
            page_size=page_size,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('parent', parent)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_processors'],
                retry=retry,
                timeout=timeout,
                metadata=metadata),
            request=request,
            items_field='processors',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def get_quantum_processor(self,
                              name,
                              retry=google.api_core.gapic_v1.method.DEFAULT,
                              timeout=google.api_core.gapic_v1.method.DEFAULT,
                              metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.processor_path('[PROJECT]', '[PROCESSOR]')
            >>>
            >>> response = client.get_quantum_processor(name)

        Args:
            name (str): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumProcessor` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'get_quantum_processor' not in self._inner_api_calls:
            self._inner_api_calls[
                'get_quantum_processor'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.get_quantum_processor,
                    default_retry=self._method_configs['GetQuantumProcessor'].
                    retry,
                    default_timeout=self._method_configs['GetQuantumProcessor'].
                    timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.GetQuantumProcessorRequest(name=name,)
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_processor'](request,
                                                              retry=retry,
                                                              timeout=timeout,
                                                              metadata=metadata)

    def list_quantum_calibrations(
            self,
            parent,
            filter_,
            page_size=None,
            retry=google.api_core.gapic_v1.method.DEFAULT,
            timeout=google.api_core.gapic_v1.method.DEFAULT,
            metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> parent = client.processor_path('[PROJECT]', '[PROCESSOR]')
            >>>
            >>> # TODO: Initialize `filter_`:
            >>> filter_ = ''
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_calibrations(parent, filter_):
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_calibrations(parent, filter_).pages:
            ...     for element in page:
            ...         # process element
            ...         pass

        Args:
            parent (str): -
            filter_ (str): -
            page_size (int): The maximum number of resources contained in the
                underlying API response. If page streaming is performed per-
                resource, this parameter does not affect the return value. If page
                streaming is performed per-page, this determines the maximum number
                of resources in a page.
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~google.api_core.page_iterator.PageIterator` instance.
            An iterable of :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumCalibration` instances.
            You can also iterate over the pages of the response
            using its `pages` property.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'list_quantum_calibrations' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_calibrations'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.list_quantum_calibrations,
                    default_retry=self.
                    _method_configs['ListQuantumCalibrations'].retry,
                    default_timeout=self.
                    _method_configs['ListQuantumCalibrations'].timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.ListQuantumCalibrationsRequest(
            parent=parent,
            filter=filter_,
            page_size=page_size,
        )
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('parent', parent)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_calibrations'],
                retry=retry,
                timeout=timeout,
                metadata=metadata),
            request=request,
            items_field='calibrations',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def get_quantum_calibration(self,
                                name,
                                retry=google.api_core.gapic_v1.method.DEFAULT,
                                timeout=google.api_core.gapic_v1.method.DEFAULT,
                                metadata=None):
        """
        -

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> name = client.calibration_path('[PROJECT]', '[PROCESSOR]', '[CALIBRATION]')
            >>>
            >>> response = client.get_quantum_calibration(name)

        Args:
            name (str): -
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumCalibration` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'get_quantum_calibration' not in self._inner_api_calls:
            self._inner_api_calls[
                'get_quantum_calibration'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.get_quantum_calibration,
                    default_retry=self._method_configs['GetQuantumCalibration'].
                    retry,
                    default_timeout=self.
                    _method_configs['GetQuantumCalibration'].timeout,
                    client_info=self._client_info,
                )

        request = engine_pb2.GetQuantumCalibrationRequest(name=name,)
        if metadata is None:
            metadata = []
        metadata = list(metadata)
        try:
            routing_header = [('name', name)]
        except AttributeError:
            pass
        else:
            routing_metadata = google.api_core.gapic_v1.routing_header.to_grpc_metadata(
                routing_header)
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_calibration'](
            request, retry=retry, timeout=timeout, metadata=metadata)

    def quantum_run_stream(self,
                           requests,
                           retry=google.api_core.gapic_v1.method.DEFAULT,
                           timeout=google.api_core.gapic_v1.method.DEFAULT,
                           metadata=None):
        """
        -

        EXPERIMENTAL: This method interface might change in the future.

        Example:
            >>> from cirq.google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # TODO: Initialize `message_id`:
            >>> message_id = ''
            >>>
            >>> # TODO: Initialize `parent`:
            >>> parent = ''
            >>> request = {'message_id': message_id, 'parent': parent}
            >>>
            >>> requests = [request]
            >>> for element in client.quantum_run_stream(requests):
            ...     # process element
            ...     pass

        Args:
            requests (iterator[dict|cirq.google.engine.client.quantum.v1alpha1.engine_pb2.QuantumRunStreamRequest]): The input objects. If a dict is provided, it must be of the
                same form as the protobuf message :class:`~cirq.google.engine.client.quantum_v1alpha1.types.QuantumRunStreamRequest`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            Iterable[~cirq.google.engine.client.quantum_v1alpha1.types.QuantumRunStreamResponse].

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'quantum_run_stream' not in self._inner_api_calls:
            self._inner_api_calls[
                'quantum_run_stream'] = google.api_core.gapic_v1.method.wrap_method(
                    self.transport.quantum_run_stream,
                    default_retry=self._method_configs['QuantumRunStream'].
                    retry,
                    default_timeout=self._method_configs['QuantumRunStream'].
                    timeout,
                    client_info=self._client_info,
                )

        return self._inner_api_calls['quantum_run_stream'](requests,
                                                           retry=retry,
                                                           timeout=timeout,
                                                           metadata=metadata)
