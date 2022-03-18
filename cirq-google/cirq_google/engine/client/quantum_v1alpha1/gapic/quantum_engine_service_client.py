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
"""Accesses the google.cloud.quantum.v1alpha1 QuantumEngineService API."""

import functools
import pkg_resources
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import warnings

from google.oauth2 import service_account
import google.api_core.client_options
import google.api_core.gapic_v1.client_info
import google.api_core.gapic_v1.config
import google.api_core.gapic_v1.method
import google.api_core.gapic_v1.routing_header
import google.api_core.grpc_helpers
import google.api_core.page_iterator
import google.api_core.protobuf_helpers
import grpc

from cirq_google.engine.client.quantum_v1alpha1 import types as pb_types
from cirq_google.engine.client.quantum_v1alpha1.gapic import enums
from cirq_google.engine.client.quantum_v1alpha1.gapic import quantum_engine_service_client_config
from cirq_google.engine.client.quantum_v1alpha1.gapic.transports import (
    quantum_engine_service_grpc_transport,
)
from cirq_google.engine.client.quantum_v1alpha1.proto import engine_pb2
from cirq_google.engine.client.quantum_v1alpha1.proto import engine_pb2_grpc
from cirq_google.engine.client.quantum_v1alpha1.proto import quantum_pb2
from google.protobuf import duration_pb2
from google.protobuf import empty_pb2
from google.protobuf import field_mask_pb2

_GAPIC_LIBRARY_VERSION = 0.1

QUANTUM_ENGINE_SERVICE_GRPC_TRANSPORT_LIKE = Union[
    quantum_engine_service_grpc_transport.QuantumEngineServiceGrpcTransport,
    Callable[..., quantum_engine_service_grpc_transport.QuantumEngineServiceGrpcTransport],
]


class QuantumEngineServiceClient(object):
    """-"""

    SERVICE_ADDRESS = 'quantum.googleapis.com:443'
    """The default address of the service."""

    # The name of the interface for this client. This is the key used to
    # find the method configuration in the client_config dictionary.
    _INTERFACE_NAME = 'google.cloud.quantum.v1alpha1.QuantumEngineService'

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
        credentials = service_account.Credentials.from_service_account_file(filename)
        kwargs['credentials'] = credentials
        return cls(*args, **kwargs)

    from_service_account_json = from_service_account_file

    def __init__(
        self,
        transport: QUANTUM_ENGINE_SERVICE_GRPC_TRANSPORT_LIKE = None,
        channel: Optional[grpc.Channel] = None,
        credentials: Optional[service_account.Credentials] = None,
        client_config: Optional[Dict[str, Any]] = None,
        client_info: Optional[google.api_core.gapic_v1.client_info.ClientInfo] = None,
        client_options: Union[Dict[str, Any], google.api_core.client_options.ClientOptions] = None,
    ):
        """Constructor.

        Args:
            transport (Union[~.QuantumEngineServiceGrpcTransport,
                    Callable[[~.Credentials, type], ~.QuantumEngineServiceGrpcTransport]): A transport
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
                each method. If not specified, the default configuration is used.
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.
            client_options (Union[dict, google.api_core.client_options.ClientOptions]):
                Client options used to set user options on the client. API Endpoint
                should be set through client_options.
        """
        # Raise deprecation warnings for things we want to go away.
        if client_config is not None:
            warnings.warn(
                'The `client_config` argument is deprecated.',
                PendingDeprecationWarning,
                stacklevel=2,
            )
        else:
            client_config = quantum_engine_service_client_config.config

        if channel:
            warnings.warn(
                'The `channel` argument is deprecated; use `transport` instead.',
                PendingDeprecationWarning,
                stacklevel=2,
            )

        api_endpoint = self.SERVICE_ADDRESS
        if client_options:
            if type(client_options) == dict:
                client_options = google.api_core.client_options.from_dict(client_options)
            if client_options.api_endpoint:
                api_endpoint = client_options.api_endpoint

        # Instantiate the transport.
        # The transport is responsible for handling serialization and
        # deserialization and actually sending data to the service.
        if transport:
            if callable(transport):
                self.transport = transport(
                    credentials=credentials,
                    default_class=quantum_engine_service_grpc_transport.QuantumEngineServiceGrpcTransport,
                    address=api_endpoint,
                )
            else:
                if credentials:
                    raise ValueError(
                        'Received both a transport instance and '
                        'credentials; these are mutually exclusive.'
                    )
                self.transport = transport
        else:
            self.transport = (
                quantum_engine_service_grpc_transport.QuantumEngineServiceGrpcTransport(
                    address=api_endpoint,
                    channel=channel,
                    credentials=credentials,
                )
            )

        if client_info is None:
            client_info = google.api_core.gapic_v1.client_info.ClientInfo(
                gapic_version=_GAPIC_LIBRARY_VERSION,
            )
        else:
            client_info.gapic_version = _GAPIC_LIBRARY_VERSION
        self._client_info = client_info

        # Parse out the default settings for retry and timeout for each RPC
        # from the client configuration.
        # (Ordinarily, these are the defaults specified in the `*_config.py`
        # file next to this one.)
        self._method_configs = google.api_core.gapic_v1.config.parse_method_configs(
            client_config['interfaces'][self._INTERFACE_NAME],
        )

        # Save a dictionary of cached API call functions.
        # These are the actual callables which invoke the proper
        # transport methods, wrapped with `wrap_method` to add retry,
        # timeout, and the like.
        self._inner_api_calls: Dict = {}

    # Service calls
    def create_quantum_program(
        self,
        parent: Optional[str] = None,
        quantum_program: Union[Dict[str, Any], pb_types.QuantumProgram] = None,
        overwrite_existing_source_code: Optional[bool] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.create_quantum_program()

        Args:
            parent (str): -
            quantum_program (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProgram]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProgram`
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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.

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
                'create_quantum_program'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.create_quantum_program,
                default_retry=self._method_configs['CreateQuantumProgram'].retry,
                default_timeout=self._method_configs['CreateQuantumProgram'].timeout,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['create_quantum_program'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def get_quantum_program(
        self,
        name: Optional[str] = None,
        return_code: Optional[bool] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.get_quantum_program()

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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.

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
                'get_quantum_program'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.get_quantum_program,
                default_retry=self._method_configs['GetQuantumProgram'].retry,
                default_timeout=self._method_configs['GetQuantumProgram'].timeout,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_program'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def list_quantum_programs(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        filter_: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_programs():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_programs().pages:
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
            filter_ (str): -
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProgram` instances.
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
                'list_quantum_programs'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_programs,
                default_retry=self._method_configs['ListQuantumPrograms'].retry,
                default_timeout=self._method_configs['ListQuantumPrograms'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ListQuantumProgramsRequest(
            parent=parent,
            page_size=page_size,
            filter=filter_,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_programs'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='programs',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def delete_quantum_program(
        self,
        name: Optional[str] = None,
        delete_jobs: Optional[bool] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> client.delete_quantum_program()

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
                'delete_quantum_program'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.delete_quantum_program,
                default_retry=self._method_configs['DeleteQuantumProgram'].retry,
                default_timeout=self._method_configs['DeleteQuantumProgram'].timeout,
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
                routing_header
            )
            metadata.append(routing_metadata)

        self._inner_api_calls['delete_quantum_program'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def update_quantum_program(
        self,
        name: Optional[str] = None,
        quantum_program: Union[Dict[str, Any], pb_types.QuantumProgram] = None,
        update_mask: Union[Dict[str, Any], field_mask_pb2.FieldMask] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.update_quantum_program()

        Args:
            name (str): -
            quantum_program (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProgram]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProgram`
            update_mask (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.FieldMask]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.FieldMask`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProgram` instance.

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
                'update_quantum_program'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.update_quantum_program,
                default_retry=self._method_configs['UpdateQuantumProgram'].retry,
                default_timeout=self._method_configs['UpdateQuantumProgram'].timeout,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['update_quantum_program'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def create_quantum_job(
        self,
        parent: Optional[str] = None,
        quantum_job: Union[Dict[str, Any], pb_types.QuantumJob] = None,
        overwrite_existing_run_context: Optional[bool] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.create_quantum_job()

        Args:
            parent (str): -
            quantum_job (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJob]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJob`
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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.

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
                'create_quantum_job'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.create_quantum_job,
                default_retry=self._method_configs['CreateQuantumJob'].retry,
                default_timeout=self._method_configs['CreateQuantumJob'].timeout,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['create_quantum_job'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def get_quantum_job(
        self,
        name: Optional[str] = None,
        return_run_context: Optional[bool] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.get_quantum_job()

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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'get_quantum_job' not in self._inner_api_calls:
            self._inner_api_calls['get_quantum_job'] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.get_quantum_job,
                default_retry=self._method_configs['GetQuantumJob'].retry,
                default_timeout=self._method_configs['GetQuantumJob'].timeout,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_job'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def list_quantum_jobs(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        filter_: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_jobs():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_jobs().pages:
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
            filter_ (str): -
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJob` instances.
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
                'list_quantum_jobs'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_jobs,
                default_retry=self._method_configs['ListQuantumJobs'].retry,
                default_timeout=self._method_configs['ListQuantumJobs'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ListQuantumJobsRequest(
            parent=parent,
            page_size=page_size,
            filter=filter_,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_jobs'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='jobs',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def delete_quantum_job(
        self,
        name: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> client.delete_quantum_job()

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
                'delete_quantum_job'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.delete_quantum_job,
                default_retry=self._method_configs['DeleteQuantumJob'].retry,
                default_timeout=self._method_configs['DeleteQuantumJob'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.DeleteQuantumJobRequest(
            name=name,
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
                routing_header
            )
            metadata.append(routing_metadata)

        self._inner_api_calls['delete_quantum_job'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def update_quantum_job(
        self,
        name: Optional[str] = None,
        quantum_job: Union[Dict[str, Any], pb_types.QuantumJob] = None,
        update_mask: Union[Dict[str, Any], field_mask_pb2.FieldMask] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.update_quantum_job()

        Args:
            name (str): -
            quantum_job (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJob]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJob`
            update_mask (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.FieldMask]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.FieldMask`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJob` instance.

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
                'update_quantum_job'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.update_quantum_job,
                default_retry=self._method_configs['UpdateQuantumJob'].retry,
                default_timeout=self._method_configs['UpdateQuantumJob'].timeout,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['update_quantum_job'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def cancel_quantum_job(
        self,
        name: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> client.cancel_quantum_job()

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
                'cancel_quantum_job'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.cancel_quantum_job,
                default_retry=self._method_configs['CancelQuantumJob'].retry,
                default_timeout=self._method_configs['CancelQuantumJob'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.CancelQuantumJobRequest(
            name=name,
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
                routing_header
            )
            metadata.append(routing_metadata)

        self._inner_api_calls['cancel_quantum_job'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def list_quantum_job_events(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_job_events():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_job_events().pages:
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumJobEvent` instances.
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
                'list_quantum_job_events'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_job_events,
                default_retry=self._method_configs['ListQuantumJobEvents'].retry,
                default_timeout=self._method_configs['ListQuantumJobEvents'].timeout,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_job_events'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='events',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def get_quantum_result(
        self,
        parent: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.get_quantum_result()

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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumResult` instance.

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
                'get_quantum_result'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.get_quantum_result,
                default_retry=self._method_configs['GetQuantumResult'].retry,
                default_timeout=self._method_configs['GetQuantumResult'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.GetQuantumResultRequest(
            parent=parent,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_result'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def list_quantum_processors(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        filter_: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_processors():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_processors().pages:
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
            filter_ (str): -
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProcessor` instances.
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
                'list_quantum_processors'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_processors,
                default_retry=self._method_configs['ListQuantumProcessors'].retry,
                default_timeout=self._method_configs['ListQuantumProcessors'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ListQuantumProcessorsRequest(
            parent=parent,
            page_size=page_size,
            filter=filter_,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_processors'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='processors',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def get_quantum_processor(
        self,
        name: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.get_quantum_processor()

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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumProcessor` instance.

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
                'get_quantum_processor'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.get_quantum_processor,
                default_retry=self._method_configs['GetQuantumProcessor'].retry,
                default_timeout=self._method_configs['GetQuantumProcessor'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.GetQuantumProcessorRequest(
            name=name,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_processor'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def list_quantum_calibrations(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        filter_: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_calibrations():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_calibrations().pages:
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
            filter_ (str): -
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumCalibration` instances.
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
                'list_quantum_calibrations'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_calibrations,
                default_retry=self._method_configs['ListQuantumCalibrations'].retry,
                default_timeout=self._method_configs['ListQuantumCalibrations'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ListQuantumCalibrationsRequest(
            parent=parent,
            page_size=page_size,
            filter=filter_,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_calibrations'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='calibrations',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def get_quantum_calibration(
        self,
        name: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.get_quantum_calibration()

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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumCalibration` instance.

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
                'get_quantum_calibration'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.get_quantum_calibration,
                default_retry=self._method_configs['GetQuantumCalibration'].retry,
                default_timeout=self._method_configs['GetQuantumCalibration'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.GetQuantumCalibrationRequest(
            name=name,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_calibration'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def create_quantum_reservation(
        self,
        parent: Optional[str] = None,
        quantum_reservation: Union[Dict[str, Any], pb_types.QuantumReservation] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.create_quantum_reservation()

        Args:
            parent (str): -
            quantum_reservation (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'create_quantum_reservation' not in self._inner_api_calls:
            self._inner_api_calls[
                'create_quantum_reservation'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.create_quantum_reservation,
                default_retry=self._method_configs['CreateQuantumReservation'].retry,
                default_timeout=self._method_configs['CreateQuantumReservation'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.CreateQuantumReservationRequest(
            parent=parent,
            quantum_reservation=quantum_reservation,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['create_quantum_reservation'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def cancel_quantum_reservation(
        self,
        name: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.cancel_quantum_reservation()

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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'cancel_quantum_reservation' not in self._inner_api_calls:
            self._inner_api_calls[
                'cancel_quantum_reservation'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.cancel_quantum_reservation,
                default_retry=self._method_configs['CancelQuantumReservation'].retry,
                default_timeout=self._method_configs['CancelQuantumReservation'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.CancelQuantumReservationRequest(
            name=name,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['cancel_quantum_reservation'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def delete_quantum_reservation(
        self,
        name: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> client.delete_quantum_reservation()

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
        if 'delete_quantum_reservation' not in self._inner_api_calls:
            self._inner_api_calls[
                'delete_quantum_reservation'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.delete_quantum_reservation,
                default_retry=self._method_configs['DeleteQuantumReservation'].retry,
                default_timeout=self._method_configs['DeleteQuantumReservation'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.DeleteQuantumReservationRequest(
            name=name,
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
                routing_header
            )
            metadata.append(routing_metadata)

        self._inner_api_calls['delete_quantum_reservation'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def get_quantum_reservation(
        self,
        name: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.get_quantum_reservation()

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
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'get_quantum_reservation' not in self._inner_api_calls:
            self._inner_api_calls[
                'get_quantum_reservation'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.get_quantum_reservation,
                default_retry=self._method_configs['GetQuantumReservation'].retry,
                default_timeout=self._method_configs['GetQuantumReservation'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.GetQuantumReservationRequest(
            name=name,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['get_quantum_reservation'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def list_quantum_reservations(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        filter_: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_reservations():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_reservations().pages:
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
            filter_ (str): -
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation` instances.
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
        if 'list_quantum_reservations' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_reservations'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_reservations,
                default_retry=self._method_configs['ListQuantumReservations'].retry,
                default_timeout=self._method_configs['ListQuantumReservations'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ListQuantumReservationsRequest(
            parent=parent,
            page_size=page_size,
            filter=filter_,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_reservations'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='reservations',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def update_quantum_reservation(
        self,
        name: Optional[str] = None,
        quantum_reservation: Union[Dict[str, Any], pb_types.QuantumReservation] = None,
        update_mask: Union[Dict[str, Any], field_mask_pb2.FieldMask] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.update_quantum_reservation()

        Args:
            name (str): -
            quantum_reservation (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation`
            update_mask (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.FieldMask]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.FieldMask`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservation` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'update_quantum_reservation' not in self._inner_api_calls:
            self._inner_api_calls[
                'update_quantum_reservation'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.update_quantum_reservation,
                default_retry=self._method_configs['UpdateQuantumReservation'].retry,
                default_timeout=self._method_configs['UpdateQuantumReservation'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.UpdateQuantumReservationRequest(
            name=name,
            quantum_reservation=quantum_reservation,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['update_quantum_reservation'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def quantum_run_stream(
        self,
        requests,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> request = {}
            >>>
            >>> requests = [request]
            >>> for element in client.quantum_run_stream(requests):
            ...     # process element
            ...     pass

        Args:
            requests (iterator[dict|cirq_google.engine.client.quantum_v1alpha1.proto.engine_pb2.QuantumRunStreamRequest]): The input objects. If a dict is provided, it must be of the
                same form as the protobuf message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumRunStreamRequest`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            Iterable[~cirq_google.engine.client.quantum_v1alpha1.types.QuantumRunStreamResponse].

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
                'quantum_run_stream'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.quantum_run_stream,
                default_retry=self._method_configs['QuantumRunStream'].retry,
                default_timeout=self._method_configs['QuantumRunStream'].timeout,
                client_info=self._client_info,
            )
            # Don't prefetch first result from stream, since this will cause deadlocks.
            # See https://github.com/googleapis/python-api-core/pull/30
            self.transport.quantum_run_stream._prefetch_first_result_ = False

        return self._inner_api_calls['quantum_run_stream'](
            requests, retry=retry, timeout=timeout, metadata=metadata
        )

    def list_quantum_reservation_grants(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        filter_: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_reservation_grants():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_reservation_grants().pages:
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
            filter_ (str): -
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservationGrant` instances.
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
        if 'list_quantum_reservation_grants' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_reservation_grants'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_reservation_grants,
                default_retry=self._method_configs['ListQuantumReservationGrants'].retry,
                default_timeout=self._method_configs['ListQuantumReservationGrants'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ListQuantumReservationGrantsRequest(
            parent=parent,
            page_size=page_size,
            filter=filter_,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_reservation_grants'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='reservation_grants',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def reallocate_quantum_reservation_grant(
        self,
        name: Optional[str] = None,
        source_project_id: Optional[str] = None,
        target_project_id: Optional[str] = None,
        duration: Union[Dict[str, Any], duration_pb2.Duration] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> response = client.reallocate_quantum_reservation_grant()

        Args:
            name (str): -
            source_project_id (str): -
            target_project_id (str): -
            duration (Union[dict, ~cirq_google.engine.client.quantum_v1alpha1.types.Duration]): -

                If a dict is provided, it must be of the same form as the protobuf
                message :class:`~cirq_google.engine.client.quantum_v1alpha1.types.Duration`
            retry (Optional[google.api_core.retry.Retry]):  A retry object used
                to retry requests. If ``None`` is specified, requests will
                be retried using a default configuration.
            timeout (Optional[float]): The amount of time, in seconds, to wait
                for the request to complete. Note that if ``retry`` is
                specified, the timeout applies to each individual attempt.
            metadata (Optional[Sequence[Tuple[str, str]]]): Additional metadata
                that is provided to the method.

        Returns:
            A :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservationGrant` instance.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: If the request
                    failed for any reason.
            google.api_core.exceptions.RetryError: If the request failed due
                    to a retryable error and retry attempts failed.
            ValueError: If the parameters are invalid.
        """
        # Wrap the transport method to add retry and timeout logic.
        if 'reallocate_quantum_reservation_grant' not in self._inner_api_calls:
            self._inner_api_calls[
                'reallocate_quantum_reservation_grant'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.reallocate_quantum_reservation_grant,
                default_retry=self._method_configs['ReallocateQuantumReservationGrant'].retry,
                default_timeout=self._method_configs['ReallocateQuantumReservationGrant'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ReallocateQuantumReservationGrantRequest(
            name=name,
            source_project_id=source_project_id,
            target_project_id=target_project_id,
            duration=duration,
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
                routing_header
            )
            metadata.append(routing_metadata)

        return self._inner_api_calls['reallocate_quantum_reservation_grant'](
            request, retry=retry, timeout=timeout, metadata=metadata
        )

    def list_quantum_reservation_budgets(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        filter_: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_reservation_budgets():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_reservation_budgets().pages:
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
            filter_ (str): -
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumReservationBudget` instances.
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
        if 'list_quantum_reservation_budgets' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_reservation_budgets'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_reservation_budgets,
                default_retry=self._method_configs['ListQuantumReservationBudgets'].retry,
                default_timeout=self._method_configs['ListQuantumReservationBudgets'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ListQuantumReservationBudgetsRequest(
            parent=parent,
            page_size=page_size,
            filter=filter_,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_reservation_budgets'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='reservation_budgets',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator

    def list_quantum_time_slots(
        self,
        parent: Optional[str] = None,
        page_size: Optional[int] = None,
        filter_: Optional[str] = None,
        retry: Optional[google.api_core.retry.Retry] = google.api_core.gapic_v1.method.DEFAULT,
        timeout: Optional[float] = google.api_core.gapic_v1.method.DEFAULT,
        metadata: Optional[Sequence[Tuple[str, str]]] = None,
    ):
        """
        -

        Example:
            >>> from cirq_google.engine.client import quantum_v1alpha1
            >>>
            >>> client = quantum_v1alpha1.QuantumEngineServiceClient()
            >>>
            >>> # Iterate over all results
            >>> for element in client.list_quantum_time_slots():
            ...     # process element
            ...     pass
            >>>
            >>>
            >>> # Alternatively:
            >>>
            >>> # Iterate over results one page at a time
            >>> for page in client.list_quantum_time_slots().pages:
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
            filter_ (str): -
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
            An iterable of :class:`~cirq_google.engine.client.quantum_v1alpha1.types.QuantumTimeSlot` instances.
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
        if 'list_quantum_time_slots' not in self._inner_api_calls:
            self._inner_api_calls[
                'list_quantum_time_slots'
            ] = google.api_core.gapic_v1.method.wrap_method(
                self.transport.list_quantum_time_slots,
                default_retry=self._method_configs['ListQuantumTimeSlots'].retry,
                default_timeout=self._method_configs['ListQuantumTimeSlots'].timeout,
                client_info=self._client_info,
            )

        request = engine_pb2.ListQuantumTimeSlotsRequest(
            parent=parent,
            page_size=page_size,
            filter=filter_,
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
                routing_header
            )
            metadata.append(routing_metadata)

        iterator = google.api_core.page_iterator.GRPCIterator(
            client=None,
            method=functools.partial(
                self._inner_api_calls['list_quantum_time_slots'],
                retry=retry,
                timeout=timeout,
                metadata=metadata,
            ),
            request=request,
            items_field='time_slots',
            request_token_field='page_token',
            response_token_field='next_page_token',
        )
        return iterator
