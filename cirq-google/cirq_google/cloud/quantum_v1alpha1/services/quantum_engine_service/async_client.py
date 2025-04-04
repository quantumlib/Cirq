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

import importlib.metadata
from typing import AsyncIterable, AsyncIterator, Awaitable, Optional, Sequence, Tuple, Union

from google.api_core import gapic_v1, retry as retries
from google.api_core.client_options import ClientOptions
from google.auth import credentials as ga_credentials

from cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service import pagers
from cirq_google.cloud.quantum_v1alpha1.types import engine, quantum

from .client import QuantumEngineServiceClient
from .transports.base import DEFAULT_CLIENT_INFO, QuantumEngineServiceTransport

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object]  # type: ignore


class QuantumEngineServiceAsyncClient:
    """-"""

    _client: QuantumEngineServiceClient

    DEFAULT_ENDPOINT = QuantumEngineServiceClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = QuantumEngineServiceClient.DEFAULT_MTLS_ENDPOINT

    quantum_job_path = staticmethod(QuantumEngineServiceClient.quantum_job_path)
    parse_quantum_job_path = staticmethod(QuantumEngineServiceClient.parse_quantum_job_path)
    quantum_processor_path = staticmethod(QuantumEngineServiceClient.quantum_processor_path)
    parse_quantum_processor_path = staticmethod(
        QuantumEngineServiceClient.parse_quantum_processor_path
    )
    quantum_program_path = staticmethod(QuantumEngineServiceClient.quantum_program_path)
    parse_quantum_program_path = staticmethod(QuantumEngineServiceClient.parse_quantum_program_path)
    quantum_reservation_path = staticmethod(QuantumEngineServiceClient.quantum_reservation_path)
    parse_quantum_reservation_path = staticmethod(
        QuantumEngineServiceClient.parse_quantum_reservation_path
    )
    common_billing_account_path = staticmethod(
        QuantumEngineServiceClient.common_billing_account_path
    )
    parse_common_billing_account_path = staticmethod(
        QuantumEngineServiceClient.parse_common_billing_account_path
    )
    common_folder_path = staticmethod(QuantumEngineServiceClient.common_folder_path)
    parse_common_folder_path = staticmethod(QuantumEngineServiceClient.parse_common_folder_path)
    common_organization_path = staticmethod(QuantumEngineServiceClient.common_organization_path)
    parse_common_organization_path = staticmethod(
        QuantumEngineServiceClient.parse_common_organization_path
    )
    common_project_path = staticmethod(QuantumEngineServiceClient.common_project_path)
    parse_common_project_path = staticmethod(QuantumEngineServiceClient.parse_common_project_path)
    common_location_path = staticmethod(QuantumEngineServiceClient.common_location_path)
    parse_common_location_path = staticmethod(QuantumEngineServiceClient.parse_common_location_path)

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            QuantumEngineServiceAsyncClient: The constructed client.
        """
        return QuantumEngineServiceClient.from_service_account_info.__func__(QuantumEngineServiceAsyncClient, info, *args, **kwargs)  # type: ignore

    @classmethod
    def from_service_account_file(cls, filename: str, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            file.

        Args:
            filename (str): The path to the service account private key json
                file.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            QuantumEngineServiceAsyncClient: The constructed client.
        """
        return QuantumEngineServiceClient.from_service_account_file.__func__(QuantumEngineServiceAsyncClient, filename, *args, **kwargs)  # type: ignore

    from_service_account_json = from_service_account_file

    @classmethod
    def get_mtls_endpoint_and_cert_source(cls, client_options: Optional[ClientOptions] = None):
        """Return the API endpoint and client cert source for mutual TLS.

        The client cert source is determined in the following order:
        (1) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is not "true", the
        client cert source is None.
        (2) if `client_options.client_cert_source` is provided, use the provided one; if the
        default client cert source exists, use the default one; otherwise the client cert
        source is None.

        The API endpoint is determined in the following order:
        (1) if `client_options.api_endpoint` if provided, use the provided one.
        (2) if `GOOGLE_API_USE_CLIENT_CERTIFICATE` environment variable is "always", use the
        default mTLS endpoint; if the environment variable is "never", use the default API
        endpoint; otherwise if client cert source exists, use the default mTLS endpoint, otherwise
        use the default API endpoint.

        More details can be found at https://google.aip.dev/auth/4114.

        Args:
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. Only the `api_endpoint` and `client_cert_source` properties may be used
                in this method.

        Returns:
            Tuple[str, Callable[[], Tuple[bytes, bytes]]]: returns the API endpoint and the
                client cert source to use.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If any errors happen.
        """
        return QuantumEngineServiceClient.get_mtls_endpoint_and_cert_source(client_options)

    @property
    def transport(self) -> QuantumEngineServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            QuantumEngineServiceTransport: The transport used by the client instance.
        """
        return self._client.transport

    get_transport_class = QuantumEngineServiceClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Union[str, QuantumEngineServiceTransport] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the quantum engine service client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Union[str, ~.QuantumEngineServiceTransport]): The
                transport to use. If set to None, a transport is chosen
                automatically.
            client_options (ClientOptions): Custom options for the client. It
                won't take effect if a ``transport`` instance is provided.
                (1) The ``api_endpoint`` property can be used to override the
                default endpoint provided by the client. GOOGLE_API_USE_MTLS_ENDPOINT
                environment variable can also be used to override the endpoint:
                "always" (always use the default mTLS endpoint), "never" (always
                use the default regular endpoint) and "auto" (auto switch to the
                default mTLS endpoint if client certificate is present, this is
                the default value). However, the ``api_endpoint`` property takes
                precedence if provided.
                (2) If GOOGLE_API_USE_CLIENT_CERTIFICATE environment variable
                is "true", then the ``client_cert_source`` property can be used
                to provide client certificate for mutual TLS transport. If
                not provided, the default SSL client certificate will be used if
                present. If GOOGLE_API_USE_CLIENT_CERTIFICATE is "false" or not
                set, no client certificate will be used.

        Raises:
            google.auth.exceptions.MutualTlsChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        self._client = QuantumEngineServiceClient(
            credentials=credentials,
            transport=transport,
            client_options=client_options,
            client_info=client_info,
        )

    async def create_quantum_program(
        self,
        request: Union[engine.CreateQuantumProgramRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumProgram:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_create_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CreateQuantumProgramRequest(
                )

                # Make the request
                response = client.create_quantum_program(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.CreateQuantumProgramRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumProgram:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.CreateQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.create_quantum_program,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def get_quantum_program(
        self,
        request: Union[engine.GetQuantumProgramRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumProgram:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_get_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumProgramRequest(
                )

                # Make the request
                response = client.get_quantum_program(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.GetQuantumProgramRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumProgram:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.GetQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.get_quantum_program,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_programs(
        self,
        request: Union[engine.ListQuantumProgramsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumProgramsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_programs():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumProgramsRequest(
                )

                # Make the request
                page_result = client.list_quantum_programs(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumProgramsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumProgramsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumProgramsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_programs,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumProgramsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def delete_quantum_program(
        self,
        request: Union[engine.DeleteQuantumProgramRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> None:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_delete_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumProgramRequest(
                )

                # Make the request
                client.delete_quantum_program(request=request)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.DeleteQuantumProgramRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        # Create or coerce a protobuf request object.
        request = engine.DeleteQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.delete_quantum_program,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    async def update_quantum_program(
        self,
        request: Union[engine.UpdateQuantumProgramRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumProgram:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_update_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.UpdateQuantumProgramRequest(
                )

                # Make the request
                response = client.update_quantum_program(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.UpdateQuantumProgramRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumProgram:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.UpdateQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.update_quantum_program,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def create_quantum_job(
        self,
        request: Union[engine.CreateQuantumJobRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumJob:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_create_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CreateQuantumJobRequest(
                )

                # Make the request
                response = client.create_quantum_job(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.CreateQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.CreateQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.create_quantum_job,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def get_quantum_job(
        self,
        request: Union[engine.GetQuantumJobRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumJob:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_get_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumJobRequest(
                )

                # Make the request
                response = client.get_quantum_job(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.GetQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.GetQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.get_quantum_job,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_jobs(
        self,
        request: Union[engine.ListQuantumJobsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumJobsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_jobs():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumJobsRequest(
                )

                # Make the request
                page_result = client.list_quantum_jobs(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumJobsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumJobsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumJobsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_jobs,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumJobsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def delete_quantum_job(
        self,
        request: Union[engine.DeleteQuantumJobRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> None:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_delete_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumJobRequest(
                )

                # Make the request
                client.delete_quantum_job(request=request)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.DeleteQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        # Create or coerce a protobuf request object.
        request = engine.DeleteQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.delete_quantum_job,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    async def update_quantum_job(
        self,
        request: Union[engine.UpdateQuantumJobRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumJob:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_update_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.UpdateQuantumJobRequest(
                )

                # Make the request
                response = client.update_quantum_job(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.UpdateQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.UpdateQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.update_quantum_job,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def cancel_quantum_job(
        self,
        request: Union[engine.CancelQuantumJobRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> None:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_cancel_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CancelQuantumJobRequest(
                )

                # Make the request
                client.cancel_quantum_job(request=request)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.CancelQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        # Create or coerce a protobuf request object.
        request = engine.CancelQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.cancel_quantum_job,
            default_timeout=None,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    async def list_quantum_job_events(
        self,
        request: Union[engine.ListQuantumJobEventsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumJobEventsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_job_events():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumJobEventsRequest(
                )

                # Make the request
                page_result = client.list_quantum_job_events(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumJobEventsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumJobEventsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_job_events,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumJobEventsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def get_quantum_result(
        self,
        request: Union[engine.GetQuantumResultRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumResult:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_get_quantum_result():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumResultRequest(
                )

                # Make the request
                response = client.get_quantum_result(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.GetQuantumResultRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumResult:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.GetQuantumResultRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.get_quantum_result,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_processors(
        self,
        request: Union[engine.ListQuantumProcessorsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumProcessorsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_processors():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumProcessorsRequest(
                )

                # Make the request
                page_result = client.list_quantum_processors(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumProcessorsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumProcessorsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_processors,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumProcessorsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def get_quantum_processor(
        self,
        request: Union[engine.GetQuantumProcessorRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumProcessor:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_get_quantum_processor():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumProcessorRequest(
                )

                # Make the request
                response = client.get_quantum_processor(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.GetQuantumProcessorRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumProcessor:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.GetQuantumProcessorRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.get_quantum_processor,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_calibrations(
        self,
        request: Union[engine.ListQuantumCalibrationsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumCalibrationsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_calibrations():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumCalibrationsRequest(
                )

                # Make the request
                page_result = client.list_quantum_calibrations(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumCalibrationsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumCalibrationsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_calibrations,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumCalibrationsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def get_quantum_calibration(
        self,
        request: Union[engine.GetQuantumCalibrationRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumCalibration:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_get_quantum_calibration():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumCalibrationRequest(
                )

                # Make the request
                response = client.get_quantum_calibration(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.GetQuantumCalibrationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumCalibration:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.GetQuantumCalibrationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.get_quantum_calibration,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def create_quantum_reservation(
        self,
        request: Union[engine.CreateQuantumReservationRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumReservation:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_create_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CreateQuantumReservationRequest(
                )

                # Make the request
                response = client.create_quantum_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.CreateQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumReservation:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.CreateQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.create_quantum_reservation,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def cancel_quantum_reservation(
        self,
        request: Union[engine.CancelQuantumReservationRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumReservation:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_cancel_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CancelQuantumReservationRequest(
                )

                # Make the request
                response = client.cancel_quantum_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.CancelQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumReservation:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.CancelQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.cancel_quantum_reservation,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def delete_quantum_reservation(
        self,
        request: Union[engine.DeleteQuantumReservationRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> None:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_delete_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumReservationRequest(
                )

                # Make the request
                client.delete_quantum_reservation(request=request)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.DeleteQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.
        """
        # Create or coerce a protobuf request object.
        request = engine.DeleteQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.delete_quantum_reservation,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    async def get_quantum_reservation(
        self,
        request: Union[engine.GetQuantumReservationRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumReservation:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_get_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumReservationRequest(
                )

                # Make the request
                response = client.get_quantum_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.GetQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumReservation:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.GetQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.get_quantum_reservation,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_reservations(
        self,
        request: Union[engine.ListQuantumReservationsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumReservationsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_reservations():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumReservationsRequest(
                )

                # Make the request
                page_result = client.list_quantum_reservations(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumReservationsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumReservationsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_reservations,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumReservationsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def update_quantum_reservation(
        self,
        request: Union[engine.UpdateQuantumReservationRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumReservation:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_update_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.UpdateQuantumReservationRequest(
                )

                # Make the request
                response = client.update_quantum_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.UpdateQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumReservation:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.UpdateQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.update_quantum_reservation,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def quantum_run_stream(
        self,
        requests: Optional[AsyncIterator[engine.QuantumRunStreamRequest]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> Awaitable[AsyncIterable[engine.QuantumRunStreamResponse]]:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_quantum_run_stream():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.QuantumRunStreamRequest(
                )

                # This method expects an iterator which contains
                # 'quantum_v1alpha1.QuantumRunStreamRequest' objects
                # Here we create a generator that yields a single `request` for
                # demonstrative purposes.
                requests = [request]

                def request_generator():
                    for request in requests:
                        yield request

                # Make the request
                stream = client.quantum_run_stream(requests=request_generator())

                # Handle the response
                for response in stream:
                    print(response)

        Args:
            requests (AsyncIterator[`google.cloud.quantum_v1alpha1.types.QuantumRunStreamRequest`]):
                The request object AsyncIterator. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            AsyncIterable[google.cloud.quantum_v1alpha1.types.QuantumRunStreamResponse]:
                -
        """

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.quantum_run_stream,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Send the request.
        response = rpc(requests, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_reservation_grants(
        self,
        request: Union[engine.ListQuantumReservationGrantsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumReservationGrantsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_reservation_grants():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumReservationGrantsRequest(
                )

                # Make the request
                page_result = client.list_quantum_reservation_grants(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationGrantsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumReservationGrantsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_reservation_grants,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumReservationGrantsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def reallocate_quantum_reservation_grant(
        self,
        request: Union[engine.ReallocateQuantumReservationGrantRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> quantum.QuantumReservationGrant:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_reallocate_quantum_reservation_grant():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ReallocateQuantumReservationGrantRequest(
                )

                # Make the request
                response = client.reallocate_quantum_reservation_grant(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ReallocateQuantumReservationGrantRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.types.QuantumReservationGrant:
                -
        """
        # Create or coerce a protobuf request object.
        request = engine.ReallocateQuantumReservationGrantRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.reallocate_quantum_reservation_grant,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_reservation_budgets(
        self,
        request: Union[engine.ListQuantumReservationBudgetsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumReservationBudgetsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_reservation_budgets():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumReservationBudgetsRequest(
                )

                # Make the request
                page_result = client.list_quantum_reservation_budgets(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationBudgetsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumReservationBudgetsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_reservation_budgets,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumReservationBudgetsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def list_quantum_time_slots(
        self,
        request: Union[engine.ListQuantumTimeSlotsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumTimeSlotsAsyncPager:
        r"""-

        .. code-block:: python

            from google.cloud import quantum_v1alpha1

            def sample_list_quantum_time_slots():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumTimeSlotsRequest(
                )

                # Make the request
                page_result = client.list_quantum_time_slots(request=request)

                # Handle the response
                for response in page_result:
                    print(response)

        Args:
            request (Union[google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumTimeSlotsAsyncPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        request = engine.ListQuantumTimeSlotsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = gapic_v1.method_async.wrap_method(
            self._client._transport.list_quantum_time_slots,
            default_timeout=60.0,
            client_info=DEFAULT_CLIENT_INFO,
        )

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumTimeSlotsAsyncPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


try:
    DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
        gapic_version=importlib.metadata.version("google-cloud-quantum")
    )
except ModuleNotFoundError:
    DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo()


__all__ = ("QuantumEngineServiceAsyncClient",)
