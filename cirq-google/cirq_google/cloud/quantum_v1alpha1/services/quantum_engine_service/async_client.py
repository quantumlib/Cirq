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
import importlib.util
import logging as std_logging
from typing import AsyncIterable, AsyncIterator, Awaitable, Callable, Optional, Sequence

import google.protobuf
from google.api_core import gapic_v1, retry_async as retries
from google.api_core.client_options import ClientOptions
from google.auth import credentials as ga_credentials

import cirq_google

try:
    OptionalRetry = retries.AsyncRetry | gapic_v1.method._MethodDefault | None
except AttributeError:  # pragma: NO COVER
    OptionalRetry = retries.AsyncRetry, object, None  # type: ignore

from cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service import pagers
from cirq_google.cloud.quantum_v1alpha1.types import engine, quantum

from .client import QuantumEngineServiceClient
from .transports.base import DEFAULT_CLIENT_INFO, QuantumEngineServiceTransport

CLIENT_LOGGING_SUPPORTED = importlib.util.find_spec("google.api_core.client_logging") is not None

_LOGGER = std_logging.getLogger(__name__)


class QuantumEngineServiceAsyncClient:
    """-"""

    _client: QuantumEngineServiceClient

    # Copy defaults from the synchronous client for use here.
    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = QuantumEngineServiceClient.DEFAULT_ENDPOINT
    DEFAULT_MTLS_ENDPOINT = QuantumEngineServiceClient.DEFAULT_MTLS_ENDPOINT
    _DEFAULT_ENDPOINT_TEMPLATE = QuantumEngineServiceClient._DEFAULT_ENDPOINT_TEMPLATE
    _DEFAULT_UNIVERSE = QuantumEngineServiceClient._DEFAULT_UNIVERSE

    quantum_job_path = staticmethod(QuantumEngineServiceClient.quantum_job_path)
    parse_quantum_job_path = staticmethod(QuantumEngineServiceClient.parse_quantum_job_path)
    quantum_processor_path = staticmethod(QuantumEngineServiceClient.quantum_processor_path)
    parse_quantum_processor_path = staticmethod(
        QuantumEngineServiceClient.parse_quantum_processor_path
    )
    quantum_processor_config_path = staticmethod(
        QuantumEngineServiceClient.quantum_processor_config_path
    )
    parse_quantum_processor_config_path = staticmethod(
        QuantumEngineServiceClient.parse_quantum_processor_config_path
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
        return QuantumEngineServiceClient.from_service_account_info.__func__(  # type: ignore
            QuantumEngineServiceAsyncClient, info, *args, **kwargs
        )

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
        return QuantumEngineServiceClient.from_service_account_file.__func__(  # type: ignore
            QuantumEngineServiceAsyncClient, filename, *args, **kwargs
        )

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

    @property
    def api_endpoint(self):
        """Return the API endpoint used by the client instance.

        Returns:
            str: The API endpoint used by the client instance.
        """
        return self._client._api_endpoint

    @property
    def universe_domain(self) -> str:
        """Return the universe domain used by the client instance.

        Returns:
            str: The universe domain used
                by the client instance.
        """
        return self._client._universe_domain

    get_transport_class = QuantumEngineServiceClient.get_transport_class

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            str | QuantumEngineServiceTransport | Callable[..., QuantumEngineServiceTransport]
        ] = "grpc_asyncio",
        client_options: Optional[ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the quantum engine service async client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (
                Optional[
                    str |
                    QuantumEngineServiceTransport |
                    Callable[..., QuantumEngineServiceTransport]
                ]
            ):
                The transport to use, or a Callable that constructs and returns a new transport to
                use. If a Callable is given, it will be called with the same set of initialization
                arguments as used in the QuantumEngineServiceTransport constructor.
                If set to None, a transport is chosen automatically.
            client_options (Optional[google.api_core.client_options.ClientOptions | dict]):
                Custom options for the client.

                1. The ``api_endpoint`` property can be used to override the
                default endpoint provided by the client when ``transport`` is
                not explicitly provided. Only if this property is not set and
                ``transport`` was not explicitly provided, the endpoint is
                determined by the GOOGLE_API_USE_MTLS_ENDPOINT environment
                variable, which have one of the following values:
                "always" (always use the default mTLS endpoint), "never" (always
                use the default regular endpoint) and "auto" (auto-switch to the
                default mTLS endpoint if client certificate is present; this is
                the default value).

                2. If the GOOGLE_API_USE_CLIENT_CERTIFICATE environment variable
                is "true", then the ``client_cert_source`` property can be used
                to provide a client certificate for mTLS transport. If
                not provided, the default SSL client certificate will be used if
                present. If GOOGLE_API_USE_CLIENT_CERTIFICATE is "false" or not
                set, no client certificate will be used.

                3. The ``universe_domain`` property can be used to override the
                default "googleapis.com" universe. Note that ``api_endpoint``
                property still takes precedence; and ``universe_domain`` is
                currently not supported for mTLS.

            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

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

        if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(std_logging.DEBUG):  # pragma: NO COVER
            _LOGGER.debug(
                (
                    "Created client `cirq_google.cloud.quantum_v1alpha1."
                    "QuantumEngineServiceAsyncClient`."
                ),
                extra=(
                    {
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "universeDomain": getattr(
                            self._client._transport._credentials, "universe_domain", ""
                        ),
                        "credentialsType": f"{type(self._client._transport._credentials).__module__}.{type(self._client._transport._credentials).__qualname__}",  # noqa E501
                        "credentialsInfo": getattr(
                            self.transport._credentials, "get_cred_info", lambda: None
                        )(),
                    }
                    if hasattr(self._client._transport, "_credentials")
                    else {
                        "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                        "credentialsType": None,
                    }
                ),
            )

    async def create_quantum_program(
        self,
        request: Optional[engine.CreateQuantumProgramRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumProgram:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_create_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CreateQuantumProgramRequest(
                )

                # Make the request
                response = await client.create_quantum_program(request=request)

                # Handle the response
                print(response)

        Args:
            request (
                Optional[
                    cirq_google.cloud.quantum_v1alpha1.types.CreateQuantumProgramRequest | dict
                ]
            ):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type
                `str`, but for metadata keys ending with the suffix `-bin`, the corresponding
                values must be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumProgram:
                -
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.CreateQuantumProgramRequest):
            request = engine.CreateQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_quantum_program
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def get_quantum_program(
        self,
        request: Optional[engine.GetQuantumProgramRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumProgram:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_get_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumProgramRequest(
                )

                # Make the request
                response = await client.get_quantum_program(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumProgramRequest | dict]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumProgram:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.GetQuantumProgramRequest):
            request = engine.GetQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.get_quantum_program]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_programs(
        self,
        request: Optional[engine.ListQuantumProgramsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumProgramsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_programs():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumProgramsRequest(
                )

                # Make the request
                page_result = client.list_quantum_programs(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumProgramsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumProgramsRequest):
            request = engine.ListQuantumProgramsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_quantum_programs
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumProgramsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def delete_quantum_program(
        self,
        request: Optional[engine.DeleteQuantumProgramRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> None:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_delete_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumProgramRequest(
                )

                # Make the request
                await client.delete_quantum_program(request=request)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.DeleteQuantumProgramRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.DeleteQuantumProgramRequest):
            request = engine.DeleteQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_quantum_program
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    async def update_quantum_program(
        self,
        request: Optional[engine.UpdateQuantumProgramRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumProgram:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_update_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.UpdateQuantumProgramRequest(
                )

                # Make the request
                response = await client.update_quantum_program(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.UpdateQuantumProgramRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumProgram:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.UpdateQuantumProgramRequest):
            request = engine.UpdateQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_quantum_program
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def create_quantum_job(
        self,
        request: Optional[engine.CreateQuantumJobRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumJob:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_create_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CreateQuantumJobRequest(
                )

                # Make the request
                response = await client.create_quantum_job(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.CreateQuantumJobRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.CreateQuantumJobRequest):
            request = engine.CreateQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.create_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def get_quantum_job(
        self,
        request: Optional[engine.GetQuantumJobRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumJob:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_get_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumJobRequest(
                )

                # Make the request
                response = await client.get_quantum_job(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumJobRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.GetQuantumJobRequest):
            request = engine.GetQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.get_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_jobs(
        self,
        request: Optional[engine.ListQuantumJobsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumJobsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_jobs():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumJobsRequest(
                )

                # Make the request
                page_result = client.list_quantum_jobs(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumJobsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumJobsRequest):
            request = engine.ListQuantumJobsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.list_quantum_jobs]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumJobsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def delete_quantum_job(
        self,
        request: Optional[engine.DeleteQuantumJobRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> None:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_delete_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumJobRequest(
                )

                # Make the request
                await client.delete_quantum_job(request=request)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.DeleteQuantumJobRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.DeleteQuantumJobRequest):
            request = engine.DeleteQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.delete_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    async def update_quantum_job(
        self,
        request: Optional[engine.UpdateQuantumJobRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumJob:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_update_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.UpdateQuantumJobRequest(
                )

                # Make the request
                response = await client.update_quantum_job(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.UpdateQuantumJobRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.UpdateQuantumJobRequest):
            request = engine.UpdateQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.update_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def cancel_quantum_job(
        self,
        request: Optional[engine.CancelQuantumJobRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> None:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_cancel_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CancelQuantumJobRequest(
                )

                # Make the request
                await client.cancel_quantum_job(request=request)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.CancelQuantumJobRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.CancelQuantumJobRequest):
            request = engine.CancelQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.cancel_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    async def list_quantum_job_events(
        self,
        request: Optional[engine.ListQuantumJobEventsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumJobEventsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_job_events():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumJobEventsRequest(
                )

                # Make the request
                page_result = client.list_quantum_job_events(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumJobEventsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumJobEventsRequest):
            request = engine.ListQuantumJobEventsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_quantum_job_events
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumJobEventsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_quantum_result(
        self,
        request: Optional[engine.GetQuantumResultRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumResult:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_get_quantum_result():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumResultRequest(
                )

                # Make the request
                response = await client.get_quantum_result(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumResultRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumResult:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.GetQuantumResultRequest):
            request = engine.GetQuantumResultRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.get_quantum_result]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_processors(
        self,
        request: Optional[engine.ListQuantumProcessorsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumProcessorsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_processors():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumProcessorsRequest(
                )

                # Make the request
                page_result = client.list_quantum_processors(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumProcessorsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumProcessorsRequest):
            request = engine.ListQuantumProcessorsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_quantum_processors
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumProcessorsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_quantum_processor(
        self,
        request: Optional[engine.GetQuantumProcessorRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumProcessor:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_get_quantum_processor():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumProcessorRequest(
                )

                # Make the request
                response = await client.get_quantum_processor(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumProcessorRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumProcessor:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.GetQuantumProcessorRequest):
            request = engine.GetQuantumProcessorRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_quantum_processor
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def get_quantum_processor_config(
        self,
        request: Optional[engine.GetQuantumProcessorConfigRequest | dict] = None,
        *,
        name: Optional[str] = None,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumProcessorConfig:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_get_quantum_processor_config():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumProcessorConfigRequest(
                    name="name_value",
                )

                # Make the request
                response = await client.get_quantum_processor_config(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumProcessorConfigRequest, dict]]):
                The request object. -
            name (:class:`str`):
                Required. -
                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumProcessorConfig:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Quick check: If we got a request object, we should *not* have
        #   gotten any keyword arguments that map to the request.
        flattened_params = [name]
        has_flattened_params = len([param for param in flattened_params if param is not None]) > 0
        if request is not None and has_flattened_params:
            raise ValueError(
                "If the `request` argument is set, then none of "
                "the individual field arguments should be set."
            )

        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.GetQuantumProcessorConfigRequest):
            request = engine.GetQuantumProcessorConfigRequest(request)

        # If we have keyword arguments corresponding to fields on the
        # request, apply these.
        if name is not None:
            request.name = name

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_quantum_processor_config
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_calibrations(
        self,
        request: Optional[engine.ListQuantumCalibrationsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumCalibrationsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_calibrations():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumCalibrationsRequest(
                )

                # Make the request
                page_result = client.list_quantum_calibrations(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumCalibrationsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumCalibrationsRequest):
            request = engine.ListQuantumCalibrationsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_quantum_calibrations
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumCalibrationsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def get_quantum_calibration(
        self,
        request: Optional[engine.GetQuantumCalibrationRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumCalibration:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_get_quantum_calibration():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumCalibrationRequest(
                )

                # Make the request
                response = await client.get_quantum_calibration(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumCalibrationRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumCalibration:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.GetQuantumCalibrationRequest):
            request = engine.GetQuantumCalibrationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_quantum_calibration
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def create_quantum_reservation(
        self,
        request: Optional[engine.CreateQuantumReservationRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumReservation:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_create_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CreateQuantumReservationRequest(
                )

                # Make the request
                response = await client.create_quantum_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.CreateQuantumReservationRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumReservation:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.CreateQuantumReservationRequest):
            request = engine.CreateQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.create_quantum_reservation
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def cancel_quantum_reservation(
        self,
        request: Optional[engine.CancelQuantumReservationRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumReservation:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_cancel_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CancelQuantumReservationRequest(
                )

                # Make the request
                response = await client.cancel_quantum_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.CancelQuantumReservationRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumReservation:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.CancelQuantumReservationRequest):
            request = engine.CancelQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.cancel_quantum_reservation
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def delete_quantum_reservation(
        self,
        request: Optional[engine.DeleteQuantumReservationRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> None:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_delete_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumReservationRequest(
                )

                # Make the request
                await client.delete_quantum_reservation(request=request)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.DeleteQuantumReservationRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.DeleteQuantumReservationRequest):
            request = engine.DeleteQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.delete_quantum_reservation
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    async def get_quantum_reservation(
        self,
        request: Optional[engine.GetQuantumReservationRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumReservation:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_get_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumReservationRequest(
                )

                # Make the request
                response = await client.get_quantum_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumReservationRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumReservation:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.GetQuantumReservationRequest):
            request = engine.GetQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.get_quantum_reservation
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_reservations(
        self,
        request: Optional[engine.ListQuantumReservationsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumReservationsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_reservations():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumReservationsRequest(
                )

                # Make the request
                page_result = client.list_quantum_reservations(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumReservationsRequest):
            request = engine.ListQuantumReservationsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_quantum_reservations
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumReservationsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def update_quantum_reservation(
        self,
        request: Optional[engine.UpdateQuantumReservationRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumReservation:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_update_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.UpdateQuantumReservationRequest(
                )

                # Make the request
                response = await client.update_quantum_reservation(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.UpdateQuantumReservationRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumReservation:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.UpdateQuantumReservationRequest):
            request = engine.UpdateQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.update_quantum_reservation
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def quantum_run_stream(
        self,
        requests: Optional[AsyncIterator[engine.QuantumRunStreamRequest]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> Awaitable[AsyncIterable[engine.QuantumRunStreamResponse]]:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_quantum_run_stream():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

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
                stream = await client.quantum_run_stream(requests=request_generator())

                # Handle the response
                async for response in stream:
                    print(response)

        Args:
            requests (AsyncIterator[`cirq_google.cloud.quantum_v1alpha1.types.QuantumRunStreamRequest`]):
                The request object AsyncIterator. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            AsyncIterable[cirq_google.cloud.quantum_v1alpha1.types.QuantumRunStreamResponse]:
                -
        """  # noqa E501

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[self._client._transport.quantum_run_stream]

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = rpc(requests, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_reservation_grants(
        self,
        request: Optional[engine.ListQuantumReservationGrantsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumReservationGrantsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_reservation_grants():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumReservationGrantsRequest(
                )

                # Make the request
                page_result = client.list_quantum_reservation_grants(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationGrantsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumReservationGrantsRequest):
            request = engine.ListQuantumReservationGrantsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_quantum_reservation_grants
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumReservationGrantsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def reallocate_quantum_reservation_grant(
        self,
        request: Optional[engine.ReallocateQuantumReservationGrantRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> quantum.QuantumReservationGrant:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_reallocate_quantum_reservation_grant():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ReallocateQuantumReservationGrantRequest(
                )

                # Make the request
                response = await client.reallocate_quantum_reservation_grant(request=request)

                # Handle the response
                print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ReallocateQuantumReservationGrantRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumReservationGrant:
                -
        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ReallocateQuantumReservationGrantRequest):
            request = engine.ReallocateQuantumReservationGrantRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.reallocate_quantum_reservation_grant
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    async def list_quantum_reservation_budgets(
        self,
        request: Optional[engine.ListQuantumReservationBudgetsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumReservationBudgetsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_reservation_budgets():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumReservationBudgetsRequest(
                )

                # Make the request
                page_result = client.list_quantum_reservation_budgets(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationBudgetsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumReservationBudgetsRequest):
            request = engine.ListQuantumReservationBudgetsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_quantum_reservation_budgets
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumReservationBudgetsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def list_quantum_time_slots(
        self,
        request: Optional[engine.ListQuantumTimeSlotsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumTimeSlotsAsyncPager:
        r"""-

        .. code-block:: python

            # This snippet has been automatically generated and should be regarded as a
            # code template only.
            # It will require modifications to work:
            # - It may require correct/in-range values for request initialization.
            # - It may require specifying regional endpoints when creating the service
            #   client as shown in:
            #   https://googleapis.dev/python/google-api-core/latest/client_options.html
            from google.cloud import quantum_v1alpha1

            async def sample_list_quantum_time_slots():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceAsyncClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.ListQuantumTimeSlotsRequest(
                )

                # Make the request
                page_result = client.list_quantum_time_slots(request=request)

                # Handle the response
                async for response in page_result:
                    print(response)

        Args:
            request (Optional[Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsRequest, dict]]):
                The request object. -
            retry (google.api_core.retry_async.AsyncRetry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str | bytes]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumTimeSlotsAsyncPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """  # noqa E501
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
        if not isinstance(request, engine.ListQuantumTimeSlotsRequest):
            request = engine.ListQuantumTimeSlotsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._client._transport._wrapped_methods[
            self._client._transport.list_quantum_time_slots
        ]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._client._validate_universe_domain()

        # Send the request.
        response = await rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__aiter__` convenience method.
        response = pagers.ListQuantumTimeSlotsAsyncPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    async def __aenter__(self) -> "QuantumEngineServiceAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(cirq_google.__version__)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__


__all__ = ("QuantumEngineServiceAsyncClient",)
