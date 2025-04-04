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
import os
import re
from collections import OrderedDict
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple, Type, Union

from google.api_core import client_options as client_options_lib, gapic_v1, retry as retries
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.auth.transport import mtls
from google.oauth2 import service_account

from cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service import pagers
from cirq_google.cloud.quantum_v1alpha1.types import engine, quantum

from .transports.base import DEFAULT_CLIENT_INFO, QuantumEngineServiceTransport
from .transports.grpc import QuantumEngineServiceGrpcTransport
from .transports.grpc_asyncio import QuantumEngineServiceGrpcAsyncIOTransport

try:
    OptionalRetry = Union[retries.Retry, gapic_v1.method._MethodDefault]
except AttributeError:  # pragma: NO COVER
    OptionalRetry = Union[retries.Retry, object]  # type: ignore


class QuantumEngineServiceClientMeta(type):
    """Metaclass for the QuantumEngineService client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """

    _transport_registry = OrderedDict()  # type: Dict[str, Type[QuantumEngineServiceTransport]]
    _transport_registry["grpc"] = QuantumEngineServiceGrpcTransport
    _transport_registry["grpc_asyncio"] = QuantumEngineServiceGrpcAsyncIOTransport

    def get_transport_class(
        cls, label: Optional[str] = None
    ) -> Type[QuantumEngineServiceTransport]:
        """Returns an appropriate transport class.

        Args:
            label: The name of the desired transport. If none is
                provided, then the first transport in the registry is used.

        Returns:
            The transport class to use.
        """
        # If a specific transport is requested, return that one.
        if label:
            return cls._transport_registry[label]

        # No transport is requested; return the default (that is, the first one
        # in the dictionary).
        return next(iter(cls._transport_registry.values()))


class QuantumEngineServiceClient(metaclass=QuantumEngineServiceClientMeta):
    """-"""

    @staticmethod
    def _get_default_mtls_endpoint(api_endpoint):
        """Converts api endpoint to mTLS endpoint.

        Convert "*.sandbox.googleapis.com" and "*.googleapis.com" to
        "*.mtls.sandbox.googleapis.com" and "*.mtls.googleapis.com" respectively.
        Args:
            api_endpoint (Optional[str]): the api endpoint to convert.
        Returns:
            str: converted mTLS api endpoint.
        """
        if not api_endpoint:
            return api_endpoint

        mtls_endpoint_re = re.compile(
            r"(?P<name>[^.]+)(?P<mtls>\.mtls)?(?P<sandbox>\.sandbox)?(?P<googledomain>\.googleapis\.com)?"
        )

        m = mtls_endpoint_re.match(api_endpoint)
        _, mtls, sandbox, googledomain = m.groups()
        if mtls or not googledomain:
            return api_endpoint

        if sandbox:
            return api_endpoint.replace("sandbox.googleapis.com", "mtls.sandbox.googleapis.com")

        return api_endpoint.replace(".googleapis.com", ".mtls.googleapis.com")

    DEFAULT_ENDPOINT = "quantum.googleapis.com"
    DEFAULT_MTLS_ENDPOINT = _get_default_mtls_endpoint.__func__(DEFAULT_ENDPOINT)  # type: ignore

    @classmethod
    def from_service_account_info(cls, info: dict, *args, **kwargs):
        """Creates an instance of this client using the provided credentials
            info.

        Args:
            info (dict): The service account private key info.
            args: Additional arguments to pass to the constructor.
            kwargs: Additional arguments to pass to the constructor.

        Returns:
            QuantumEngineServiceClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_info(info)
        kwargs["credentials"] = credentials
        return cls(*args, **kwargs)

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
            QuantumEngineServiceClient: The constructed client.
        """
        credentials = service_account.Credentials.from_service_account_file(filename)
        kwargs["credentials"] = credentials
        return cls(*args, **kwargs)

    from_service_account_json = from_service_account_file

    @property
    def transport(self) -> QuantumEngineServiceTransport:
        """Returns the transport used by the client instance.

        Returns:
            QuantumEngineServiceTransport: The transport used by the client
                instance.
        """
        return self._transport

    @staticmethod
    def quantum_job_path(project: str, program: str, job: str) -> str:
        """Returns a fully-qualified quantum_job string."""
        return f"projects/{project}/programs/{program}/jobs/{job}"

    @staticmethod
    def parse_quantum_job_path(path: str) -> Dict[str, str]:
        """Parses a quantum_job path into its component segments."""
        m = re.match(
            r"^projects/(?P<project>.+?)/programs/(?P<program>.+?)/jobs/(?P<job>.+?)$", path
        )
        return m.groupdict() if m else {}

    @staticmethod
    def quantum_processor_path(project_id: str, processor_id: str) -> str:
        """Returns a fully-qualified quantum_processor string."""
        return f"projects/{project_id}/processors/{processor_id}"

    @staticmethod
    def parse_quantum_processor_path(path: str) -> Dict[str, str]:
        """Parses a quantum_processor path into its component segments."""
        m = re.match(r"^projects/(?P<project_id>.+?)/processors/(?P<processor_id>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def quantum_program_path(project: str, program: str) -> str:
        """Returns a fully-qualified quantum_program string."""
        return f"projects/{project}/programs/{program}"

    @staticmethod
    def parse_quantum_program_path(path: str) -> Dict[str, str]:
        """Parses a quantum_program path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)/programs/(?P<program>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def quantum_reservation_path(project_id: str, processor_id: str, reservation_id: str) -> str:
        """Returns a fully-qualified quantum_reservation string."""
        return f"projects/{project_id}/processors/{processor_id}/reservations/{reservation_id}"

    @staticmethod
    def parse_quantum_reservation_path(path: str) -> Dict[str, str]:
        """Parses a quantum_reservation path into its component segments."""
        m = re.match(
            r"^projects/(?P<project_id>.+?)/processors/(?P<processor_id>.+?)/reservations/(?P<reservation_id>.+?)$",
            path,
        )
        return m.groupdict() if m else {}

    @staticmethod
    def common_billing_account_path(billing_account: str) -> str:
        """Returns a fully-qualified billing_account string."""
        return f"billingAccounts/{billing_account}"

    @staticmethod
    def parse_common_billing_account_path(path: str) -> Dict[str, str]:
        """Parse a billing_account path into its component segments."""
        m = re.match(r"^billingAccounts/(?P<billing_account>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_folder_path(folder: str) -> str:
        """Returns a fully-qualified folder string."""
        return f"folders/{folder}"

    @staticmethod
    def parse_common_folder_path(path: str) -> Dict[str, str]:
        """Parse a folder path into its component segments."""
        m = re.match(r"^folders/(?P<folder>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_organization_path(organization: str) -> str:
        """Returns a fully-qualified organization string."""
        return f"organizations/{organization}"

    @staticmethod
    def parse_common_organization_path(path: str) -> Dict[str, str]:
        """Parse a organization path into its component segments."""
        m = re.match(r"^organizations/(?P<organization>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_project_path(project: str) -> str:
        """Returns a fully-qualified project string."""
        return f"projects/{project}"

    @staticmethod
    def parse_common_project_path(path: str) -> Dict[str, str]:
        """Parse a project path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_location_path(project: str, location: str) -> str:
        """Returns a fully-qualified location string."""
        return f"projects/{project}/locations/{location}"

    @staticmethod
    def parse_common_location_path(path: str) -> Dict[str, str]:
        """Parse a location path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)/locations/(?P<location>.+?)$", path)
        return m.groupdict() if m else {}

    @classmethod
    def get_mtls_endpoint_and_cert_source(
        cls, client_options: Optional[client_options_lib.ClientOptions] = None
    ):
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
        if client_options is None:
            client_options = client_options_lib.ClientOptions()
        use_client_cert = os.getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
        use_mtls_endpoint = os.getenv("GOOGLE_API_USE_MTLS_ENDPOINT", "auto")
        if use_client_cert not in ("true", "false"):
            raise ValueError(
                "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` or `false`"
            )
        if use_mtls_endpoint not in ("auto", "never", "always"):
            raise MutualTLSChannelError(
                "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or `always`"
            )

        # Figure out the client cert source to use.
        client_cert_source = None
        if use_client_cert == "true":
            if client_options.client_cert_source:
                client_cert_source = client_options.client_cert_source
            elif mtls.has_default_client_cert_source():
                client_cert_source = mtls.default_client_cert_source()

        # Figure out which api endpoint to use.
        if client_options.api_endpoint is not None:
            api_endpoint = client_options.api_endpoint
        elif use_mtls_endpoint == "always" or (use_mtls_endpoint == "auto" and client_cert_source):
            api_endpoint = cls.DEFAULT_MTLS_ENDPOINT
        else:
            api_endpoint = cls.DEFAULT_ENDPOINT

        return api_endpoint, client_cert_source

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Union[str, QuantumEngineServiceTransport, None] = None,
        client_options: Optional[client_options_lib.ClientOptions] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the quantum engine service client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Union[str, QuantumEngineServiceTransport]): The
                transport to use. If set to None, a transport is chosen
                automatically.
            client_options (google.api_core.client_options.ClientOptions): Custom options for the
                client. It won't take effect if a ``transport`` instance is provided.
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
            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
                creation failed for any reason.
        """
        if isinstance(client_options, dict):
            client_options = client_options_lib.from_dict(client_options)
        if client_options is None:
            client_options = client_options_lib.ClientOptions()

        api_endpoint, client_cert_source_func = self.get_mtls_endpoint_and_cert_source(
            client_options
        )

        api_key_value = getattr(client_options, "api_key", None)
        if api_key_value and credentials:
            raise ValueError("client_options.api_key and credentials are mutually exclusive")

        # Save or instantiate the transport.
        # Ordinarily, we provide the transport, but allowing a custom transport
        # instance provides an extensibility point for unusual situations.
        if isinstance(transport, QuantumEngineServiceTransport):
            # transport is a QuantumEngineServiceTransport instance.
            if credentials or client_options.credentials_file or api_key_value:
                raise ValueError(
                    "When providing a transport instance, " "provide its credentials directly."
                )
            if client_options.scopes:
                raise ValueError(
                    "When providing a transport instance, provide its scopes " "directly."
                )
            self._transport = transport
        else:
            import google.auth._default

            if api_key_value and hasattr(google.auth._default, "get_api_key_credentials"):
                credentials = google.auth._default.get_api_key_credentials(api_key_value)

            Transport = type(self).get_transport_class(transport)
            self._transport = Transport(
                credentials=credentials,
                credentials_file=client_options.credentials_file,
                host=api_endpoint,
                scopes=client_options.scopes,
                client_cert_source_for_mtls=client_cert_source_func,
                quota_project_id=client_options.quota_project_id,
                client_info=client_info,
                always_use_jwt_access=True,
            )

    def create_quantum_program(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.CreateQuantumProgramRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.CreateQuantumProgramRequest):
            request = engine.CreateQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.create_quantum_program]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def get_quantum_program(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.GetQuantumProgramRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.GetQuantumProgramRequest):
            request = engine.GetQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_program]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_programs(
        self,
        request: Union[engine.ListQuantumProgramsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumProgramsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumProgramsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumProgramsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumProgramsRequest):
            request = engine.ListQuantumProgramsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_programs]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumProgramsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def delete_quantum_program(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.DeleteQuantumProgramRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.DeleteQuantumProgramRequest):
            request = engine.DeleteQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_quantum_program]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def update_quantum_program(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.UpdateQuantumProgramRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.UpdateQuantumProgramRequest):
            request = engine.UpdateQuantumProgramRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.update_quantum_program]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def create_quantum_job(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.CreateQuantumJobRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.CreateQuantumJobRequest):
            request = engine.CreateQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.create_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def get_quantum_job(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.GetQuantumJobRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.GetQuantumJobRequest):
            request = engine.GetQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_jobs(
        self,
        request: Union[engine.ListQuantumJobsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumJobsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumJobsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumJobsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumJobsRequest):
            request = engine.ListQuantumJobsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_jobs]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumJobsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def delete_quantum_job(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.DeleteQuantumJobRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.DeleteQuantumJobRequest):
            request = engine.DeleteQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def update_quantum_job(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.UpdateQuantumJobRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.UpdateQuantumJobRequest):
            request = engine.UpdateQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.update_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def cancel_quantum_job(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.CancelQuantumJobRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.CancelQuantumJobRequest):
            request = engine.CancelQuantumJobRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.cancel_quantum_job]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def list_quantum_job_events(
        self,
        request: Union[engine.ListQuantumJobEventsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumJobEventsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumJobEventsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumJobEventsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumJobEventsRequest):
            request = engine.ListQuantumJobEventsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_job_events]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumJobEventsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def get_quantum_result(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.GetQuantumResultRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.GetQuantumResultRequest):
            request = engine.GetQuantumResultRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_result]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_processors(
        self,
        request: Union[engine.ListQuantumProcessorsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumProcessorsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumProcessorsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumProcessorsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumProcessorsRequest):
            request = engine.ListQuantumProcessorsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_processors]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumProcessorsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def get_quantum_processor(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.GetQuantumProcessorRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.GetQuantumProcessorRequest):
            request = engine.GetQuantumProcessorRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_processor]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_calibrations(
        self,
        request: Union[engine.ListQuantumCalibrationsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumCalibrationsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumCalibrationsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumCalibrationsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumCalibrationsRequest):
            request = engine.ListQuantumCalibrationsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_calibrations]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumCalibrationsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def get_quantum_calibration(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.GetQuantumCalibrationRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.GetQuantumCalibrationRequest):
            request = engine.GetQuantumCalibrationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_calibration]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def create_quantum_reservation(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.CreateQuantumReservationRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.CreateQuantumReservationRequest):
            request = engine.CreateQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.create_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def cancel_quantum_reservation(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.CancelQuantumReservationRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.CancelQuantumReservationRequest):
            request = engine.CancelQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.cancel_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def delete_quantum_reservation(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.DeleteQuantumReservationRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.DeleteQuantumReservationRequest):
            request = engine.DeleteQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.delete_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def get_quantum_reservation(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.GetQuantumReservationRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.GetQuantumReservationRequest):
            request = engine.GetQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_reservations(
        self,
        request: Union[engine.ListQuantumReservationsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumReservationsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumReservationsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumReservationsRequest):
            request = engine.ListQuantumReservationsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_reservations]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumReservationsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def update_quantum_reservation(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.UpdateQuantumReservationRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.UpdateQuantumReservationRequest):
            request = engine.UpdateQuantumReservationRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.update_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def quantum_run_stream(
        self,
        requests: Optional[Iterator[engine.QuantumRunStreamRequest]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> Iterable[engine.QuantumRunStreamResponse]:
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
            requests (Iterator[google.cloud.quantum_v1alpha1.types.QuantumRunStreamRequest]):
                The request object iterator. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, str]]): Strings which should be
                sent along with the request as metadata.

        Returns:
            Iterable[google.cloud.quantum_v1alpha1.types.QuantumRunStreamResponse]:
                -
        """

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.quantum_run_stream]

        # Send the request.
        response = rpc(requests, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_reservation_grants(
        self,
        request: Union[engine.ListQuantumReservationGrantsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumReservationGrantsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationGrantsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumReservationGrantsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumReservationGrantsRequest):
            request = engine.ListQuantumReservationGrantsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_reservation_grants]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumReservationGrantsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def reallocate_quantum_reservation_grant(
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
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ReallocateQuantumReservationGrantRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ReallocateQuantumReservationGrantRequest):
            request = engine.ReallocateQuantumReservationGrantRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.reallocate_quantum_reservation_grant]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_reservation_budgets(
        self,
        request: Union[engine.ListQuantumReservationBudgetsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumReservationBudgetsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationBudgetsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumReservationBudgetsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumReservationBudgetsRequest):
            request = engine.ListQuantumReservationBudgetsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_reservation_budgets]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumReservationBudgetsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def list_quantum_time_slots(
        self,
        request: Union[engine.ListQuantumTimeSlotsRequest, dict, None] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: Optional[float] = None,
        metadata: Sequence[Tuple[str, str]] = (),
    ) -> pagers.ListQuantumTimeSlotsPager:
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
            google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumTimeSlotsPager:
                -
                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # Minor optimization to avoid making a copy if the user passes
        # in a engine.ListQuantumTimeSlotsRequest.
        # There's no risk of modifying the input as we've already verified
        # there are no flattened fields.
        if not isinstance(request, engine.ListQuantumTimeSlotsRequest):
            request = engine.ListQuantumTimeSlotsRequest(request)

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_time_slots]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumTimeSlotsPager(
            method=rpc, request=request, response=response, metadata=metadata
        )

        # Done; return the response.
        return response

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        """Releases underlying transport's resources.

        .. warning::
            ONLY use as a context manager if the transport is NOT shared
            with other clients! Exiting the with block will CLOSE the transport
            and may cause errors in other clients!
        """
        self.transport.close()


try:
    DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(
        gapic_version=importlib.metadata.version("google-cloud-quantum")
    )
except ModuleNotFoundError:
    DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo()


__all__ = ("QuantumEngineServiceClient",)
