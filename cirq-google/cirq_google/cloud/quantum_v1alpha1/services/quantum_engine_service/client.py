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
import json
import logging as std_logging
import os
import re
import warnings
from collections import OrderedDict
from http import HTTPStatus
from typing import Callable, cast, Iterable, Iterator, Optional, Sequence

import google.protobuf
from google.api_core import (
    client_options as client_options_lib,
    exceptions as core_exceptions,
    gapic_v1,
    retry as retries,
)
from google.auth import credentials as ga_credentials
from google.auth.exceptions import MutualTLSChannelError
from google.auth.transport import mtls
from google.oauth2 import service_account

import cirq_google
from cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service import pagers
from cirq_google.cloud.quantum_v1alpha1.types import engine, quantum

from .transports.base import DEFAULT_CLIENT_INFO, QuantumEngineServiceTransport
from .transports.grpc import QuantumEngineServiceGrpcTransport
from .transports.grpc_asyncio import QuantumEngineServiceGrpcAsyncIOTransport
from .transports.rest import QuantumEngineServiceRestTransport

try:
    OptionalRetry = retries.Retry | gapic_v1.method._MethodDefault | None
except AttributeError:  # pragma: NO COVER
    OptionalRetry = retries.Retry | object | None  # type: ignore

try:
    from google.api_core import client_logging

    CLIENT_LOGGING_SUPPORTED = True  # pragma: NO COVER
except ImportError:  # pragma: NO COVER
    CLIENT_LOGGING_SUPPORTED = False

_LOGGER = std_logging.getLogger(__name__)


class QuantumEngineServiceClientMeta(type):
    """Metaclass for the QuantumEngineService client.

    This provides class-level methods for building and retrieving
    support objects (e.g. transport) without polluting the client instance
    objects.
    """

    _transport_registry: dict[str, type[QuantumEngineServiceTransport]] = OrderedDict()
    _transport_registry["grpc"] = QuantumEngineServiceGrpcTransport
    _transport_registry["grpc_asyncio"] = QuantumEngineServiceGrpcAsyncIOTransport
    _transport_registry["rest"] = QuantumEngineServiceRestTransport

    def get_transport_class(
        cls, label: Optional[str] = None
    ) -> type[QuantumEngineServiceTransport]:
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
        name, mtls, sandbox, googledomain = m.groups()
        if mtls or not googledomain:
            return api_endpoint

        if sandbox:
            return api_endpoint.replace("sandbox.googleapis.com", "mtls.sandbox.googleapis.com")

        return api_endpoint.replace(".googleapis.com", ".mtls.googleapis.com")

    # Note: DEFAULT_ENDPOINT is deprecated. Use _DEFAULT_ENDPOINT_TEMPLATE instead.
    DEFAULT_ENDPOINT = "quantum.googleapis.com"
    DEFAULT_MTLS_ENDPOINT = _get_default_mtls_endpoint.__func__(DEFAULT_ENDPOINT)  # type: ignore

    _DEFAULT_ENDPOINT_TEMPLATE = "quantum.{UNIVERSE_DOMAIN}"
    _DEFAULT_UNIVERSE = "googleapis.com"

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
        return "projects/{project}/programs/{program}/jobs/{job}".format(
            project=project, program=program, job=job
        )

    @staticmethod
    def parse_quantum_job_path(path: str) -> dict[str, str]:
        """Parses a quantum_job path into its component segments."""
        m = re.match(
            r"^projects/(?P<project>.+?)/programs/(?P<program>.+?)/jobs/(?P<job>.+?)$", path
        )
        return m.groupdict() if m else {}

    @staticmethod
    def quantum_processor_path(project_id: str, processor_id: str) -> str:
        """Returns a fully-qualified quantum_processor string."""
        return "projects/{project_id}/processors/{processor_id}".format(
            project_id=project_id, processor_id=processor_id
        )

    @staticmethod
    def parse_quantum_processor_path(path: str) -> dict[str, str]:
        """Parses a quantum_processor path into its component segments."""
        m = re.match(r"^projects/(?P<project_id>.+?)/processors/(?P<processor_id>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def quantum_processor_config_path(
        project: str, processor: str, snapshot_id: str, quantum_processor_config: str
    ) -> str:
        """Returns a fully-qualified quantum_processor_config string."""
        return "projects/{project}/processors/{processor}/configSnapshots/{snapshot_id}/configs/{quantum_processor_config}".format(  # noqa E501
            project=project,
            processor=processor,
            snapshot_id=snapshot_id,
            quantum_processor_config=quantum_processor_config,
        )

    @staticmethod
    def parse_quantum_processor_config_path(path: str) -> dict[str, str]:
        """Parses a quantum_processor_config path into its component segments."""
        m = re.match(
            r"^projects/(?P<project>.+?)/processors/(?P<processor>.+?)/configSnapshots/(?P<snapshot_id>.+?)/configs/(?P<quantum_processor_config>.+?)$",
            path,
        )
        return m.groupdict() if m else {}

    @staticmethod
    def quantum_program_path(project: str, program: str) -> str:
        """Returns a fully-qualified quantum_program string."""
        return "projects/{project}/programs/{program}".format(project=project, program=program)

    @staticmethod
    def parse_quantum_program_path(path: str) -> dict[str, str]:
        """Parses a quantum_program path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)/programs/(?P<program>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def quantum_reservation_path(project_id: str, processor_id: str, reservation_id: str) -> str:
        """Returns a fully-qualified quantum_reservation string."""
        return (
            "projects/{project_id}/processors/{processor_id}/reservations/{reservation_id}".format(
                project_id=project_id, processor_id=processor_id, reservation_id=reservation_id
            )
        )

    @staticmethod
    def parse_quantum_reservation_path(path: str) -> dict[str, str]:
        """Parses a quantum_reservation path into its component segments."""
        m = re.match(
            r"^projects/(?P<project_id>.+?)/processors/(?P<processor_id>.+?)/reservations/(?P<reservation_id>.+?)$",
            path,
        )
        return m.groupdict() if m else {}

    @staticmethod
    def common_billing_account_path(billing_account: str) -> str:
        """Returns a fully-qualified billing_account string."""
        return "billingAccounts/{billing_account}".format(billing_account=billing_account)

    @staticmethod
    def parse_common_billing_account_path(path: str) -> dict[str, str]:
        """Parse a billing_account path into its component segments."""
        m = re.match(r"^billingAccounts/(?P<billing_account>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_folder_path(folder: str) -> str:
        """Returns a fully-qualified folder string."""
        return "folders/{folder}".format(folder=folder)

    @staticmethod
    def parse_common_folder_path(path: str) -> dict[str, str]:
        """Parse a folder path into its component segments."""
        m = re.match(r"^folders/(?P<folder>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_organization_path(organization: str) -> str:
        """Returns a fully-qualified organization string."""
        return "organizations/{organization}".format(organization=organization)

    @staticmethod
    def parse_common_organization_path(path: str) -> dict[str, str]:
        """Parse a organization path into its component segments."""
        m = re.match(r"^organizations/(?P<organization>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_project_path(project: str) -> str:
        """Returns a fully-qualified project string."""
        return "projects/{project}".format(project=project)

    @staticmethod
    def parse_common_project_path(path: str) -> dict[str, str]:
        """Parse a project path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)$", path)
        return m.groupdict() if m else {}

    @staticmethod
    def common_location_path(project: str, location: str) -> str:
        """Returns a fully-qualified location string."""
        return "projects/{project}/locations/{location}".format(project=project, location=location)

    @staticmethod
    def parse_common_location_path(path: str) -> dict[str, str]:
        """Parse a location path into its component segments."""
        m = re.match(r"^projects/(?P<project>.+?)/locations/(?P<location>.+?)$", path)
        return m.groupdict() if m else {}

    @classmethod
    def get_mtls_endpoint_and_cert_source(
        cls, client_options: Optional[client_options_lib.ClientOptions] = None
    ):
        """Deprecated. Return the API endpoint and client cert source for mutual TLS.

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

        warnings.warn(
            (
                "get_mtls_endpoint_and_cert_source is deprecated. Use the api_endpoint property "
                "instead."
            ),
            DeprecationWarning,
        )
        if client_options is None:
            client_options = client_options_lib.ClientOptions()
        use_client_cert = os.getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false")
        use_mtls_endpoint = os.getenv("GOOGLE_API_USE_MTLS_ENDPOINT", "auto")
        if use_client_cert not in ("true", "false"):
            raise ValueError(
                "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` "
                "or `false`"
            )
        if use_mtls_endpoint not in ("auto", "never", "always"):
            raise MutualTLSChannelError(
                "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or "
                "`always`"
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

    @staticmethod
    def _read_environment_variables():
        """Returns the environment variables used by the client.

        Returns:
            Tuple[bool, str, str]: returns the GOOGLE_API_USE_CLIENT_CERTIFICATE,
            GOOGLE_API_USE_MTLS_ENDPOINT, and GOOGLE_CLOUD_UNIVERSE_DOMAIN environment variables.

        Raises:
            ValueError: If GOOGLE_API_USE_CLIENT_CERTIFICATE is not
                any of ["true", "false"].
            google.auth.exceptions.MutualTLSChannelError: If GOOGLE_API_USE_MTLS_ENDPOINT
                is not any of ["auto", "never", "always"].
        """
        use_client_cert = os.getenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "false").lower()
        use_mtls_endpoint = os.getenv("GOOGLE_API_USE_MTLS_ENDPOINT", "auto").lower()
        universe_domain_env = os.getenv("GOOGLE_CLOUD_UNIVERSE_DOMAIN")
        if use_client_cert not in ("true", "false"):
            raise ValueError(
                "Environment variable `GOOGLE_API_USE_CLIENT_CERTIFICATE` must be either `true` or "
                "`false`"
            )
        if use_mtls_endpoint not in ("auto", "never", "always"):
            raise MutualTLSChannelError(
                "Environment variable `GOOGLE_API_USE_MTLS_ENDPOINT` must be `never`, `auto` or "
                "`always`"
            )
        return use_client_cert == "true", use_mtls_endpoint, universe_domain_env

    @staticmethod
    def _get_client_cert_source(provided_cert_source, use_cert_flag):
        """Return the client cert source to be used by the client.

        Args:
            provided_cert_source (bytes): The client certificate source provided.
            use_cert_flag (bool): A flag indicating whether to use the client certificate.

        Returns:
            bytes or None: The client cert source to be used by the client.
        """
        client_cert_source = None
        if use_cert_flag:
            if provided_cert_source:
                client_cert_source = provided_cert_source
            elif mtls.has_default_client_cert_source():
                client_cert_source = mtls.default_client_cert_source()
        return client_cert_source

    @staticmethod
    def _get_api_endpoint(api_override, client_cert_source, universe_domain, use_mtls_endpoint):
        """Return the API endpoint used by the client.

        Args:
            api_override (str): The API endpoint override. If specified, this is always
                the return value of this function and the other arguments are not used.
            client_cert_source (bytes): The client certificate source used by the client.
            universe_domain (str): The universe domain used by the client.
            use_mtls_endpoint (str): How to use the mTLS endpoint, which depends also on the other
                parameters. Possible values are "always", "auto", or "never".

        Returns:
            str: The API endpoint to be used by the client.
        """
        if api_override is not None:
            api_endpoint = api_override
        elif use_mtls_endpoint == "always" or (use_mtls_endpoint == "auto" and client_cert_source):
            _default_universe = QuantumEngineServiceClient._DEFAULT_UNIVERSE
            if universe_domain != _default_universe:
                raise MutualTLSChannelError(
                    f"mTLS is not supported in any universe other than {_default_universe}."
                )
            api_endpoint = QuantumEngineServiceClient.DEFAULT_MTLS_ENDPOINT
        else:
            api_endpoint = QuantumEngineServiceClient._DEFAULT_ENDPOINT_TEMPLATE.format(
                UNIVERSE_DOMAIN=universe_domain
            )
        return api_endpoint

    @staticmethod
    def _get_universe_domain(
        client_universe_domain: Optional[str], universe_domain_env: Optional[str]
    ) -> str:
        """Return the universe domain used by the client.

        Args:
            client_universe_domain (Optional[str]): The universe domain configured via the client
                options.
            universe_domain_env (Optional[str]): The universe domain configured via the
                "GOOGLE_CLOUD_UNIVERSE_DOMAIN" environment variable.

        Returns:
            str: The universe domain to be used by the client.

        Raises:
            ValueError: If the universe domain is an empty string.
        """
        universe_domain = QuantumEngineServiceClient._DEFAULT_UNIVERSE
        if client_universe_domain is not None:
            universe_domain = client_universe_domain
        elif universe_domain_env is not None:
            universe_domain = universe_domain_env
        if len(universe_domain.strip()) == 0:
            raise ValueError("Universe Domain cannot be an empty string.")
        return universe_domain

    def _validate_universe_domain(self):
        """Validates client's and credentials' universe domains are consistent.

        Returns:
            bool: True iff the configured universe domain is valid.

        Raises:
            ValueError: If the configured universe domain is not valid.
        """

        # NOTE (b/349488459): universe validation is disabled until further notice.
        return True

    def _add_cred_info_for_auth_errors(self, error: core_exceptions.GoogleAPICallError) -> None:
        """Adds credential info string to error details for 401/403/404 errors.

        Args:
            error (google.api_core.exceptions.GoogleAPICallError): The error to add the cred info.
        """
        if error.code not in [HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND]:
            return

        cred = self._transport._credentials

        # get_cred_info is only available in google-auth>=2.35.0
        if not hasattr(cred, "get_cred_info"):
            return

        # ignore the type check since pypy test fails when get_cred_info
        # is not available
        cred_info = cred.get_cred_info()  # type: ignore
        if cred_info and hasattr(error._details, "append"):
            error._details.append(json.dumps(cred_info))

    @property
    def api_endpoint(self):
        """Return the API endpoint used by the client instance.

        Returns:
            str: The API endpoint used by the client instance.
        """
        return self._api_endpoint

    @property
    def universe_domain(self) -> str:
        """Return the universe domain used by the client instance.

        Returns:
            str: The universe domain used by the client instance.
        """
        return self._universe_domain

    def __init__(
        self,
        *,
        credentials: Optional[ga_credentials.Credentials] = None,
        transport: Optional[
            str | QuantumEngineServiceTransport | Callable[..., QuantumEngineServiceTransport]
        ] = None,
        client_options: Optional[client_options_lib.ClientOptions | dict] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
    ) -> None:
        """Instantiates the quantum engine service client.

        Args:
            credentials (Optional[google.auth.credentials.Credentials]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
            transport (Optional[Union[str,QuantumEngineServiceTransport,Callable[..., QuantumEngineServiceTransport]]]):
                The transport to use, or a Callable that constructs and returns a new transport.
                If a Callable is given, it will be called with the same set of initialization
                arguments as used in the QuantumEngineServiceTransport constructor.
                If set to None, a transport is chosen automatically.
            client_options (Optional[Union[google.api_core.client_options.ClientOptions, dict]]):
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
                default "googleapis.com" universe. Note that the ``api_endpoint``
                property still takes precedence; and ``universe_domain`` is
                currently not supported for mTLS.

            client_info (google.api_core.gapic_v1.client_info.ClientInfo):
                The client info used to send a user-agent string along with
                API requests. If ``None``, then default info will be used.
                Generally, you only need to set this if you're developing
                your own client library.

        Raises:
            google.auth.exceptions.MutualTLSChannelError: If mutual TLS transport
                creation failed for any reason.
        """  # noqa E501
        self._client_options = client_options
        if isinstance(self._client_options, dict):
            self._client_options = client_options_lib.from_dict(self._client_options)
        if self._client_options is None:
            self._client_options = client_options_lib.ClientOptions()
        universe_domain_opt = getattr(self._client_options, 'universe_domain', None)

        self._use_client_cert, self._use_mtls_endpoint, self._universe_domain_env = (
            QuantumEngineServiceClient._read_environment_variables()
        )
        self._client_cert_source = QuantumEngineServiceClient._get_client_cert_source(
            self._client_options.client_cert_source, self._use_client_cert
        )
        self._universe_domain = QuantumEngineServiceClient._get_universe_domain(
            universe_domain_opt, self._universe_domain_env
        )
        self._api_endpoint = None  # updated below, depending on `transport`

        # Initialize the universe domain validation.
        self._is_universe_domain_valid = False

        if CLIENT_LOGGING_SUPPORTED:  # pragma: NO COVER
            # Setup logging.
            client_logging.initialize_logging()

        api_key_value = getattr(self._client_options, "api_key", None)
        if api_key_value and credentials:
            raise ValueError("client_options.api_key and credentials are mutually exclusive")

        # Save or instantiate the transport.
        # Ordinarily, we provide the transport, but allowing a custom transport
        # instance provides an extensibility point for unusual situations.
        transport_provided = isinstance(transport, QuantumEngineServiceTransport)
        if transport_provided:
            # transport is a QuantumEngineServiceTransport instance.
            if credentials or self._client_options.credentials_file or api_key_value:
                raise ValueError(
                    "When providing a transport instance, provide its credentials directly."
                )
            if self._client_options.scopes:
                raise ValueError(
                    "When providing a transport instance, provide its scopes directly."
                )
            self._transport = cast(QuantumEngineServiceTransport, transport)
            self._api_endpoint = self._transport.host

        self._api_endpoint = self._api_endpoint or QuantumEngineServiceClient._get_api_endpoint(
            self._client_options.api_endpoint,
            self._client_cert_source,
            self._universe_domain,
            self._use_mtls_endpoint,
        )

        if not transport_provided:
            import google.auth._default

            if api_key_value and hasattr(google.auth._default, "get_api_key_credentials"):
                credentials = google.auth._default.get_api_key_credentials(api_key_value)

            transport_init: (
                type[QuantumEngineServiceTransport] | Callable[..., QuantumEngineServiceTransport]
            ) = (
                QuantumEngineServiceClient.get_transport_class(transport)
                if isinstance(transport, str) or transport is None
                else cast(Callable[..., QuantumEngineServiceTransport], transport)
            )
            # initialize with the provided callable or the passed in class
            self._transport = transport_init(
                credentials=credentials,
                credentials_file=self._client_options.credentials_file,
                host=self._api_endpoint,
                scopes=self._client_options.scopes,
                client_cert_source_for_mtls=self._client_cert_source,
                quota_project_id=self._client_options.quota_project_id,
                client_info=client_info,
                always_use_jwt_access=True,
                api_audience=self._client_options.api_audience,
            )

        if "async" not in str(self._transport):
            if CLIENT_LOGGING_SUPPORTED and _LOGGER.isEnabledFor(
                std_logging.DEBUG
            ):  # pragma: NO COVER
                _LOGGER.debug(
                    (
                        "Created client `cirq_google.cloud.quantum_v1alpha1."
                        "QuantumEngineServiceClient`."
                    ),
                    extra=(
                        {
                            "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                            "universeDomain": getattr(
                                self._transport._credentials, "universe_domain", ""
                            ),
                            "credentialsType": f"{type(self._transport._credentials).__module__}.{type(self._transport._credentials).__qualname__}",  # noqa E501
                            "credentialsInfo": getattr(
                                self.transport._credentials, "get_cred_info", lambda: None
                            )(),
                        }
                        if hasattr(self._transport, "_credentials")
                        else {
                            "serviceName": "google.cloud.quantum.v1alpha1.QuantumEngineService",
                            "credentialsType": None,
                        }
                    ),
                )

    def create_quantum_program(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.CreateQuantumProgramRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def get_quantum_program(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumProgramRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_program]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_programs(
        self,
        request: Optional[engine.ListQuantumProgramsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumProgramsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProgramsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumProgramsPager:
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
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_programs]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumProgramsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def delete_quantum_program(
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

            def sample_delete_quantum_program():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumProgramRequest(
                )

                # Make the request
                client.delete_quantum_program(request=request)

        Args:
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.DeleteQuantumProgramRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.delete_quantum_program]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def update_quantum_program(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.UpdateQuantumProgramRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.update_quantum_program]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def create_quantum_job(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.CreateQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
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

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def get_quantum_job(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
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

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_jobs(
        self,
        request: Optional[engine.ListQuantumJobsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumJobsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumJobsPager:
                -

                Iterating over this object will yield
                results and resolve additional pages
                automatically.

        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
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

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumJobsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def delete_quantum_job(
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

            def sample_delete_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumJobRequest(
                )

                # Make the request
                client.delete_quantum_job(request=request)

        Args:
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.DeleteQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
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

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def update_quantum_job(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.UpdateQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumJob:
                -
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
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

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def cancel_quantum_job(
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

            def sample_cancel_quantum_job():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.CancelQuantumJobRequest(
                )

                # Make the request
                client.cancel_quantum_job(request=request)

        Args:
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.CancelQuantumJobRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
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

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def list_quantum_job_events(
        self,
        request: Optional[engine.ListQuantumJobEventsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumJobEventsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumJobEventsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumJobEventsPager:
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
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_job_events]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumJobEventsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_quantum_result(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumResultRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.types.QuantumResult:
                -
        """
        # Create or coerce a protobuf request object.
        # - Use the request object if provided (there's no risk of modifying the input as
        #   there are no flattened fields), or create one.
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

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_processors(
        self,
        request: Optional[engine.ListQuantumProcessorsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumProcessorsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumProcessorsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumProcessorsPager:
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
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_processors]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumProcessorsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_quantum_processor(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumProcessorRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_processor]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def get_quantum_processor_config(
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

            def sample_get_quantum_processor_config():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.GetQuantumProcessorConfigRequest(
                    name="name_value",
                )

                # Make the request
                response = client.get_quantum_processor_config(request=request)

                # Handle the response
                print(response)

        Args:
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumProcessorConfigRequest, dict]):
                The request object. -
            name (str):
                Required. -
                This corresponds to the ``name`` field
                on the ``request`` instance; if ``request`` is provided, this
                should not be set.
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
                'If the `request` argument is set, then none of '
                'the individual field arguments should be set.'
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
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_processor_config]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_calibrations(
        self,
        request: Optional[engine.ListQuantumCalibrationsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumCalibrationsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumCalibrationsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumCalibrationsPager:
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
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_calibrations]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumCalibrationsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def get_quantum_calibration(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumCalibrationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_calibration]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def create_quantum_reservation(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.CreateQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.create_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def cancel_quantum_reservation(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.CancelQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.cancel_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def delete_quantum_reservation(
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

            def sample_delete_quantum_reservation():
                # Create a client
                client = quantum_v1alpha1.QuantumEngineServiceClient()

                # Initialize request argument(s)
                request = quantum_v1alpha1.DeleteQuantumReservationRequest(
                )

                # Make the request
                client.delete_quantum_reservation(request=request)

        Args:
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.DeleteQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.delete_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        rpc(request, retry=retry, timeout=timeout, metadata=metadata)

    def get_quantum_reservation(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.GetQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.get_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_reservations(
        self,
        request: Optional[engine.ListQuantumReservationsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumReservationsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationsPager:
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
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_reservations]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumReservationsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def update_quantum_reservation(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.UpdateQuantumReservationRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.update_quantum_reservation]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def quantum_run_stream(
        self,
        requests: Optional[Iterator[engine.QuantumRunStreamRequest]] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> Iterable[engine.QuantumRunStreamResponse]:
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
            requests (Iterator[cirq_google.cloud.quantum_v1alpha1.types.QuantumRunStreamRequest]):
                The request object iterator. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            Iterable[cirq_google.cloud.quantum_v1alpha1.types.QuantumRunStreamResponse]:
                -
        """

        # Wrap the RPC method; this adds retry and timeout information,
        # and friendly error handling.
        rpc = self._transport._wrapped_methods[self._transport.quantum_run_stream]

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(requests, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_reservation_grants(
        self,
        request: Optional[engine.ListQuantumReservationGrantsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumReservationGrantsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationGrantsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationGrantsPager:
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
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_reservation_grants]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumReservationGrantsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def reallocate_quantum_reservation_grant(
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ReallocateQuantumReservationGrantRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
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
        rpc = self._transport._wrapped_methods[self._transport.reallocate_quantum_reservation_grant]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("name", request.name),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # Done; return the response.
        return response

    def list_quantum_reservation_budgets(
        self,
        request: Optional[engine.ListQuantumReservationBudgetsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumReservationBudgetsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumReservationBudgetsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumReservationBudgetsPager:
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
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_reservation_budgets]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumReservationBudgetsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def list_quantum_time_slots(
        self,
        request: Optional[engine.ListQuantumTimeSlotsRequest | dict] = None,
        *,
        retry: OptionalRetry = gapic_v1.method.DEFAULT,
        timeout: float | object = gapic_v1.method.DEFAULT,
        metadata: Sequence[tuple[str, str | bytes]] = (),
    ) -> pagers.ListQuantumTimeSlotsPager:
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
            request (Union[cirq_google.cloud.quantum_v1alpha1.types.ListQuantumTimeSlotsRequest, dict]):
                The request object. -
            retry (google.api_core.retry.Retry): Designation of what errors, if any,
                should be retried.
            timeout (float): The timeout for this request.
            metadata (Sequence[Tuple[str, Union[str, bytes]]]): Key/value pairs which should be
                sent along with the request as metadata. Normally, each value must be of type `str`,
                but for metadata keys ending with the suffix `-bin`, the corresponding values must
                be of type `bytes`.

        Returns:
            cirq_google.cloud.quantum_v1alpha1.services.quantum_engine_service.pagers.ListQuantumTimeSlotsPager:
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
        rpc = self._transport._wrapped_methods[self._transport.list_quantum_time_slots]

        # Certain fields should be provided within the metadata header;
        # add these here.
        metadata = tuple(metadata) + (
            gapic_v1.routing_header.to_grpc_metadata((("parent", request.parent),)),
        )

        # Validate the universe domain.
        self._validate_universe_domain()

        # Send the request.
        response = rpc(request, retry=retry, timeout=timeout, metadata=metadata)

        # This method is paged; wrap the response in a pager, which provides
        # an `__iter__` convenience method.
        response = pagers.ListQuantumTimeSlotsPager(
            method=rpc,
            request=request,
            response=response,
            retry=retry,
            timeout=timeout,
            metadata=metadata,
        )

        # Done; return the response.
        return response

    def __enter__(self) -> "QuantumEngineServiceClient":
        return self

    def __exit__(self, typ, value, traceback):
        """Releases underlying transport's resources.

        .. warning::
            ONLY use as a context manager if the transport is NOT shared
            with other clients! Exiting the with block will CLOSE the transport
            and may cause errors in other clients!
        """
        self.transport.close()


DEFAULT_CLIENT_INFO = gapic_v1.client_info.ClientInfo(gapic_version=cirq_google.__version__)

if hasattr(DEFAULT_CLIENT_INFO, "protobuf_runtime_version"):  # pragma: NO COVER
    DEFAULT_CLIENT_INFO.protobuf_runtime_version = google.protobuf.__version__

__all__ = ("QuantumEngineServiceClient",)
