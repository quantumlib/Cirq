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
import re
from typing import Any, Optional

from google.api_core import gapic_v1, path_template
from google.protobuf import json_format

from cirq_google.cloud.quantum_v1alpha1.types import engine

from .base import DEFAULT_CLIENT_INFO, QuantumEngineServiceTransport


class _BaseQuantumEngineServiceRestTransport(QuantumEngineServiceTransport):
    """Base REST backend transport for QuantumEngineService.

    Note: This class is not meant to be used directly. Use its sync and
    async sub-classes instead.

    This class defines the same methods as the primary client, so the
    primary client can load the underlying transport implementation
    and call it.

    It sends JSON representations of protocol buffers over HTTP/1.1
    """

    def __init__(
        self,
        *,
        host: str = 'quantum.googleapis.com',
        credentials: Optional[Any] = None,
        client_info: gapic_v1.client_info.ClientInfo = DEFAULT_CLIENT_INFO,
        always_use_jwt_access: Optional[bool] = False,
        url_scheme: str = 'https',
        api_audience: Optional[str] = None,
    ) -> None:
        """Instantiate the transport.
        Args:
            host (Optional[str]):
                 The hostname to connect to (default: 'quantum.googleapis.com').
            credentials (Optional[Any]): The
                authorization credentials to attach to requests. These
                credentials identify the application to the service; if none
                are specified, the client will attempt to ascertain the
                credentials from the environment.
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
        maybe_url_match = re.match("^(?P<scheme>http(?:s)?://)?(?P<host>.*)$", host)
        if maybe_url_match is None:
            raise ValueError(f"Unexpected hostname structure: {host}")  # pragma: NO COVER

        url_match_items = maybe_url_match.groupdict()

        host = f"{url_scheme}://{host}" if not url_match_items["scheme"] else host

        super().__init__(
            host=host,
            credentials=credentials,
            client_info=client_info,
            always_use_jwt_access=always_use_jwt_access,
            api_audience=api_audience,
        )

    class _BaseCancelQuantumJob:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'post',
                    'uri': '/v1alpha1/{name=projects/*/programs/*/jobs/*}:cancel',
                    'body': '*',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.CancelQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseCancelQuantumReservation:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'post',
                    'uri': '/v1alpha1/{name=projects/*/processors/*/reservations/*}:cancel',
                    'body': '*',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.CancelQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseCreateQuantumJob:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'post',
                    'uri': '/v1alpha1/{parent=projects/*/programs/*}/jobs',
                    'body': 'quantum_job',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.CreateQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseCreateQuantumProgram:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'post',
                    'uri': '/v1alpha1/{parent=projects/*}/programs',
                    'body': 'quantum_program',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.CreateQuantumProgramRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseCreateQuantumReservation:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'post',
                    'uri': '/v1alpha1/{parent=projects/*/processors/*}/reservations',
                    'body': 'quantum_reservation',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.CreateQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseDeleteQuantumJob:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'delete', 'uri': '/v1alpha1/{name=projects/*/programs/*/jobs/*}'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.DeleteQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseDeleteQuantumProgram:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'delete', 'uri': '/v1alpha1/{name=projects/*/programs/*}'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.DeleteQuantumProgramRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseDeleteQuantumReservation:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'delete',
                    'uri': '/v1alpha1/{name=projects/*/processors/*/reservations/*}',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.DeleteQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseGetQuantumCalibration:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{name=projects/*/processors/*/calibrations/*}'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.GetQuantumCalibrationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseGetQuantumJob:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{name=projects/*/programs/*/jobs/*}'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.GetQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseGetQuantumProcessor:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{name=projects/*/processors/*}'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.GetQuantumProcessorRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseGetQuantumProcessorConfig:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        __REQUIRED_FIELDS_DEFAULT_VALUES: dict[str, Any] = {}

        @classmethod
        def _get_unset_required_fields(cls, message_dict):
            return {
                k: v
                for k, v in cls.__REQUIRED_FIELDS_DEFAULT_VALUES.items()
                if k not in message_dict
            }

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'get',
                    'uri': '/v1alpha1/{name=projects/*/processors/*/configSnapshots/*/configs/*}',
                },
                {
                    'method': 'get',
                    'uri': '/v1alpha1/{name=projects/*/processors/*/configAutomationRuns/*/configs/*}',  # noqa E501
                },
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.GetQuantumProcessorConfigRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )
            query_params.update(
                _BaseQuantumEngineServiceRestTransport._BaseGetQuantumProcessorConfig._get_unset_required_fields(
                    query_params
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseGetQuantumProgram:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{name=projects/*/programs/*}'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.GetQuantumProgramRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseGetQuantumReservation:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{name=projects/*/processors/*/reservations/*}'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.GetQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseGetQuantumResult:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*/programs/*/jobs/*}/result'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.GetQuantumResultRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumCalibrations:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*/processors/*}/calibrations'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumCalibrationsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumJobEvents:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*/programs/*/jobs/*}/events'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumJobEventsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumJobs:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*/programs/*}/jobs'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumJobsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumProcessors:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*}/processors'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumProcessorsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumPrograms:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*}/programs'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumProgramsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumReservationBudgets:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*}/reservationBudgets'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumReservationBudgetsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumReservationGrants:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*}/reservationGrant'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumReservationGrantsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumReservations:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*/processors/*}/reservations'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumReservationsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseListQuantumTimeSlots:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {'method': 'get', 'uri': '/v1alpha1/{parent=projects/*/processors/*}/timeSlots'}
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ListQuantumTimeSlotsRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseQuantumRunStream:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

    class _BaseReallocateQuantumReservationGrant:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'post',
                    'uri': '/v1alpha1/{name=projects/*/reservationGrant/*}:reallocate',
                    'body': '*',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.ReallocateQuantumReservationGrantRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseUpdateQuantumJob:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'patch',
                    'uri': '/v1alpha1/{name=projects/*/programs/*/jobs/*}',
                    'body': 'quantum_job',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.UpdateQuantumJobRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseUpdateQuantumProgram:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'patch',
                    'uri': '/v1alpha1/{name=projects/*/programs/*}',
                    'body': 'quantum_program',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.UpdateQuantumProgramRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params

    class _BaseUpdateQuantumReservation:
        def __hash__(self):  # pragma: NO COVER
            return NotImplementedError("__hash__ must be implemented.")

        @staticmethod
        def _get_http_options() -> list[dict[str, str]]:
            http_options: list[dict[str, str]] = [
                {
                    'method': 'patch',
                    'uri': '/v1alpha1/{name=projects/*/processors/*/reservations/*}',
                    'body': 'quantum_reservation',
                }
            ]
            return http_options

        @staticmethod
        def _get_transcoded_request(http_options, request):
            pb_request = engine.UpdateQuantumReservationRequest.pb(request)
            transcoded_request = path_template.transcode(http_options, pb_request)
            return transcoded_request

        @staticmethod
        def _get_request_body_json(transcoded_request):
            # Jsonify the request body

            body = json_format.MessageToJson(
                transcoded_request['body'], use_integers_for_enums=True
            )
            return body

        @staticmethod
        def _get_query_params_json(transcoded_request):
            query_params = json.loads(
                json_format.MessageToJson(
                    transcoded_request['query_params'], use_integers_for_enums=True
                )
            )

            query_params["$alt"] = "json;enum-encoding=int"
            return query_params


__all__ = ('_BaseQuantumEngineServiceRestTransport',)
