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
"""Client for making requests to IonQ's API."""

from __future__ import annotations

import datetime
import json.decoder as jd
import platform
import sys
import time
import urllib
import warnings
from collections.abc import Callable
from typing import Any, cast

import requests

import cirq_ionq
from cirq import __version__ as cirq_version
from cirq_ionq import ionq_exceptions

# https://support.cloudflare.com/hc/en-us/articles/115003014512-4xx-Client-Error
# "Cloudflare will generate and serve a 409 response for a Error 1001: DNS Resolution Error."
# We may want to condition on the body as well, to allow for some GET requests to return 409 in
# the future.
RETRIABLE_FOR_GETS = {requests.codes.conflict}
# Retriable regardless of the source
# Handle 52x responses from cloudflare.
# See https://support.cloudflare.com/hc/en-us/articles/115003011431/
RETRIABLE_STATUS_CODES = {
    requests.codes.too_many_requests,
    requests.codes.internal_server_error,
    requests.codes.bad_gateway,
    requests.codes.service_unavailable,
    *list(range(520, 530)),
}


def _is_retriable(code, method):
    return code in RETRIABLE_STATUS_CODES or (method == "GET" and code in RETRIABLE_FOR_GETS)


class _IonQClient:
    """Handles calls to IonQ's API.

    Users should not instantiate this themselves, but instead should use `cirq_ionq.Service`.
    """

    SUPPORTED_TARGETS = {'qpu', 'simulator'}
    SUPPORTED_VERSIONS = {'v0.4'}

    def __init__(
        self,
        remote_host: str,
        api_key: str,
        default_target: str | None = None,
        api_version: str = 'v0.4',
        max_retry_seconds: int = 3600,  # 1 hour
        verbose: bool = False,
    ):
        """Creates the IonQClient.

        Users should use `cirq_ionq.Service` instead of this class directly.

        The IonQClient handles making requests to the IonQClient, returning dictionary results.
        It handles retry and authentication.

        Args:
            remote_host: The url of the server exposing the IonQ API. This will strip anything
                besides the base scheme and netloc, i.e. it only takes the part of the host of
                the form `http://example.com` of `http://example.com/test`.
            api_key: The key used for authenticating against the IonQ API.
            default_target: The default target to run against. Supports one of 'qpu' and
                'simulator'. Can be overridden by calls with target in their signature.
            api_version: Which version of the api to use. As of June, 2025, accepts 'v0.4' only,
                which is the default.
            max_retry_seconds: The time to continue retriable responses. Defaults to 3600.
            verbose: Whether to print to stderr and stdio any retriable errors that are encountered.
        """
        url = urllib.parse.urlparse(remote_host)
        assert url.scheme and url.netloc, (
            f'Specified remote_host {remote_host} is not a valid url, for example '
            'http://example.com'
        )
        assert (
            api_version in self.SUPPORTED_VERSIONS
        ), f'Only api v0.4 is accepted but was {api_version}'
        assert (
            default_target is None or default_target in self.SUPPORTED_TARGETS
        ), f'Target can only be one of {self.SUPPORTED_TARGETS} but was {default_target}.'
        assert max_retry_seconds >= 0, 'Negative retry not possible without time machine.'

        self.url = f'{url.scheme}://{url.netloc}/{api_version}'
        self.headers = self.api_headers(api_key)
        self.default_target = default_target
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose
        self.batch_mode = False

    def create_job(
        self,
        serialized_program: cirq_ionq.SerializedProgram,
        repetitions: int | None = None,
        target: str | None = None,
        name: str | None = None,
        extra_query_params: dict | None = None,
        batch_mode: bool = False,
    ) -> dict:
        """Create a job.

        Args:
            serialized_program: The `cirq_ionq.SerializedProgram` containing the serialized
                information about the circuit to run.
            repetitions: The number of times to repeat the circuit. For simulation the repeated
                sampling is not done on the server, but is passed as metadata to be recovered
                from the returned job.
            target: If supplied the target to run on. Supports one of `qpu` or `simulator`. If not
                set, uses `default_target`.
            name: An optional name of the job. Different than the `job_id` of the job.
            extra_query_params: Specify any parameters to include in the request.
            batch_mode: bool determines whether to submit a single circuit or a batch of circuits.

        Returns:
            The json body of the response as a dict. This does not contain populated information
            about the job, but does contain the job id.

        Raises:
            An IonQException if the request fails.
        """
        actual_target = self._target(target)

        json: dict[str, Any] = {
            'backend': actual_target,
            "type": "ionq.multi-circuit.v1" if batch_mode else "ionq.circuit.v1",
            'lang': 'json',
            'input': serialized_program.input,
        }
        if name:
            json['name'] = name
        # We have to pass measurement keys through the metadata.
        json['metadata'] = serialized_program.metadata
        if serialized_program.settings:
            json['settings'] = serialized_program.settings

        # Shots are ignored by simulator, but pass them anyway.
        json['shots'] = str(repetitions)
        # API does not return number of shots so pass this through as metadata.
        json['metadata']['shots'] = str(repetitions)

        if serialized_program.error_mitigation:
            if not 'settings' in json.keys():
                json['settings'] = {}
            json['settings']['error_mitigation'] = serialized_program.error_mitigation

        if serialized_program.compilation:
            if not 'settings' in json.keys():
                json['settings'] = {}
            json['settings']['compilation'] = serialized_program.compilation

        if serialized_program.noise:
            json['noise'] = serialized_program.noise

        if serialized_program.dry_run:
            json['dry_run'] = serialized_program.dry_run
            if json['backend'] == 'simulator':
                warnings.warn(
                    'Please note that the `dry_run` option has no effect on the simulator target.'
                )

        if extra_query_params:
            json.update(extra_query_params)

        def request():
            return requests.post(f'{self.url}/jobs', json=json, headers=self.headers)

        request_response = self._make_request(request, json).json()
        self.batch_mode = batch_mode

        return request_response

    def get_job(self, job_id: str) -> dict:
        """Get the job from the IonQ API.

        Args:
            job_id: The UUID of the job (returned when the job was created).

        Returns:
            The json body of the response as a dict.

        Raises:
            IonQNotFoundException: If a job with the given job_id does not exist.
            IonQException: For other API call failures.
        """

        def request():
            return requests.get(f'{self.url}/jobs/{job_id}', headers=self.headers)

        return self._make_request(request, {}).json()

    def get_results(
        self, job_id: str, sharpen: bool | None = None, extra_query_params: dict | None = None
    ):
        """Get job results from IonQ API.

        Args:
            job_id: The UUID of the job (returned when the job was created).
            sharpen: A boolean that determines how to aggregate error mitigated.
                If True, apply majority vote mitigation; if False, apply average mitigation.
            extra_query_params: Specify any parameters to include in the request.

        Returns:
            extra_query_paramsresponse as a dict.

        Raises:
            IonQNotFoundException: If job or results don't exist.
            IonQException: For other API call failures.
        """

        params = {}

        if sharpen is not None:
            params["sharpen"] = sharpen

        if extra_query_params:
            params.update(extra_query_params)

        def request():
            if self.batch_mode:
                return requests.get(
                    f'{self.url}/jobs/{job_id}/results/probabilities/aggregated',
                    params=params,
                    headers=self.headers,
                )
            elif not self.batch_mode:
                return requests.get(
                    f'{self.url}/jobs/{job_id}/results/probabilities',
                    params=params,
                    headers=self.headers,
                )

        return self._make_request(request, {}).json()

    def list_jobs(
        self, status: str | None = None, limit: int = 100, batch_size: int = 1000
    ) -> list[dict[str, Any]]:
        """Lists jobs from the IonQ API.

        Args:
            status: If not None, filter to jobs with this status.
            limit: The maximum number of jobs to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A list of the json bodies of the job dicts.

        Raises:
            IonQException: If the API call fails.
        """
        params = {}
        if status:
            params['status'] = status
        return self._list('jobs', params, 'jobs', limit, batch_size)

    def cancel_job(self, job_id: str) -> dict:
        """Cancel a job on the IonQ API.

        Args:
            job_id: The UUID of the job (returned when the job was created).

        Note that the IonQ API v0.4 can cancel a completed job, which updates its status to
        canceled.

        Returns:
            The json body of the response as a dict.
        """

        def request():
            return requests.put(f'{self.url}/jobs/{job_id}/status/cancel', headers=self.headers)

        return self._make_request(request, {}).json()

    def delete_job(self, job_id: str) -> dict:
        """Permanently delete the job on the IonQ API.

        Args:
            job_id: The UUID of the job (returned when the job was created).

        Returns:
            The json body of the response as a dict.
        """

        def request():
            return requests.delete(f'{self.url}/jobs/{job_id}', headers=self.headers)

        return self._make_request(request, {}).json()

    def get_current_calibration(self) -> dict:
        """Returns the current calibration as an `cirq_ionq.Calibration` object.

        Currently returns the current calibration for the only target `qpu`.
        """

        def request():
            return requests.get(f'{self.url}/calibrations/current', headers=self.headers)

        return self._make_request(request, {}).json()

    def list_calibrations(
        self,
        start: datetime.datetime | None = None,
        end: datetime.datetime | None = None,
        limit: int = 100,
        batch_size: int = 1000,
    ) -> list[dict]:
        """Lists calibrations from the IonQ API.

        Args:
            start: If supplied, only calibrations after this date and time. Accurate to seconds.
            end: If supplied, only calibrations before this date and time. Accurate to seconds.
            limit: The maximum number of calibrations to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A list of the json bodies of the calibration dicts.

        Raises:
            IonQException: If the API call fails.
        """

        params = {}
        epoch = datetime.datetime.fromtimestamp(0, datetime.UTC)
        if start:
            params['start'] = int((start - epoch).total_seconds() * 1000)
        if end:
            params['end'] = int((end - epoch).total_seconds() * 1000)
        return self._list('calibrations', params, 'calibrations', limit, batch_size)

    def api_headers(self, api_key: str):
        """API Headers needed to make calls to the REST API.

        Args:
            api_key: The key used for authenticating against the IonQ API.

        Returns:
            dict[str, str]: A dict of :class:`requests.Request` headers.
        """
        return {
            'Authorization': f'apiKey {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': self._user_agent(),
        }

    def _user_agent(self):
        """Generates the user agent string which is helpful in identifying
        different tools in the internet. Valid user-agent ionq_client header that
        indicates the request is from cirq_ionq along with the system, os,
        python,libraries details.

        Returns:
            str: A string of generated user agent.
        """
        cirq_version_string = f'cirq/{cirq_version}'
        python_version_string = f'python/{platform.python_version()}'
        return f'{cirq_version_string} ({python_version_string})'

    def _target(self, target: str | None) -> str:
        """Returns the target if not None or the default target.

        Raises:
            AssertionError: if both `target` and `default_target` are not set.
        """
        assert target is not None or self.default_target is not None, (
            'One must specify a target on this call, or a default_target on the service/client, '
            'but neither were set.'
        )
        return cast(str, target or self.default_target)

    def _make_request(
        self, request: Callable[[], requests.Response], json: dict
    ) -> requests.Response:
        """Make a request to the API, retrying if necessary.

        Args:
            request: A function that returns a `requests.Response`.
            json: POST body to be logged on failures.

        Returns:
            The request.Response from the final successful request call.

        Raises:
            IonQException: If there was a not-retriable error from the API.
            IonQNotFoundException: If the api returned not found.
            TimeoutError: If the requests retried for more than `max_retry_seconds`.

        """
        # Initial backoff of 100ms.
        delay_seconds = 0.1

        while True:
            try:
                response = request()
                if response.ok:
                    return response
                if response.status_code == requests.codes.unauthorized:
                    raise ionq_exceptions.IonQException(
                        '"Not authorized" returned by IonQ API. Check to ensure you have supplied '
                        'the correct API key.',
                        response.status_code,
                    )
                if response.status_code == requests.codes.not_found:
                    raise ionq_exceptions.IonQNotFoundException(
                        'IonQ could not find requested resource.'
                    )
                if not _is_retriable(response.status_code, response.request.method):
                    error = {}
                    try:
                        error = response.json()
                    except jd.JSONDecodeError:  # pragma: no cover
                        pass
                    raise ionq_exceptions.IonQException(
                        'Non-retry-able error making request to IonQ API. '
                        f'Request Body: {json} '
                        f'Response Body: {error} '
                        f'Status: {response.status_code} '
                        f'Error:{response.reason}',
                        response.status_code,
                    )
                message = response.reason
                # Fallthrough should retry.
            except requests.RequestException as e:
                # Connection error, timeout at server, or too many redirects.
                # Retry these.
                message = f'RequestException of type {type(e)}.'
            if delay_seconds > self.max_retry_seconds:
                raise TimeoutError(f'Reached maximum number of retries. Last error: {message}')
            if self.verbose:
                print(message, file=sys.stderr)
                print(f'Waiting {delay_seconds} seconds before retrying.')
            time.sleep(delay_seconds)
            delay_seconds *= 2

    def _list(
        self, resource_path: str, params: dict, response_key: str, limit: int, batch_size: int
    ) -> list[dict]:
        """Helper method for list calls.

        Args:
            resource_path: The resource path for the object being listed. Follows the base url
                and version. No leading slash.
            params: The params to pass with the list call.
            response_key: The key to get the list of objects that have been listed.
            limit: The maximum number of objects to return.
            batch_size: The size of the batches requested per http GET call.

        Returns:
            A sequence of dictionaries corresponding to the objects listed.
        """
        json = {'limit': batch_size}
        token: str | None = None
        results: list[dict[str, Any]] = []
        while len(results) < limit:
            full_params = params.copy()
            if token:
                full_params['next'] = token

            def request():
                return requests.get(
                    f'{self.url}/{resource_path}',
                    headers=self.headers,
                    json=json,
                    params=full_params,
                )

            response = self._make_request(request, json).json()
            results.extend(response[response_key])
            if 'next' not in response:
                break
            token = response['next']
        return results[:limit]
