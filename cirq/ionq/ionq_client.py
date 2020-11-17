# Copyright 2020 The Cirq Developers
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

import sys
import time
import urllib
from typing import Callable, cast, Optional
import requests


class IonQException(Exception):
    """An exception for errors coming from IonQ's API.

    Attributes:
        status_code: A http status code, if coming from an http response
            with a failing status.
    """

    def __init__(self, message, status_code: int = None):
        super().__init__(f'Status code: {status_code}, Message: \'{message}\'')
        self.status_code = status_code


class IonQNotFoundException(IonQException):
    """An exception for errors from IonQ's API when a resource is not found."""

    def __init__(self, message):
        super().__init__(message, status_code=requests.codes.not_found)


class _IonQClient:
    """Handles calls to IonQ's API.

    Users should not instantiate this themselves, but instead should use
    `cirq.ionq.Service`.
    """

    RETRIABLE_STATUS_CODES = {
        requests.codes.internal_server_error, requests.codes.service_unavailable
    }
    SUPPORTED_TARGETS = {'qpu', 'simulator'}
    SUPPORTED_VERSIONS = {
        'v0.1',
    }

    def __init__(
            self,
            remote_host: str,
            api_key: str,
            default_target: Optional[str] = None,
            api_version: str = 'v0.1',
            max_retry_seconds: int = 3600,  # 1 hour
            verbose: bool = False):
        """Creates the IonQClient.

        Users should use `cirq.ionq.Service` instead of this class directly.

        The IonQClient handles making requests to the IonQClient, returning
        dictionary results. It handles retry and authentication.

        Args:
            remote_host: The url of the server exposing the IonQ API.
                This will strip anything besides the base scheme and netloc,
                i.e. it only takes the part of the host of the form
                `http://example.com` of `http://example.com/test`.
            api_key: The key used for authenticating against the IonQ API.
            default_target: The default target to run against. Supports
                one of 'qpu' and 'simulator'. Can be overridden by calls with
                target in their signature.
            api_version: Which version fo the api to use. Currently accepts
                'v0.1' only, which is the default.
            max_retry_seconds: The time to continue retriable responses.
                Defaults to 3600.
            verbose: Whether to print to stderr and stdio any retriable errors
                that are encountered.
        """
        url = urllib.parse.urlparse(remote_host)
        assert url.scheme and url.netloc, (
            f'Specified remote_host {remote_host} is not a valid url, '
            'for example http://example.com')
        assert api_version in self.SUPPORTED_VERSIONS, (
            f'Only api v0.1 is accepted but was {api_version}')
        assert (
            default_target is None or
            default_target in self.SUPPORTED_TARGETS), (
                f'Target can only be one of {self.SUPPORTED_TARGETS} but was '
                f'{default_target}.')
        assert max_retry_seconds >= 0, (
            'Negative retry not possible without time machine.')

        self.url = f'{url.scheme}://{url.netloc}/{api_version}'
        self.headers = {
            'Authorization': f'apiKey {api_key}',
            'Content-Type': 'application/json'
        }
        self.default_target = default_target
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose

    def _target(self, target: Optional[str]) -> str:
        """Returns the target if not None or the default target.

        Raises:
            AssertionError: if both `target` and `default_target` are not set.
        """
        assert target is not None or self.default_target is not None, (
            'One must specify a target on this call, or a default_target on '
            'the service/client, but neither were set.')
        return cast(str, target or self.default_target)

    def _make_request(self, request: Callable[[], requests.Response]
                     ) -> requests.Response:
        """"Make a request to the API, retrying if necessary.

        Args:
            request: A function that returns a `requests.Response`.

        Raises:
            IonQException: If there was a not-retriable error from the API.
            TimeoutError: If the requests retried for more than
                `max_retry_seconds`.

        Returns:
            The request.Response from the final successful request call.
        """
        # Initial backoff of 100ms.
        delay_seconds = 0.1

        while True:
            try:
                response = request()
                if response.ok:
                    return response
                if response.status_code == requests.codes.unauthorized:
                    raise IonQException(
                        '"Not authorized" returned by IonQ API. Check to '
                        'ensure you have supplied the correct API key.',
                        response.status_code)
                if response.status_code == requests.codes.not_found:
                    raise IonQNotFoundException(
                        'IonQ could not find requested resource.')
                if (response.status_code not in self.RETRIABLE_STATUS_CODES):
                    raise IonQException(
                        'Non-retry-able error making request to IonQ API. '
                        f'Status: {response.status_code} '
                        f'Error :{response.reason}', response.status_code)
                message = response.reason
                # Fallthrough should retry.
            except requests.RequestException as e:
                # Connection error, timeout at server, or too many redirects.
                # Retry these.
                message = f'RequestException of type {type(e)}.'
            if delay_seconds > self.max_retry_seconds:
                raise TimeoutError(
                    f'Reached maximum number of retries. Last error: {message}')
            if self.verbose:
                print(message, file=sys.stderr)
                print(f'Waiting {delay_seconds} seconds before retrying.')
            time.sleep(delay_seconds)
            delay_seconds *= 2

    def create_job(
            self,
            circuit_dict: dict,
            repetitions: Optional[int] = None,
            target: Optional[str] = None,
            name: Optional[str] = None,
    ) -> dict:
        """Create a job.

        Args:
            circuit_dict: A dict corresponding to the json encoding of the
                circuit for the IonQ API.
            repetitions: The number of times to repeat the circuit. Only can
                be set if the target is `qpu`. If not specified and target is
                `qpu`
            target: If supplied the target to run on. Supports one of `qpu` or
                `simulator`. If not set, uses `default_target`.
            name: An optional name of the job. Different than the `job_id` of
                the job.

        Returns:
            The json body of the response as a dict. This does not contain
            populated information about the job, but does contain the job id.

        Raises:
            An IonQ exception if the request fails.
        """
        actual_target = self._target(target)
        assert actual_target != 'qpu' or repetitions is not None, (
            'If the target is qpu, repetitions must be specified.')
        assert actual_target != 'simulator' or repetitions is None, (
            'If the target is simulator, repetitions should not be specified '
            'as the simulator is a full wavefunction simulator.')
        json = {
            'target': actual_target,
            'body': circuit_dict,
            'lang': 'json',
        }
        if name:
            json['name'] = name
        if repetitions:
            # API does not return number of shots, only histogram of
            # percentages, so we set it as metadata.
            json['metadata'] = {'shots': str(repetitions)}

        def request():
            return requests.post(f'{self.url}/jobs',
                                 json=json,
                                 headers=self.headers)

        return self._make_request(request).json()

    def get_job(self, job_id: str):
        """Get the job from the IonQ API.

        Args:
            job_id: The UUID of the job (returned when the job is created).

        Returns:
            A `cirq.ionq.IonQJob` corresponding to the job.

        Raises:
            IonQNotFoundException: If a job with the given job_id does not
                exist.
            IonQException: For other API call failures.
        """

        def request():
            return requests.get(f'{self.url}/jobs/{job_id}',
                                headers=self.headers)

        return self._make_request(request).json()
