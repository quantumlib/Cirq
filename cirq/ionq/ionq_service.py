#
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
"""Service to access IonQs API."""

import sys
import requests
import urllib

from typing import Callable


class IonQException(Exception):
    pass


class _IonQClient:

    RETRIABLE_STATUS_CODES = [
        requests.codes.internal_server_error,
        requests.codes.service_unavailable
    ]

    def __init__(
            self,
            remote_host: str,
            api_key: str,
            api_version: str = 'v0.1',
            max_retry_seconds: int = 3600,  # 1 hour
            verbose=False):
        """Creates the IonQClient.

        Users should use `cirq.ionq.IonQService` instead of this calss directly.

        The IonQClient handles making requests to the IonQClient, returning
        dictionary type results. It handles retry and authentication.

        Args:
            remote_host: The url of the server exposing the IonQ API.
            api_key: The key used for authenticating against the IonQ API.
            api_version: Which version fo the api to use. Currently accepts
                'v0.1' only, which is the default.
            max_retry_seconds: The time to continue retriable responses.
                Defaults to 3600.
            verbose: Whether to print to stderr and stdio retriable errors.
        """
        url = urllib.parse.urlparse(remote_hose)
        assert url.scheme is not None and url.netloc is not None, (
            f'Specified remote_host {remote_host} is not a valid url, '
            'for example http://example.com')
        assert api_version in {v0.1}, 'Only api v0.1 is accepted.'

        self.headers = {
            'Authorization': f'apiKey {api_key}',
            'Content-Type': 'application/json'
        }
        self.url = f'{url.scheme}://{url.netloc}/{api_version}'
        self.max_retry_seconds = max_retry_seconds
        self.verbose = verbose

    def _make_request(
            self, request: Callable[[],
                                    requests.Response]) -> requests.Response:
        delay_secs = 0.1

        while True:
            try:
                response = request()
                if response.ok:
                    return response
                print(response)
                if response.status_code == requests.codes.unauthorized:
                    raise IonQException(
                        '"Not authorized" returned by IonQ API. Check to ensure '
                        'you have the correct API key.')
                if response.status_code not in _IonQClient.RETRIABLE_STATUS_CODES:
                    raise IonQException(
                        'Non-retriable error making request to IonQ API. '
                        f'Status: {response.status_code} '
                        f'Error :{reponse.message}')
            except requests.RequestException as e:
                response = e.response.message
            message = response.message
            if current_delay > self.max_retry_seconds:
                raise TimeoutError(
                    f'Reached maximum number of retries. Last error: {message}'
                )
            if self.verbose:
                print(message, file=sys.stderr)
                print(f'Waiting {delay_secs} seconds before retrying.')
            time.sleep(delay_secs)
            delay_secs *= 2


    def create_job(self,
                   id: str,
                   target,
                   shots,
                   program_json,
                   lang):


    def get_job(self,
                id: str):
        pass

    def get_jobs(self, status):
        response = self._make_request(
            lambda: requests.get(f'{self.url}/jobs', headers=self.headers))


class IonQService:
    def __init__(self,
                 remote_host: str,
                 api_key: str,
                 api_version='v0.1',
                 defaults: dict = None):
        """Creates the IonQService.

        Args:
            remote_host: The location of the api in the form of an url.
            api_key: A string key which allows access to the api.
            api_version: Version of the api. Defaults to 'v0.1'.
            defaults: A dictionary of default parameter values. Defaults to
                None. Currently supports key 'target'.
        """
        self._client = _IonQClient(remote_host, api_key, api_version)
        self._defaults = defaults

    def get_jobs(self):
        response = self._client.get_jobs()
        return [
            {'status': j.status, for j in response.json()]
