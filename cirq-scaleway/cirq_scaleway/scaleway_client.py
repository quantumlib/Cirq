# Copyright 2024 Scaleway
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
import httpx

from typing import Optional

_ENDPOINT_PLATFORM = "/platforms"
_ENDPOINT_SESSION = "/sessions"
_ENDPOINT_JOB = "/jobs"


class QaaSClient:
    def __init__(self, project_id: str, token: str, url: str) -> None:
        self.__token = token
        self.__url = url
        self.__project_id = project_id

    def _http_client(self) -> httpx.Client:
        return httpx.Client(
            headers=self._api_headers(), base_url=self.__url, timeout=10.0, verify=True
        )

    def _api_headers(self) -> dict:
        return {"X-Auth-Token": self.__token}

    def _build_endpoint(self, endpoint: str) -> str:
        return f"{self.__url}{endpoint}"

    def get_platform(self, platform_id: str) -> dict:
        http_client = self._http_client()
        endpoint = f"{self._build_endpoint(_ENDPOINT_PLATFORM)}/{platform_id}"

        resp = http_client.get(endpoint)
        resp.raise_for_status()

        return resp.json()

    def list_platforms(self, name: Optional[str] = None) -> dict:
        filter_by_name = ""
        if name:
            filter_by_name = f"?name={name}"

        http_client = self._http_client()
        endpoint = f"{self._build_endpoint(_ENDPOINT_PLATFORM)}{filter_by_name}"

        resp = http_client.get(endpoint)
        resp.raise_for_status()

        return resp.json()

    def create_session(
        self,
        name: str,
        platform_id: str,
        deduplication_id: str,
        max_duration: str,
        max_idle_duration: str,
    ) -> str:
        http_client = self._http_client()

        payload = {
            "name": name,
            "project_id": self.__project_id,
            "platform_id": platform_id,
            "deduplication_id": deduplication_id,
            "max_duration": max_duration,
            "max_idle_duration": max_idle_duration,
        }

        response = http_client.post(self._build_endpoint(_ENDPOINT_SESSION), json=payload)

        response.raise_for_status()
        response_dict = response.json()
        session_id = response_dict["id"]

        return session_id

    def get_session(self, session_id: str) -> dict:
        http_client = self._http_client()
        endpoint = f"{self._build_endpoint(_ENDPOINT_SESSION)}/{session_id}"

        resp = http_client.get(endpoint)
        resp.raise_for_status()

        return resp.json()

    def update_session(
        self, session_id: str, name: str, max_duration: str, max_idle_duration: str
    ) -> str:
        http_client = self._http_client()

        payload = {
            "name": name,
            "max_duration": max_duration,
            "max_idle_duration": max_idle_duration,
        }

        response = http_client.patch(
            self._build_endpoint(f"{_ENDPOINT_SESSION}/{session_id}"), json=payload
        )

        response.raise_for_status()
        response_dict = response.json()
        session_id = response_dict["id"]

        return session_id

    def terminate_session(self, session_id: str) -> str:
        http_client = self._http_client()

        response = http_client.post(
            self._build_endpoint(f"{_ENDPOINT_SESSION}/{session_id}/terminate")
        )

        response.raise_for_status()
        response_dict = response.json()
        session_id = response_dict["id"]

        return session_id

    def delete_session(self, session_id: str):
        http_client = self._http_client()

        http_client.delete(self._build_endpoint(f"{_ENDPOINT_SESSION}/{session_id}"))

    def create_job(self, name: str, session_id: str, circuits: dict) -> str:
        http_client = self._http_client()

        payload = {
            "name": name,
            "session_id": session_id,
            "circuit": {"qiskit_circuit": f"{circuits}"},
        }

        response = http_client.post(self._build_endpoint(_ENDPOINT_JOB), json=payload)

        response.raise_for_status()
        response_dict = response.json()

        return response_dict["id"]

    def get_job(self, job_id: str) -> dict:
        http_client = self._http_client()
        endpoint = f"{self._build_endpoint(_ENDPOINT_JOB)}/{job_id}"

        resp = http_client.get(endpoint)
        resp.raise_for_status()

        return resp.json()

    def get_job_results(self, job_id: str) -> list:
        http_client = self._http_client()
        endpoint = f"{self._build_endpoint(_ENDPOINT_JOB)}/{job_id}/results"

        resp = http_client.get(endpoint)
        resp.raise_for_status()

        results_dict = resp.json()

        return results_dict["job_results"]
