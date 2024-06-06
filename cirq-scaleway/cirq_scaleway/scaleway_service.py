# Copyright 2024 The Cirq Developers
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
import os

from dotenv import dotenv_values

from .scaleway_device import ScalewayDevice
from .scaleway_client import QaaSClient


_ENDPOINT_URL = "https://api.scaleway.com/qaas/v1alpha1"


class ScalewayQuantumService:
    """
    :param project_id: optional UUID of the Scaleway Project, if the provided ``project_id`` is None, the value is loaded from the SCALEWAY_PROJECT_ID variables in the dotenv file or the CIRQ_SCALEWAY_PROJECT_ID environment variables

    :param secret_key: optional authentication token required to access the Scaleway API, if the provided ``secret_key`` is None, the value is loaded from the SCALEWAY_API_TOKEN variables in the dotenv file or the CIRQ_SCALEWAY_API_TOKEN environment variables

    :param url: optional value, endpoint URL of the API, if the provided ``url`` is None, the value is loaded from the SCALEWAY_API_URL variables in the dotenv file or the CIRQ_SCALEWAY_API_URL environment variables, if no url is found, then ``_ENDPOINT_URL`` is used.
    """

    def __init__(self, project_id: str = None, secret_key: str = None, url: str = None) -> None:
        env_token = dotenv_values().get("CIRQ_SCALEWAY_API_TOKEN") or os.getenv(
            "CIRQ_SCALEWAY_API_TOKEN"
        )
        env_project_id = dotenv_values().get("CIRQ_SCALEWAY_PROJECT_ID") or os.getenv(
            "CIRQ_SCALEWAY_PROJECT_ID"
        )
        env_api_url = dotenv_values().get("CIRQ_SCALEWAY_API_URL") or os.getenv(
            "CIRQ_SCALEWAY_API_URL"
        )

        token = secret_key or env_token
        if token is None:
            raise Exception("secret_key is missing")

        project_id = project_id or env_project_id
        if project_id is None:
            raise Exception("project_id is missing")

        api_url = url or env_api_url or _ENDPOINT_URL

        self.__client = QaaSClient(url=api_url, token=token, project_id=project_id)

    def devices(self, name: str = None, **kwargs) -> list[ScalewayDevice]:
        """Return a list of backends matching the specified filtering.

        Args:
            name (str): name of the backend.

        Returns:
            list[ScalewayBackend]: a list of Backends that match the filtering
                criteria.
        """

        scaleway_backends = []
        filters = {}

        if kwargs.get("operational") is not None:
            filters["operational"] = kwargs.pop("operational", None)

        if kwargs.get("min_num_qubits") is not None:
            filters["min_num_qubits"] = kwargs.pop("min_num_qubits", None)

        json_resp = self.__client.list_platforms(name)

        for platform_dict in json_resp["platforms"]:
            name = platform_dict.get("name")

            if name.startswith("qsim"):
                scaleway_backends.append(
                    ScalewayDevice(
                        client=self.__client,
                        id=platform_dict.get("id"),
                        name=name,
                        version=platform_dict.get("version"),
                        num_qubits=platform_dict.get("max_qubit_count"),
                        metadata=platform_dict.get("metadata", None),
                    )
                )

        if filters is not None:
            scaleway_backends = self.filters(scaleway_backends, filters)

        return scaleway_backends

    def filters(backends: list[ScalewayDevice], filters: dict) -> list[ScalewayDevice]:
        def _filter_availability(self, operational, availability):
            availabilities = (
                ["ailability_unknown", "available", "scarce"] if operational else ["shortage"]
            )

            return availability in availabilities

        operational = filters.get("operational")
        min_num_qubits = filters.get("min_num_qubits")

        if operational is not None:
            backends = [b for b in backends if _filter_availability(operational, b.availability)]

        if min_num_qubits is not None:
            backends = [b for b in backends if b.num_qubits >= min_num_qubits]

        return backends
