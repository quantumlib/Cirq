import os
import cirq

from collections.abc import Callable
from dotenv import dotenv_values

from .scaleway_sampler import ScalewaySampler
from .scaleway_client import QaaSClient


_ENDPOINT_URL = "https://api.scaleway.com/qaas/v1alpha1"


class ScalewayService:
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

    def samplers(self, name: str = None, **kwargs) -> list[ScalewaySampler]:
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

            backend = None

            if name.startswith("qsim"):
                backend = ScalewaySampler(
                    client=self.__client,
                    id=platform_dict.get("id"),
                    name=name,
                    availability=platform_dict.get("availability"),
                    version=platform_dict.get("version"),
                    num_qubits=platform_dict.get("max_qubit_count"),
                    metadata=platform_dict.get("metadata", None),
                )

            scaleway_backends.append(backend)

        if filters is not None:
            scaleway_backends = self.filters(scaleway_backends, filters)

        return filter_samplers(scaleway_backends, **kwargs)

    def _filter_availability(self, operational, availability):
        availabilities = (
            ["ailability_unknown", "available", "scarce"] if operational else ["shortage"]
        )

        return availability in availabilities

    def filters(self, backends: list[ScalewaySampler], filters: dict) -> list[ScalewaySampler]:
        operational = filters.get("operational")
        min_num_qubits = filters.get("min_num_qubits")

        if operational is not None:
            backends = [
                b for b in backends if self._filter_availability(operational, b.availability)
            ]

        if min_num_qubits is not None:
            backends = [b for b in backends if b.num_qubits >= min_num_qubits]

        return backends


def filter_samplers(
    backends: list[cirq.work.Sampler], filters: Callable = None, **kwargs
) -> list[cirq.work.Sampler]:
    def _match_all(obj, criteria):
        """Return True if all items in criteria matches items in obj."""
        return all(getattr(obj, key_, None) == value_ for key_, value_ in criteria.items())

    # Inspect the backends to decide which filters belong to
    # backend.configuration and which ones to backend.status, as it does
    # not involve querying the API.
    configuration_filters = {}
    status_filters = {}
    for key, value in kwargs.items():
        if all(key in backend.configuration() for backend in backends):
            configuration_filters[key] = value
        else:
            status_filters[key] = value

    # 1. Apply backend.configuration filtering.
    if configuration_filters:
        backends = [b for b in backends if _match_all(b.configuration(), configuration_filters)]

    # 2. Apply backend.status filtering (it involves one API call for
    # each backend).
    if status_filters:
        backends = [b for b in backends if _match_all(b.status(), status_filters)]

    # 3. Apply acceptor filter.
    backends = list(filter(filters, backends))

    return backends
