# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest
import httpx
from pytest_httpx import HTTPXMock

from cirq_rigetti import get_rigetti_qcs_service, RigettiQCSService


@pytest.mark.rigetti_integration
def test_get_rigetti_qcs_service():
    """test that get_rigetti_qcs_service can initialize a `RigettiQCSService`
    through `pyquil.get_qc`."""
    service = get_rigetti_qcs_service('9q-square', as_qvm=True, noisy=False)
    assert service._quantum_computer.name == '9q-square-qvm'


@pytest.mark.rigetti_integration
def test_rigetti_qcs_service_api_call(httpx_mock: HTTPXMock):
    """test that `RigettiQCSService` will use a custom defined client when the
    user specifies one to make an API call."""

    httpx_mock.add_response(json={"quantumProcessors": [{"id": "Aspen-8"}]})

    client = httpx.Client(base_url="https://mock.api.qcs.rigetti.com")

    response = RigettiQCSService.list_quantum_processors(client=client)
    assert 1 == len(response.quantum_processors)
    assert 'Aspen-8' == response.quantum_processors[0].id
