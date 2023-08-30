# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import Optional, Iterator
import pytest
import httpx
from cirq_rigetti import get_rigetti_qcs_service, RigettiQCSService


@pytest.mark.rigetti_integration
def test_get_rigetti_qcs_service():
    """test that get_rigetti_qcs_service can initialize a `RigettiQCSService`
    through `pyquil.get_qc`."""
    service = get_rigetti_qcs_service('9q-square', as_qvm=True, noisy=False)
    assert service._quantum_computer.name == '9q-square-qvm'


@pytest.mark.rigetti_integration
def test_rigetti_qcs_service_api_call():
    """test that `RigettiQCSService` will use a custom defined client when the
    user specifies one to make an API call."""

    class Response(httpx.Response):
        def iter_bytes(self, chunk_size: Optional[int] = None) -> Iterator[bytes]:
            yield b"{\"quantumProcessors\": [{\"id\": \"Aspen-8\"}]}"  # pragma: no cover

    class Transport(httpx.BaseTransport):
        def handle_request(self, request: httpx.Request) -> httpx.Response:
            return Response(200)

    client = httpx.Client(base_url="https://mock.api.qcs.rigetti.com", transport=Transport())

    response = RigettiQCSService.list_quantum_processors(client=client)
    assert 1 == len(response.quantum_processors)
    assert 'Aspen-8' == response.quantum_processors[0].id
