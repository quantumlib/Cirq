# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest
from unittest.mock import patch
from cirq_rigetti import get_rigetti_qcs_service


@pytest.mark.rigetti_integration
def test_get_rigetti_qcs_service():
    """test that get_rigetti_qcs_service can initialize a `RigettiQCSService`
    through `pyquil.get_qc`."""
    service = get_rigetti_qcs_service('9q-square', as_qvm=True, noisy=False)
    assert service._quantum_computer.name == '9q-square-qvm'  # pragma: no cover


@patch('cirq_rigetti.service.QCSClient')
@patch('cirq_rigetti.service.list_quantum_processors')
def test_list_quantum_processors(mock_list_quantum_processors, MockQCSClient):
    client = MockQCSClient()

    get_rigetti_qcs_service('Aspen-8').list_quantum_processors(client=client)

    mock_list_quantum_processors.assert_called_with(client=client)
