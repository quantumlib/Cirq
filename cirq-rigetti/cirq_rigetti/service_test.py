# pylint: disable=wrong-or-nonexistent-copyright-notice
from unittest.mock import patch

import pytest

from cirq_rigetti import get_rigetti_qcs_service
from cirq_rigetti.deprecation import allow_deprecated_cirq_rigetti_use_in_tests


@pytest.mark.rigetti_integration
@allow_deprecated_cirq_rigetti_use_in_tests
def test_get_rigetti_qcs_service():
    """test that get_rigetti_qcs_service can initialize a `RigettiQCSService`
    through `pyquil.get_qc`."""
    service = get_rigetti_qcs_service('9q-square', as_qvm=True, noisy=False)
    assert service._quantum_computer.name == '9q-square-qvm'


@pytest.mark.rigetti_integration
@patch('cirq_rigetti.service.QCSClient')
@patch('cirq_rigetti.service.list_quantum_processors')
@allow_deprecated_cirq_rigetti_use_in_tests
def test_list_quantum_processors(mock_list_quantum_processors, MockQCSClient):
    client = MockQCSClient()

    get_rigetti_qcs_service('9q-square', as_qvm=True).list_quantum_processors(client=client)

    mock_list_quantum_processors.assert_called_with(client=client)
