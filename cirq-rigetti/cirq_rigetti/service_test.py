# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest
from cirq_rigetti import get_rigetti_qcs_service


@pytest.mark.rigetti_integration
def test_get_rigetti_qcs_service():
    """test that get_rigetti_qcs_service can initialize a `RigettiQCSService`
    through `pyquil.get_qc`."""
    service = get_rigetti_qcs_service('9q-square', as_qvm=True, noisy=False)
    assert service._quantum_computer.name == '9q-square-qvm'  # pragma: no cover
