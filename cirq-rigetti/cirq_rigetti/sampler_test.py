# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest

from cirq_rigetti import get_rigetti_qcs_sampler
from cirq_rigetti.deprecation import allow_deprecated_cirq_rigetti_use_in_tests


@pytest.mark.rigetti_integration
@allow_deprecated_cirq_rigetti_use_in_tests
def test_get_rigetti_qcs_sampler():
    """test that get_rigetti_qcs_sampler can initialize a `RigettiQCSSampler`
    through `pyquil.get_qc`."""
    sampler = get_rigetti_qcs_sampler('9q-square', as_qvm=True, noisy=False)
    assert sampler._quantum_computer.name == '9q-square-qvm'
