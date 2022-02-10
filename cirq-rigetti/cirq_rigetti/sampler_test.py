# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest
from cirq_rigetti import get_rigetti_qcs_sampler


@pytest.mark.rigetti_integration
def test_get_rigetti_qcs_sampler():
    """Ttest that get_rigetti_qcs_sampler can initialize a `RigettiQCSSampler`.

    This tests that it uses `pyquil.get_qc` to do this.
    """
    sampler = get_rigetti_qcs_sampler('9q-square', as_qvm=True, noisy=False)
    assert sampler._quantum_computer.name == '9q-square-qvm'
