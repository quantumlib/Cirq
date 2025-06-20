# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import cast
import pytest
import cirq
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler
from cirq_rigetti.deprecation import allow_deprecated_cirq_rigetti_use_in_tests


@pytest.mark.rigetti_integration
@allow_deprecated_cirq_rigetti_use_in_tests
def test_bell_circuit_through_sampler(bell_circuit: cirq.Circuit) -> None:
    """test that RigettiQCSSampler can run a basic bell circuit on the QVM and return an accurate
    ``cirq.study.Result``.
    """
    qc = get_qc('9q-square', as_qvm=True)
    sampler = RigettiQCSSampler(quantum_computer=qc)

    # set the seed so we get a deterministic set of results.
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 0

    repetitions = 10
    results = sampler.run_sweep(
        program=bell_circuit, params=[cirq.ParamResolver({})], repetitions=repetitions
    )
    assert 1 == len(results)

    result = results[0]
    assert isinstance(result, cirq.study.Result)
    assert 0 == len(result.params.param_dict)

    assert 'm' in result.measurements
    assert (repetitions, 2) == result.measurements['m'].shape

    counter = result.histogram(key='m')
    assert 6 == counter.get(0)
    assert counter.get(1) is None
    assert counter.get(2) is None
    assert 4 == counter.get(3)
