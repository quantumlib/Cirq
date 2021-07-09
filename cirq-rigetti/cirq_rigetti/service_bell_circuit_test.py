from typing import cast
import pytest
import cirq
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSService


@pytest.mark.rigetti_integration
def test_bell_circuit_through_service(bell_circuit: cirq.Circuit) -> None:
    """test that RigettiQCSService can run a basic bell circuit on the QVM and return an accurate
    ``cirq.study.Result``.
    """
    qc = get_qc('9q-square', as_qvm=True)
    service = RigettiQCSService(
        quantum_computer=qc,
    )

    # set the seed so we get a deterministic set of results.
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 0

    repetitions = 10
    result = service.run(circuit=bell_circuit, repetitions=repetitions)
    assert isinstance(result, cirq.study.Result)
    assert 0 == len(result.params.param_dict)

    assert 'm' in result.measurements
    assert (repetitions, 2) == result.measurements['m'].shape

    counter = result.histogram(key='m')
    assert 6 == counter.get(0)
    assert counter.get(1) is None
    assert counter.get(2) is None
    assert 4 == counter.get(3)
