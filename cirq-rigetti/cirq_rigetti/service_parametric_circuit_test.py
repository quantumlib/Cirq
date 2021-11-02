# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import cast, Tuple
import pytest
import cirq
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSService


@pytest.mark.rigetti_integration
def test_parametric_circuit_through_service(
    parametric_circuit_with_params: Tuple[cirq.Circuit, cirq.Linspace]
) -> None:
    """test that RigettiQCSService can run a basic parametric circuit on
    the QVM and return an accurate `cirq.study.Result`.
    """
    circuit, sweepable = parametric_circuit_with_params

    qc = get_qc('9q-square', as_qvm=True)
    service = RigettiQCSService(
        quantum_computer=qc,
    )

    # set the seed so we get a deterministic set of results.
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 0

    repetitions = 10
    param_resolvers = [r for r in cirq.study.to_resolvers(sweepable)]
    result = service.run(
        circuit=circuit,
        repetitions=repetitions,
        param_resolver=param_resolvers[1],
    )
    assert isinstance(result, cirq.study.Result)
    assert sweepable[1] == result.params

    assert 'm' in result.measurements
    assert (repetitions, 1) == result.measurements['m'].shape

    counter = result.histogram(key='m')
    assert 4 == counter[0]
    assert 6 == counter[1]
