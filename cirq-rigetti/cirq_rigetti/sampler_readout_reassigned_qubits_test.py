# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import cast, List, Tuple

import pytest
import sympy
from pyquil import get_qc
from pyquil.api import QVM

import cirq
from cirq_rigetti import circuit_transformers, RigettiQCSSampler
from cirq_rigetti.deprecation import allow_deprecated_cirq_rigetti_use_in_tests


@pytest.fixture
def circuit_data() -> Tuple[cirq.Circuit, List[cirq.LineQubit], cirq.Linspace]:
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.X(qubits[0]) ** sympy.Symbol('t'))
    circuit.append(cirq.measure(qubits[0], qubits[1], key='m'))

    param_sweep = cirq.Linspace('t', start=0, stop=2, length=5)

    return circuit, qubits, param_sweep


@pytest.mark.rigetti_integration
@allow_deprecated_cirq_rigetti_use_in_tests
def test_readout_on_reassigned_qubits(
    circuit_data: Tuple[cirq.Circuit, List[cirq.LineQubit], cirq.Linspace],
) -> None:
    """test that RigettiQCSSampler can properly readout qubits after quilc has
    reassigned those qubits in the compiled native Quil.
    """
    qc = get_qc('9q-square', as_qvm=True)
    circuit, qubits, sweepable = circuit_data

    transformer = circuit_transformers.build(qubit_id_map={qubits[0]: '100', qubits[1]: '101'})
    sampler = RigettiQCSSampler(quantum_computer=qc, transformer=transformer)

    # set the seed so we get a deterministic set of results.
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 11

    repetitions = 10

    results = sampler.run_sweep(program=circuit, params=sweepable, repetitions=repetitions)
    assert len(sweepable) == len(results)

    for i, result in enumerate(results):
        assert isinstance(result, cirq.study.Result)
        assert sweepable[i] == result.params

        assert 'm' in result.measurements
        assert (repetitions, 2) == result.measurements['m'].shape

        counter = result.histogram(key='m')
        assert 2 == len(counter)
        assert 3 == counter.get(0)
        assert 7 == counter.get(2)
