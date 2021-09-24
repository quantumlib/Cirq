import dataclasses
import itertools

import cirq
import numpy as np
import pytest
from cirq import TiltedSquareLattice

from cirq_google import QuantumExecutable, BitstringsMeasurement, ExecutableSpec


def _get_random_circuit(qubits, n_moments=10, op_density=0.8, random_state=52):
    return cirq.testing.random_circuit(
        qubits, n_moments=n_moments, op_density=op_density, random_state=random_state
    )


def get_all_diagonal_rect_topologies(min_side_length=2, max_side_length=8):
    width_heights = np.arange(min_side_length, max_side_length + 1)
    return [
        TiltedSquareLattice(width, height)
        for width, height in itertools.combinations_with_replacement(width_heights, r=2)
    ]


@dataclasses.dataclass(frozen=True)
class ExampleSpec(ExecutableSpec):
    name: str
    executable_family = 'cirq_google.algo_benchmarks.example'

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self, namespace='cirq.google.testing')


def _testing_resolver(cirq_type: str):
    if cirq_type == 'cirq.google.testing.ExampleSpec':
        return ExampleSpec


def test_quantum_executable(tmpdir):
    qubits = cirq.LineQubit.range(10)
    exe = QuantumExecutable(
        spec=ExampleSpec(name='example-program'),
        circuit=_get_random_circuit(qubits),
        measurement=BitstringsMeasurement(n_repetitions=10),
    )

    # Check args get turned into immutable fields
    assert isinstance(exe.circuit, cirq.FrozenCircuit)

    assert hash(exe) is not None
    assert hash(dataclasses.astuple(exe)) is not None
    assert hash(dataclasses.astuple(exe)) == exe._hash

    prog2 = QuantumExecutable(
        spec=ExampleSpec(name='example-program'),
        circuit=_get_random_circuit(qubits),
        measurement=BitstringsMeasurement(n_repetitions=10),
    )
    assert exe == prog2
    assert hash(exe) == hash(prog2)

    prog3 = QuantumExecutable(
        spec=ExampleSpec(name='example-program'),
        circuit=_get_random_circuit(qubits),
        measurement=BitstringsMeasurement(n_repetitions=20),  # note: changed n_repetitions
    )
    assert exe != prog3
    assert hash(exe) != hash(prog3)

    with pytest.raises(dataclasses.FrozenInstanceError):
        prog3.measurement.n_repetitions = 10

    cirq.to_json(exe, f'{tmpdir}/exe.json')
    exe_reconstructed = cirq.read_json(
        f'{tmpdir}/exe.json', resolvers=[_testing_resolver] + cirq.DEFAULT_RESOLVERS
    )
    assert exe == exe_reconstructed
