# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses

import cirq
import cirq_google
import pytest
from cirq_google import QuantumExecutable, BitstringsMeasurement, ExecutableSpec


def test_bitstrings_measurement():
    bs = BitstringsMeasurement(n_repetitions=10_000)
    cirq.testing.assert_equivalent_repr(bs, global_vals={'cirq_google': cirq_google})


def _get_random_circuit(qubits, n_moments=10, op_density=0.8, random_state=52):
    return cirq.testing.random_circuit(
        qubits, n_moments=n_moments, op_density=op_density, random_state=random_state
    )


@dataclasses.dataclass(frozen=True)
class ExampleSpec(ExecutableSpec):
    name: str
    executable_family: str = 'cirq_google.algo_benchmarks.example'

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

    assert (
        str(exe) == "QuantumExecutable(spec=ExampleSpec(name='example-program', "
        "executable_family='cirq_google.algo_benchmarks.example'))"
    )
    cirq.testing.assert_equivalent_repr(
        exe, global_vals={'ExampleSpec': ExampleSpec, 'cirq_google': cirq_google}
    )


def test_quantum_executable_inputs():
    qubits = cirq.LineQubit.range(10)
    spec = ExampleSpec(name='example-program')
    circuit = _get_random_circuit(qubits)
    measurement = BitstringsMeasurement(n_repetitions=10)

    params1 = {'theta': 0.2}
    params2 = cirq.ParamResolver({'theta': 0.2})
    params3 = [('theta', 0.2)]
    params4 = (('theta', 0.2),)
    exes = [
        QuantumExecutable(spec=spec, circuit=circuit, measurement=measurement, params=p)
        for p in [params1, params2, params3, params4]
    ]
    for exe in exes:
        assert exe == exes[0]

    with pytest.raises(ValueError):
        _ = QuantumExecutable(
            spec=spec, circuit=circuit, measurement=measurement, params='theta=0.2'
        )
    with pytest.raises(TypeError):
        _ = QuantumExecutable(spec={'name': 'main'}, circuit=circuit, measurement=measurement)
