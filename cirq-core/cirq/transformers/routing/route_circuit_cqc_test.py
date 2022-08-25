# Copyright 2022 The Cirq Developers
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

from typing import Dict

from timeit import default_timer as timer
import cirq
import pytest

# add tests for
#   circuit with CircuitOp(s)
#   preserve_moment_structure = True
#   circuit that is alrady executable
#   circuit that is empty

def assert_same_unitary(
    c_orig,
    c_routed,
    imap: Dict['cirq.Qid', 'cirq.Qid'],
    fmap: Dict['cirq.Qid', 'cirq.Qid']):

    inverse_fmap = {v: k for k, v in fmap.items()}
    final_to_initial_mapping = {k: imap[inverse_fmap[k]] for k in inverse_fmap}
    sorted_grid_qubits = sorted(c_routed.all_qubits())
    if final_to_initial_mapping:
        x, y = zip(*sorted(final_to_initial_mapping.items(), key=lambda x: x[1]))
        perm = [*range(len(sorted_grid_qubits))]
        for i, q in enumerate(sorted_grid_qubits):
            index = y.index(x[i])
            perm[index] = i
        c_routed.append(cirq.QubitPermutationGate(perm).on(*sorted_grid_qubits))

        _, grid_order = zip(*sorted(list(imap.items()), key=lambda x: x[0]))
        cirq.testing.assert_allclose_up_to_global_phase(
            c_orig.unitary(), c_routed.unitary(qubit_order=grid_order), atol=1e-8
        )
    else:
        cirq.testing.assert_allclose_up_to_global_phase(
            c_orig.unitary(), c_routed.unitary(), atol=1e-8
        )


@pytest.mark.parametrize(
    "n_qubits, n_moments, op_density, seed",
    [
        (8 , size, op_density, seed)
        for size in [50, 100, 500]
        for seed in range(3)
        for op_density in [0.3, 0.5, 0.7]
    ],
)
def test_route_small_circuit_random(n_qubits, n_moments, op_density, seed):
    c_orig = cirq.testing.random_circuit(
        qubits=n_qubits, n_moments=n_moments, op_density=op_density, random_state=seed
    )
    device = cirq.testing.construct_grid_device(4, 4)
    router = cirq.RouteCQC(device)
    c_routed, initial_mapping, final_mapping = router.route_circuit(c_orig)

    device.validate_circuit(c_routed)
    assert_same_unitary(c_orig, c_routed, initial_mapping, final_mapping)


def test_high_qubit_count():
    c_orig = cirq.testing.random_circuit(
        qubits=54,
        n_moments=500,
        op_density=0.4,
        random_state=0,
    ) 
    device = cirq.testing.construct_grid_device(7, 8)
    router = cirq.RouteCQC(device)
    c_routed = router(c_orig)
    device.validate_circuit(c_routed)


def construct_valid_circuit():
    return cirq.Circuit([
        cirq.Moment(
            cirq.X(cirq.NamedQubit('2')),
            cirq.X(cirq.NamedQubit('11')),
            cirq.X(cirq.NamedQubit('1')),
            cirq.X(cirq.NamedQubit('13')),
            cirq.X(cirq.NamedQubit('0')),
            cirq.X(cirq.NamedQubit('14')),
            cirq.X(cirq.NamedQubit('6')),
            cirq.CNOT(cirq.NamedQubit('3'), cirq.NamedQubit('9')),
            cirq.CNOT(cirq.NamedQubit('8'), cirq.NamedQubit('12')),
            cirq.X(cirq.NamedQubit('7')),
            cirq.X(cirq.NamedQubit('4')),
            cirq.X(cirq.NamedQubit('15')),
        ),
        cirq.Moment(
            cirq.X(cirq.NamedQubit('2')),
            cirq.X(cirq.NamedQubit('9')),
            cirq.X(cirq.NamedQubit('3')),
            cirq.X(cirq.NamedQubit('1')),
            cirq.CNOT(cirq.NamedQubit('10'), cirq.NamedQubit('11')),
            cirq.CNOT(cirq.NamedQubit('8'), cirq.NamedQubit('12')),
            cirq.CNOT(cirq.NamedQubit('14'), cirq.NamedQubit('6')),
            cirq.CNOT(cirq.NamedQubit('5'), cirq.NamedQubit('4')),
            cirq.X(cirq.NamedQubit('15')),
        ),
        cirq.Moment(
            cirq.X(cirq.NamedQubit('1')),
            cirq.X(cirq.NamedQubit('12')),
            cirq.CNOT(cirq.NamedQubit('8'), cirq.NamedQubit('2')),
            cirq.CNOT(cirq.NamedQubit('3'), cirq.NamedQubit('9')),
            cirq.CNOT(cirq.NamedQubit('6'), cirq.NamedQubit('0')),
            cirq.CNOT(cirq.NamedQubit('14'), cirq.NamedQubit('10')),
        ),
        cirq.Moment(
            cirq.X(cirq.NamedQubit('10')),
            cirq.CNOT(cirq.NamedQubit('8'), cirq.NamedQubit('12')),
            cirq.CNOT(cirq.NamedQubit('14'), cirq.NamedQubit('6')),
            cirq.CNOT(cirq.NamedQubit('1'), cirq.NamedQubit('4')),
            cirq.X(cirq.NamedQubit('9')),
            cirq.X(cirq.NamedQubit('0')),
        ),
        cirq.Moment(
            cirq.X(cirq.NamedQubit('1')),
            cirq.X(cirq.NamedQubit('6')),
            cirq.CNOT(cirq.NamedQubit('8'), cirq.NamedQubit('12')),
            cirq.CNOT(cirq.NamedQubit('14'), cirq.NamedQubit('10')),
        ),
    ])


def test_empty_and_executable():
    device = cirq.testing.construct_grid_device(5, 5)
    empty_circuit = cirq.Circuit(cirq.Moment() for i in range(10))
    valid_circuit = construct_valid_circuit()
    print(valid_circuit)
    # device.validate_circuit(empty_circuit)
    # device.validate_circuit(valid_circuit)

    # router = cirq.RouteCQC(device)
    # empty_circuit_routed, imap_empty, fmap_empty = router(empty_circuit)
    # valid_circuit_routed, imap_valid, fmap_routed = router(valid_circuit)

    # device.validate_circuit(empty_circuit_routed)
    # device.validate_circuit(valid_circuit_routed)
    # assert_same_unitary(empty_circuit, empty_circuit_routed, imap_empty, fmap_empty)
    # assert_same_unitary(valid_circuit, valid_circuit_routed, imap_valid, fmap_routed)
    # assert len(list(empty_circuit.all_operations())) == len(list(empty_circuit_routed.all_operations()))
    # assert len(list(valid_circuit.all_operations())) == len(list(valid_circuit_routed.all_operations()))

