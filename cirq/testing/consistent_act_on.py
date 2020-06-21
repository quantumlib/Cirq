# Copyright 2019 The Cirq Developers
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

from typing import Any

import numpy as np

from cirq.circuits.circuit import Circuit
from cirq.devices import LineQubit
from cirq.ops import common_gates
from cirq.ops.dense_pauli_string import DensePauliString
from cirq import protocols
from cirq.sim import act_on_state_vector_args, final_state_vector
from cirq.sim.clifford import act_on_clifford_tableau_args, clifford_tableau


def state_vector_has_stabilizer(state_vector: np.ndarray,
                                stabilizer: DensePauliString) -> bool:
    original_state_vector = state_vector.copy()
    args = act_on_state_vector_args.ActOnStateVectorArgs(
        target_tensor=state_vector,
        available_buffer=np.empty(state_vector.shape, dtype=np.complex64),
        axes=range(protocols.num_qubits(stabilizer)),
        prng=np.random.RandomState(),
        log_of_measurement_results={})
    protocols.act_on(stabilizer, args)
    return np.allclose(args.target_tensor, original_state_vector)


def assert_act_on_clifford_tableau_effect_matches_unitary(val: Any) -> None:
    """Checks that act_on with CliffordTableau generates stabilizers that
    stabilize the final state vector."""

    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    if not protocols.has_unitary(val):
        return None

    qubits = LineQubit.range(2 * protocols.num_qubits(val))
    qubit_map = {qubit: i for i, qubit in enumerate(qubits)}
    circuit = Circuit()
    for i in range(protocols.num_qubits(val)):
        circuit.append([
            common_gates.H(qubits[2 * i]),
            common_gates.CNOT(qubits[2 * i], qubits[2 * i + 1])
        ])
    circuit.append(val.on(*qubits[::2]))
    state_vector = np.reshape(final_state_vector(circuit, qubit_order=qubits),
                              protocols.qid_shape(qubits))

    tableau = clifford_tableau.CliffordTableau(len(qubits))
    for op in circuit.all_operations():
        try:
            args = act_on_clifford_tableau_args.ActOnCliffordTableauArgs(
                tableau=tableau,
                axes=[qubit_map[qid] for qid in op.qubits],  # type: ignore
                prng=np.random.RandomState(),
                log_of_measurement_results={},
            )
            protocols.act_on(op, args, allow_decompose=True)
        except TypeError:
            return None

    assert all(
        state_vector_has_stabilizer(state_vector.copy(), stab)
        for stab in tableau.stabilizers()), (
        "act_on clifford tableau is not consistent with "
        "final_state_vector simulation.\n\nval: {!r}".format(val))
