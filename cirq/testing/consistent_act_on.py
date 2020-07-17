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

from typing import Any, List, Optional

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
    """Checks that the stabilizer does not modify the value of the
    state_vector, including the global phase. Does not mutate the input
    state_vector."""

    args = act_on_state_vector_args.ActOnStateVectorArgs(
        target_tensor=state_vector.copy(),
        available_buffer=np.empty_like(state_vector),
        axes=range(protocols.num_qubits(stabilizer)),
        prng=np.random.RandomState(),
        log_of_measurement_results={})
    protocols.act_on(stabilizer, args)
    return np.allclose(args.target_tensor, state_vector)


def assert_act_on_clifford_tableau_effect_matches_unitary(val: Any) -> None:
    """Checks that act_on with CliffordTableau generates stabilizers that
    stabilize the final state vector. Does not work with Operations or Gates
    expecting non-qubit Qids."""

    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    num_qubits_val = protocols.num_qubits(val)

    if not protocols.has_unitary(val) or \
            protocols.qid_shape(val) != (2,) * num_qubits_val:
        return None

    qubits = LineQubit.range(protocols.num_qubits(val) * 2)
    qubit_map = {qubit: i for i, qubit in enumerate(qubits)}

    circuit = Circuit()
    for i in range(num_qubits_val):
        circuit.append([
            common_gates.H(qubits[i]),
            common_gates.CNOT(qubits[i], qubits[-i - 1])
        ])
    if hasattr(val, "on"):
        circuit.append(val.on(*qubits[:num_qubits_val]))
    else:
        circuit.append(val.with_qubits(*qubits[:num_qubits_val]))

    tableau = _final_clifford_tableau(circuit, qubit_map)
    if tableau is None:
        return None

    state_vector = np.reshape(final_state_vector(circuit, qubit_order=qubits),
                              protocols.qid_shape(qubits))

    assert all(
        state_vector_has_stabilizer(state_vector, stab)
        for stab in tableau.stabilizers()), (
            "act_on clifford tableau is not consistent with "
            "final_state_vector simulation.\n\nval: {!r}".format(val))


def _final_clifford_tableau(circuit: Circuit, qubit_map
                           ) -> Optional[clifford_tableau.CliffordTableau]:
    """Initializes a CliffordTableau with default args for the given qubits and
    evolves it by having each operation act on the tableau. Returns None if any
    of the operation can not act on a CliffordTableau, returns the tableau
    otherwise."""

    tableau = clifford_tableau.CliffordTableau(len(qubit_map))
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
    return tableau
