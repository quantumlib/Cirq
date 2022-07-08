# Copyright 2020 The Cirq Developers
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

from typing import Any, cast, Optional, Type

import numpy as np

from cirq.circuits.circuit import Circuit
from cirq.devices import LineQubit
from cirq.ops import common_gates
from cirq.ops.dense_pauli_string import DensePauliString
from cirq import protocols
from cirq.qis import clifford_tableau
from cirq.sim import state_vector_simulation_state, final_state_vector
from cirq.sim.clifford import (
    clifford_tableau_simulation_state,
    stabilizer_state_ch_form,
    stabilizer_ch_form_simulation_state,
)


def state_vector_has_stabilizer(state_vector: np.ndarray, stabilizer: DensePauliString) -> bool:
    """Checks that the state_vector is stabilized by the given stabilizer.

    The stabilizer should not modify the value of the state_vector, up to the
    global phase.

    Args:
        state_vector: An input state vector. Is not mutated by this function.
        stabilizer: A potential stabilizer of the above state_vector as a
          DensePauliString.

    Returns:
        Whether the stabilizer stabilizes the supplied state.
    """

    qubits = LineQubit.range(protocols.num_qubits(stabilizer))
    complex_dtype: Type[np.complexfloating] = np.complex64
    if np.issubdtype(state_vector.dtype, np.complexfloating):
        complex_dtype = cast(Type[np.complexfloating], state_vector.dtype)
    args = state_vector_simulation_state.StateVectorSimulationState(
        available_buffer=np.empty_like(state_vector),
        qubits=qubits,
        prng=np.random.RandomState(),
        initial_state=state_vector.copy(),
        dtype=complex_dtype,
    )
    protocols.act_on(stabilizer, args, qubits)
    return np.allclose(args.target_tensor, state_vector)


def assert_all_implemented_act_on_effects_match_unitary(
    val: Any, assert_tableau_implemented: bool = False, assert_ch_form_implemented: bool = False
) -> None:
    """Uses val's effect on final_state_vector to check act_on(val)'s behavior.

    Checks that act_on with CliffordTableau or StabilizerStateCHForm behaves
    consistently with act_on through final state vector. Does not work with
    Operations or Gates expecting non-qubit Qids. If either of the
    assert_*_implmented args is true, fails if the corresponding method is not
    implemented for the test circuit.

    Args:
        val: A gate or operation that may be an input to protocols.act_on.
        assert_tableau_implemented: asserts that protocols.act_on() works with
          val and CliffordTableauSimulationState inputs.
        assert_ch_form_implemented: asserts that protocols.act_on() works with
          val and StabilizerChFormSimulationState inputs.
    """

    # pylint: disable=unused-variable
    __tracebackhide__ = True
    # pylint: enable=unused-variable

    num_qubits_val = protocols.num_qubits(val)

    if (
        protocols.is_parameterized(val)
        or not protocols.has_unitary(val)
        or protocols.qid_shape(val) != (2,) * num_qubits_val
    ):
        if assert_tableau_implemented or assert_ch_form_implemented:
            assert False, (
                "Could not assert if any act_on methods were "
                "implemented. Operating on qudits or with a "
                "non-unitary or parameterized operation is "
                "unsupported.\n\nval: {!r}".format(val)
            )
        return None

    qubits = LineQubit.range(num_qubits_val * 2)
    qubit_map = {qubit: i for i, qubit in enumerate(qubits)}

    circuit = Circuit()
    for i in range(num_qubits_val):
        circuit.append([common_gates.H(qubits[i]), common_gates.CNOT(qubits[i], qubits[-i - 1])])
    if hasattr(val, "on"):
        circuit.append(val.on(*qubits[:num_qubits_val]))
    else:
        circuit.append(val.with_qubits(*qubits[:num_qubits_val]))

    state_vector = np.reshape(
        final_state_vector(circuit, qubit_order=qubits), protocols.qid_shape(qubits)
    )

    tableau = _final_clifford_tableau(circuit, qubit_map)
    if tableau is None:
        assert (
            not assert_tableau_implemented
        ), f"Failed to generate final tableau for the test circuit.\n\nval: {val!r}"
    else:
        assert all(
            state_vector_has_stabilizer(state_vector, stab) for stab in tableau.stabilizers()
        ), (
            "act_on clifford tableau is not consistent with "
            "final_state_vector simulation.\n\nval: {!r}".format(val)
        )

    stabilizer_ch_form = _final_stabilizer_state_ch_form(circuit, qubit_map)
    if stabilizer_ch_form is None:
        assert not assert_ch_form_implemented, (
            "Failed to generate final "
            "stabilizer state CH form "
            "for the test circuit."
            "\n\nval: {!r}".format(val)
        )
    else:
        np.testing.assert_allclose(
            np.reshape(stabilizer_ch_form.state_vector(), protocols.qid_shape(qubits)),
            state_vector,
            atol=1e-07,
            err_msg=f"stabilizer_ch_form.state_vector disagrees with state_vector for {val!r}",
            verbose=True,
        )


def _final_clifford_tableau(
    circuit: Circuit, qubit_map
) -> Optional[clifford_tableau.CliffordTableau]:
    """Evolves a default CliffordTableau through the input circuit.

    Initializes a CliffordTableau with default args for the given qubits and
    evolves it by having each operation act on the tableau.

    Args:
        circuit: An input circuit that acts on the zero state
        qubit_map: A map from qid to the qubit index for the above circuit

    Returns:
        None if any of the operations can not act on a CliffordTableau, returns
        the tableau otherwise."""

    tableau = clifford_tableau.CliffordTableau(len(qubit_map))
    args = clifford_tableau_simulation_state.CliffordTableauSimulationState(
        tableau=tableau, qubits=list(qubit_map.keys()), prng=np.random.RandomState()
    )
    for op in circuit.all_operations():
        try:
            protocols.act_on(op, args, allow_decompose=True)
        except TypeError:
            return None
    return tableau


def _final_stabilizer_state_ch_form(
    circuit: Circuit, qubit_map
) -> Optional[stabilizer_state_ch_form.StabilizerStateChForm]:
    """Evolves a default StabilizerStateChForm through the input circuit.

    Initializes a StabilizerStateChForm with default args for the given qubits
    and evolves it by having each operation act on the state.

    Args:
        circuit: An input circuit that acts on the zero state
        qubit_map: A map from qid to the qubit index for the above circuit

    Returns:
        None if any of the operations can not act on a StabilizerStateChForm,
        returns the StabilizerStateChForm otherwise."""

    stabilizer_ch_form = stabilizer_state_ch_form.StabilizerStateChForm(len(qubit_map))
    args = stabilizer_ch_form_simulation_state.StabilizerChFormSimulationState(
        qubits=list(qubit_map.keys()),
        prng=np.random.RandomState(),
        initial_state=stabilizer_ch_form,
    )
    for op in circuit.all_operations():
        try:
            protocols.act_on(op, args, allow_decompose=True)
        except TypeError:
            return None
    return stabilizer_ch_form
