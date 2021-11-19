# Copyright 2018 The Cirq Developers
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
"""A protocol for implementing high performance clifford tableau evolutions
 for Clifford Simulator."""

from typing import Any, Dict, TYPE_CHECKING, List, Sequence, Union

import numpy as np

from cirq import protocols, ops
from cirq.ops import common_gates
from cirq.ops import pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.qis.clifford_tableau import CliffordTableau
from cirq.sim.act_on_args import ActOnArgs
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


class ActOnCliffordTableauArgs(ActOnArgs):
    """State and context for an operation acting on a clifford tableau.

    To act on this object, directly edit the `tableau` property, which is
    storing the density matrix of the quantum system with one axis per qubit.
    """

    def __init__(
        self,
        tableau: CliffordTableau,
        prng: np.random.RandomState,
        log_of_measurement_results: Dict[str, Any],
        qubits: Sequence['cirq.Qid'] = None,
    ):
        """Inits ActOnCliffordTableauArgs.

        Args:
            tableau: The CliffordTableau to act on. Operations are expected to
                perform inplace edits of this object.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
        """
        super().__init__(prng, qubits, log_of_measurement_results)
        self.tableau = tableau

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> Union[bool, NotImplementedType]:
        strats = [self._strat_apply_to_tableau]
        if allow_decompose:
            strats.append(self._strat_act_on_clifford_tableau_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, qubits)
            if result is False:
                break  # coverage: ignore
            if result is True:
                return True
            assert result is NotImplemented, str(result)

        return NotImplemented

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Returns the measurement from the tableau."""
        return [self.tableau._measure(self.qubit_map[q], self.prng) for q in qubits]

    def _on_copy(self, target: 'ActOnCliffordTableauArgs'):
        target.tableau = self.tableau.copy()

    def sample(
        self,
        qubits: Sequence['cirq.Qid'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        # Unnecessary for now but can be added later if there is a use case.
        raise NotImplementedError()

    def _x(self, exponent: float, axis: int):
        assert exponent % 0.5 == 0.0
        tableau = self.tableau
        effective_exponent = exponent % 2
        if effective_exponent == 0.5:
            tableau.xs[:, axis] ^= tableau.zs[:, axis]
            tableau.rs[:] ^= tableau.xs[:, axis] & tableau.zs[:, axis]
        elif effective_exponent == 1:
            tableau.rs[:] ^= tableau.zs[:, axis]
        elif effective_exponent == 1.5:
            tableau.rs[:] ^= tableau.xs[:, axis] & tableau.zs[:, axis]
            tableau.xs[:, axis] ^= tableau.zs[:, axis]

    def _y(self, exponent: float, axis: int):
        assert exponent % 0.5 == 0.0
        tableau = self.tableau
        effective_exponent = exponent % 2
        if effective_exponent == 0.5:
            tableau.rs[:] ^= tableau.xs[:, axis] & (~tableau.zs[:, axis])
            (tableau.xs[:, axis], tableau.zs[:, axis]) = (
                tableau.zs[:, axis].copy(),
                tableau.xs[:, axis].copy(),
            )
        elif effective_exponent == 1:
            tableau.rs[:] ^= tableau.xs[:, axis] ^ tableau.zs[:, axis]
        elif effective_exponent == 1.5:
            tableau.rs[:] ^= ~(tableau.xs[:, axis]) & tableau.zs[:, axis]
            (tableau.xs[:, axis], tableau.zs[:, axis]) = (
                tableau.zs[:, axis].copy(),
                tableau.xs[:, axis].copy(),
            )

    def _z(self, exponent: float, axis: int):
        assert exponent % 0.5 == 0.0
        tableau = self.tableau
        effective_exponent = exponent % 2
        if effective_exponent == 0.5:
            tableau.rs[:] ^= tableau.xs[:, axis] & tableau.zs[:, axis]
            tableau.zs[:, axis] ^= tableau.xs[:, axis]
        elif effective_exponent == 1:
            tableau.rs[:] ^= tableau.xs[:, axis]
        elif effective_exponent == 1.5:
            tableau.rs[:] ^= tableau.xs[:, axis] & (~tableau.zs[:, axis])
            tableau.zs[:, axis] ^= tableau.xs[:, axis]

    def _cz(self, exponent: float, axis1: int, axis2: int):
        assert exponent % 2 == 1
        tableau = self.tableau
        (tableau.xs[:, axis2], tableau.zs[:, axis2]) = (
            tableau.zs[:, axis2].copy(),
            tableau.xs[:, axis2].copy(),
        )
        tableau.rs[:] ^= tableau.xs[:, axis2] & tableau.zs[:, axis2]
        tableau.rs[:] ^= (
            tableau.xs[:, axis1]
            & tableau.zs[:, axis2]
            & (~(tableau.xs[:, axis2] ^ tableau.zs[:, axis1]))
        )
        tableau.xs[:, axis2] ^= tableau.xs[:, axis1]
        tableau.zs[:, axis1] ^= tableau.zs[:, axis2]
        (tableau.xs[:, axis2], tableau.zs[:, axis2]) = (
            tableau.zs[:, axis2].copy(),
            tableau.xs[:, axis2].copy(),
        )
        tableau.rs[:] ^= tableau.xs[:, axis2] & tableau.zs[:, axis2]

    def _cx(self, exponent: float, axis1: int, axis2: int):
        assert exponent % 2 == 1
        tableau = self.tableau
        tableau.rs[:] ^= (
            tableau.xs[:, axis1]
            & tableau.zs[:, axis2]
            & (~(tableau.xs[:, axis2] ^ tableau.zs[:, axis1]))
        )
        tableau.xs[:, axis2] ^= tableau.xs[:, axis1]
        tableau.zs[:, axis1] ^= tableau.zs[:, axis2]

    def _strat_apply_to_tableau(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
        val = val.gate if isinstance(val, ops.Operation) else val
        paulis = protocols.as_paulis(val, self.prng)
        if paulis is not NotImplemented:
            for pauli, exponent, raw_axes in paulis:
                qubits = [qubits[i] for i in raw_axes]
                axes = self.get_axes(qubits)
                if pauli == 'X':
                    self._x(exponent, axes[0])
                elif pauli == 'Y':
                    self._y(exponent, axes[0])
                elif pauli == 'Z':
                    self._z(exponent, axes[0])
                elif pauli == 'CZ':
                    self._cz(exponent, axes[0], axes[1])
                elif pauli == 'CX':
                    self._cx(exponent, axes[0], axes[1])
                else:
                    assert False
            return True
        else:
            gate = val.gate if isinstance(val, ops.Operation) else val
            return protocols.apply_to_tableau(gate, self.tableau, self.get_axes(qubits), self.prng)

    def _strat_act_on_clifford_tableau_from_single_qubit_decompose(
        self, val: Any, qubits: Sequence['cirq.Qid']
    ) -> bool:
        if num_qubits(val) == 1:
            if not has_unitary(val):
                return NotImplemented
            u = unitary(val)
            clifford_gate = SingleQubitCliffordGate.from_unitary(u)
            if clifford_gate is not None:
                axis = self.qubit_map[qubits[0]]
                for gate, quarter_turns in clifford_gate.decompose_rotation():
                    if gate == pauli_gates.X:
                        self._x(quarter_turns / 2, axis)
                    elif gate == pauli_gates.Y:
                        self._y(quarter_turns / 2, axis)
                    else:
                        assert gate == pauli_gates.Z
                        self._z(quarter_turns / 2, axis)
                return True

        return NotImplemented
