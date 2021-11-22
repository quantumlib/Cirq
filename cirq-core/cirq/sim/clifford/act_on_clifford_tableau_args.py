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
from cirq.ops import common_gates, matrix_gates
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
        strats = [self._strat_apply_to_tableau, self._strat_apply_mixture_to_tableau]
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

    def _x(self, g: common_gates.XPowGate, axis: int):
        exponent = g.exponent
        if exponent % 2 == 0:
            return
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

    def _y(self, g: common_gates.YPowGate, axis: int):
        exponent = g.exponent
        if exponent % 2 == 0:
            return
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

    def _z(self, g: common_gates.ZPowGate, axis: int):
        exponent = g.exponent
        if exponent % 2 == 0:
            return
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

    def _h(self, g: common_gates.HPowGate, axis: int):
        exponent = g.exponent
        if exponent % 2 == 0:
            return
        assert exponent % 2 == 1
        self._y(common_gates.YPowGate(exponent=0.5), axis)
        self._x(common_gates.XPowGate(), axis)

    def _cz(self, g: common_gates.CZPowGate, axis1: int, axis2: int):
        exponent = g.exponent
        if exponent % 2 == 0:
            return
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

    def _cx(self, g: common_gates.CXPowGate, axis1: int, axis2: int):
        exponent = g.exponent
        if exponent % 2 == 0:
            return
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
        if not protocols.has_stabilizer_effect(val):
            return NotImplemented
        gate = val.gate if isinstance(val, ops.Operation) else val
        axes = self.get_axes(qubits)
        if isinstance(gate, common_gates.XPowGate):
            self._x(gate, axes[0])
        elif isinstance(gate, common_gates.YPowGate):
            self._y(gate, axes[0])
        elif isinstance(gate, common_gates.ZPowGate):
            self._z(gate, axes[0])
        elif isinstance(gate, common_gates.HPowGate):
            self._h(gate, axes[0])
        elif isinstance(gate, common_gates.CXPowGate):
            self._cx(gate, axes[0], axes[1])
        elif isinstance(gate, common_gates.CZPowGate):
            self._cz(gate, axes[0], axes[1])
        else:
            return NotImplemented
        return True

    def _strat_apply_mixture_to_tableau(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
        mixture = protocols.mixture(val, None)
        if mixture is None:
            return False
        rand = self.prng.random()
        psum = 0.0
        for p, mix in mixture:
            psum += p
            if psum >= rand:
                return self._strat_act_on_clifford_tableau_from_single_qubit_decompose(
                    matrix_gates.MatrixGate(mix), qubits
                )
        raise AssertionError("Probablities don't add to 1")

    def _strat_act_on_clifford_tableau_from_single_qubit_decompose(
        self, val: Any, qubits: Sequence['cirq.Qid']
    ) -> bool:
        if num_qubits(val) == 1:
            if not has_unitary(val):
                return NotImplemented
            u = unitary(val)
            clifford_gate = SingleQubitCliffordGate.from_unitary(u)
            if clifford_gate is not None:
                for gate, quarter_turns in clifford_gate.decompose_rotation():
                    self._strat_apply_to_tableau(gate ** (quarter_turns / 2), qubits)
                return True

        return NotImplemented
