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

from typing import Any, Dict, TYPE_CHECKING, List, Sequence

import numpy as np

from cirq.ops import common_gates
from cirq.ops import global_phase_op
from cirq.sim.clifford.act_on_stabilizer_args import ActOnStabilizerArgs

if TYPE_CHECKING:
    import cirq


class ActOnCliffordTableauArgs(ActOnStabilizerArgs):
    """State and context for an operation acting on a clifford tableau."""

    def __init__(
        self,
        tableau: 'cirq.CliffordTableau',
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

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Returns the measurement from the tableau."""
        return [self.tableau._measure(self.qubit_map[q], self.prng) for q in qubits]

    def _on_copy(self, target: 'ActOnCliffordTableauArgs', deep_copy_buffers: bool = True):
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
        if exponent % 0.5 != 0.0:
            raise ValueError('X exponent must be half integer')  # coverage: ignore
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
        if exponent % 0.5 != 0.0:
            raise ValueError('Y exponent must be half integer')  # coverage: ignore
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
        if exponent % 0.5 != 0.0:
            raise ValueError('Z exponent must be half integer')  # coverage: ignore
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
        if exponent % 1 != 0:
            raise ValueError('H exponent must be integer')  # coverage: ignore
        self._y(common_gates.YPowGate(exponent=0.5), axis)
        self._x(common_gates.XPowGate(), axis)

    def _cz(self, g: common_gates.CZPowGate, control_axis: int, target_axis: int):
        exponent = g.exponent
        if exponent % 2 == 0:
            return
        if exponent % 1 != 0:
            raise ValueError('CZ exponent must be integer')  # coverage: ignore
        tableau = self.tableau
        (tableau.xs[:, target_axis], tableau.zs[:, target_axis]) = (
            tableau.zs[:, target_axis].copy(),
            tableau.xs[:, target_axis].copy(),
        )
        tableau.rs[:] ^= tableau.xs[:, target_axis] & tableau.zs[:, target_axis]
        tableau.rs[:] ^= (
            tableau.xs[:, control_axis]
            & tableau.zs[:, target_axis]
            & (~(tableau.xs[:, target_axis] ^ tableau.zs[:, control_axis]))
        )
        tableau.xs[:, target_axis] ^= tableau.xs[:, control_axis]
        tableau.zs[:, control_axis] ^= tableau.zs[:, target_axis]
        (tableau.xs[:, target_axis], tableau.zs[:, target_axis]) = (
            tableau.zs[:, target_axis].copy(),
            tableau.xs[:, target_axis].copy(),
        )
        tableau.rs[:] ^= tableau.xs[:, target_axis] & tableau.zs[:, target_axis]

    def _cx(self, g: common_gates.CXPowGate, control_axis: int, target_axis: int):
        exponent = g.exponent
        if exponent % 2 == 0:
            return
        if exponent % 1 != 0:
            raise ValueError('CX exponent must be integer')  # coverage: ignore
        tableau = self.tableau
        tableau.rs[:] ^= (
            tableau.xs[:, control_axis]
            & tableau.zs[:, target_axis]
            & (~(tableau.xs[:, target_axis] ^ tableau.zs[:, control_axis]))
        )
        tableau.xs[:, target_axis] ^= tableau.xs[:, control_axis]
        tableau.zs[:, control_axis] ^= tableau.zs[:, target_axis]

    def _global_phase(self, g: global_phase_op.GlobalPhaseGate):
        pass
