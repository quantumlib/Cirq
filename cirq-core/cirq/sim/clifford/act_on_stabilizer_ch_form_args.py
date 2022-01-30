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

from typing import Any, Dict, TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

from cirq import _compat, value, ops, protocols
from cirq.ops import common_gates, pauli_gates
from cirq.ops import global_phase_op
from cirq.sim.clifford import clifford_simulator
from cirq.sim.clifford.act_on_stabilizer_args import ActOnStabilizerArgs

if TYPE_CHECKING:
    import cirq


class ActOnStabilizerCHFormArgs(ActOnStabilizerArgs):
    """Wrapper around a stabilizer state in CH form for the act_on protocol."""

    @_compat.deprecated_parameter(
        deadline='v0.15',
        fix='Specify all the arguments with keywords, use initial_state instead of state.',
        parameter_desc='positional arguments',
        match=lambda args, kwargs: len(args) != 1 or 'state' in kwargs,
    )
    def __init__(
        self,
        state: Optional['cirq.StabilizerStateChForm'] = None,
        prng: Optional[np.random.RandomState] = None,
        log_of_measurement_results: Optional[Dict[str, Any]] = None,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        initial_state: Union[int, 'cirq.StabilizerStateChForm'] = 0,
    ):
        """Initializes with the given state and the axes for the operation.

        Args:
            state: The StabilizerStateChForm to act on. Operations are expected
                to perform inplace edits of this object.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
            initial_state: The initial state for the simulation. This can be a
                full CH form passed by reference which will be modified inplace,
                or a big-endian int in the computational basis.
        """
        super().__init__(prng, qubits, log_of_measurement_results)
        initial_state = state or initial_state
        if isinstance(initial_state, int):
            qubit_map = {q: i for i, q in enumerate(self.qubits)}
            initial_state = clifford_simulator.CliffordState(
                qubit_map, initial_state=initial_state
            ).ch_form
        self.state = initial_state

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Returns the measurement from the stabilizer state form."""
        return [self.state._measure(self.qubit_map[q], self.prng) for q in qubits]

    def _on_copy(self, target: 'ActOnStabilizerCHFormArgs', deep_copy_buffers: bool = True):
        target.state = self.state.copy()

    def _on_kronecker_product(
        self, other: 'cirq.ActOnStabilizerCHFormArgs', target: 'cirq.ActOnStabilizerCHFormArgs'
    ):
        target.state = self.state.kron(other.state)

    def _on_transpose_to_qubit_order(
        self, qubits: Sequence['cirq.Qid'], target: 'cirq.ActOnStabilizerCHFormArgs'
    ):
        axes = [self.qubit_map[q] for q in qubits]
        target.state = self.state.reindex(axes)

    def sample(
        self,
        qubits: Sequence['cirq.Qid'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        measurements: Dict[str, List[np.ndarray]] = {}
        prng = value.parse_random_state(seed)
        for i in range(repetitions):
            op = ops.measure(*qubits, key=str(i))
            state = self.state.copy()
            ch_form_args = ActOnStabilizerCHFormArgs(
                prng=prng,
                log_of_measurement_results=measurements,
                qubits=self.qubits,
                initial_state=state,
            )
            protocols.act_on(op, ch_form_args)
        return np.array(list(measurements.values()), dtype=bool)

    def _x(self, g: common_gates.XPowGate, axis: int):
        exponent = g.exponent
        if exponent % 2 != 0:
            if exponent % 0.5 != 0.0:
                raise ValueError('Y exponent must be half integer')  # coverage: ignore
            self._h(common_gates.H, axis)
            self._z(common_gates.ZPowGate(exponent=exponent), axis)
            self._h(common_gates.H, axis)
        self.state.omega *= _phase(g)

    def _y(self, g: common_gates.YPowGate, axis: int):
        exponent = g.exponent
        if exponent % 0.5 != 0.0:
            raise ValueError('Y exponent must be half integer')  # coverage: ignore
        if exponent % 2 == 0:
            self.state.omega *= _phase(g)
        elif exponent % 2 == 0.5:
            self._z(pauli_gates.Z, axis)
            self._h(common_gates.H, axis)
            self.state.omega *= _phase(g) * (1 + 1j) / (2 ** 0.5)
        elif exponent % 2 == 1:
            self._z(pauli_gates.Z, axis)
            self._h(common_gates.H, axis)
            self._z(pauli_gates.Z, axis)
            self._h(common_gates.H, axis)
            self.state.omega *= _phase(g) * 1j
        elif exponent % 2 == 1.5:
            self._h(common_gates.H, axis)
            self._z(pauli_gates.Z, axis)
            self.state.omega *= _phase(g) * (1 - 1j) / (2 ** 0.5)

    def _z(self, g: common_gates.ZPowGate, axis: int):
        exponent = g.exponent
        state = self.state
        if exponent % 2 != 0:
            if exponent % 0.5 != 0.0:
                raise ValueError('Z exponent must be half integer')  # coverage: ignore
            effective_exponent = exponent % 2
            for _ in range(int(effective_exponent * 2)):
                # Prescription for S left multiplication.
                # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
                state.M[axis, :] ^= state.G[axis, :]
                state.gamma[axis] = (state.gamma[axis] - 1) % 4
        state.omega *= _phase(g)

    def _h(self, g: common_gates.HPowGate, axis: int):
        exponent = g.exponent
        state = self.state
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('H exponent must be integer')  # coverage: ignore
            # Prescription for H left multiplication
            # Reference: https://arxiv.org/abs/1808.00128
            # Equations 48, 49 and Proposition 4
            t = state.s ^ (state.G[axis, :] & state.v)
            u = state.s ^ (state.F[axis, :] & (~state.v)) ^ (state.M[axis, :] & state.v)
            alpha = sum(state.G[axis, :] & (~state.v) & state.s) % 2
            beta = sum(state.M[axis, :] & (~state.v) & state.s)
            beta += sum(state.F[axis, :] & state.v & state.M[axis, :])
            beta += sum(state.F[axis, :] & state.v & state.s)
            beta %= 2
            delta = (state.gamma[axis] + 2 * (alpha + beta)) % 4
            state.update_sum(t, u, delta=delta, alpha=alpha)
        state.omega *= _phase(g)

    def _cz(self, g: common_gates.CZPowGate, control_axis: int, target_axis: int):
        exponent = g.exponent
        state = self.state
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('CZ exponent must be integer')  # coverage: ignore
            # Prescription for CZ left multiplication.
            # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
            state.M[control_axis, :] ^= state.G[target_axis, :]
            state.M[target_axis, :] ^= state.G[control_axis, :]
        state.omega *= _phase(g)

    def _cx(self, g: common_gates.CXPowGate, control_axis: int, target_axis: int):
        exponent = g.exponent
        state = self.state
        if exponent % 2 != 0:
            if exponent % 1 != 0:
                raise ValueError('CX exponent must be integer')  # coverage: ignore
            # Prescription for CX left multiplication.
            # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
            state.gamma[control_axis] = (
                state.gamma[control_axis]
                + state.gamma[target_axis]
                + 2 * (sum(state.M[control_axis, :] & state.F[target_axis, :]) % 2)
            ) % 4
            state.G[target_axis, :] ^= state.G[control_axis, :]
            state.F[control_axis, :] ^= state.F[target_axis, :]
            state.M[control_axis, :] ^= state.M[target_axis, :]
        state.omega *= _phase(g)

    def _global_phase(self, g: global_phase_op.GlobalPhaseGate):
        self.state.omega *= g.coefficient


def _phase(gate):
    return np.exp(1j * np.pi * gate.global_shift * gate.exponent)
