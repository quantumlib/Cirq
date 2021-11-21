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

from typing import Any, Dict, TYPE_CHECKING, List, Sequence, Union

import numpy as np

from cirq import value, ops, protocols
from cirq.ops import common_gates, pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.act_on_args import ActOnArgs
from cirq.sim.clifford.stabilizer_state_ch_form import StabilizerStateChForm
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq
    from typing import Optional


class ActOnStabilizerCHFormArgs(ActOnArgs):
    """Wrapper around a stabilizer state in CH form for the act_on protocol.

    To act on this object, directly edit the `state` property, which is
    storing the stabilizer state of the quantum system with one axis per qubit.
    """

    def __init__(
        self,
        state: StabilizerStateChForm,
        prng: np.random.RandomState,
        log_of_measurement_results: Dict[str, Any],
        qubits: Sequence['cirq.Qid'] = None,
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
        """
        super().__init__(prng, qubits, log_of_measurement_results)
        self.state = state

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> Union[bool, NotImplementedType]:
        strats = [self._strat_apply_to_ch_form]
        if allow_decompose:
            strats.append(self._strat_act_on_stabilizer_ch_form_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, qubits)
            if result is True:
                return True
            assert result is NotImplemented, str(result)

        return NotImplemented

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Returns the measurement from the stabilizer state form."""
        return [self.state._measure(self.qubit_map[q], self.prng) for q in qubits]

    def _on_copy(self, target: 'ActOnStabilizerCHFormArgs'):
        target.state = self.state.copy()

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
            ch_form_args = ActOnStabilizerCHFormArgs(state, prng, measurements, self.qubits)
            protocols.act_on(op, ch_form_args)
        return np.array(list(measurements.values()), dtype=bool)

    def _x(self, exponent: float, axis: int, phase: complex):
        self._h(1, axis, 1)
        self._z(exponent, axis, 1)
        self._h(1, axis, 1)
        self.state.omega *= phase

    def _y(self, exponent: float, axis: int, phase: complex):
        if exponent == 0.5:
            phase *= (1 + 1j) / (2 ** 0.5)
            self._z(1, axis, 1)
            self._h(1, axis, 1)
            self.state.omega *= phase
        elif exponent == 1:
            phase *= 1j
            self._z(1, axis, 1)
            self._h(1, axis, 1)
            self._z(1, axis, 1)
            self._h(1, axis, 1)
            self.state.omega *= phase
        if exponent == 1.5:
            phase *= (1 - 1j) / (2 ** 0.5)
            self._h(1, axis, 1)
            self._z(1, axis, 1)
            self.state.omega *= phase

    def _z(self, exponent: float, axis: int, phase: complex):
        effective_exponent = exponent % 2
        state = self.state
        for _ in range(int(effective_exponent * 2)):
            # Prescription for S left multiplication.
            # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
            state.M[axis, :] ^= state.G[axis, :]
            state.gamma[axis] = (state.gamma[axis] - 1) % 4
        state.omega *= phase

    def _h(self, exponent: float, axis: int, phase: complex):
        state = self.state
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
        state.omega *= phase

    def _cz(self, exponent: float, axis1: int, axis2: int, phase: complex):
        assert exponent % 2 == 1
        state = self.state
        # Prescription for CZ left multiplication.
        # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
        state.M[axis1, :] ^= state.G[axis2, :]
        state.M[axis2, :] ^= state.G[axis1, :]
        state.omega *= phase

    def _cx(self, exponent: float, axis1: int, axis2: int, phase: complex):
        assert exponent % 2 == 1
        state = self.state
        # Prescription for CX left multiplication.
        # Reference: https://arxiv.org/abs/1808.00128 Proposition 4 end
        state.gamma[axis1] = (
            state.gamma[axis1]
            + state.gamma[axis2]
            + 2 * (sum(state.M[axis1, :] & state.F[axis2, :]) % 2)
        ) % 4
        state.G[axis2, :] ^= state.G[axis1, :]
        state.F[axis1, :] ^= state.F[axis2, :]
        state.M[axis1, :] ^= state.M[axis2, :]
        state.omega *= phase

    def _strat_apply_to_ch_form(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
        val = val.gate if isinstance(val, ops.Operation) else val
        paulis = protocols.as_paulis(val, self.prng)
        if paulis is NotImplemented:
            return NotImplemented
        paulis, phase = paulis
        for pauli, exponent, indexes in paulis:
            affected_qubits = [qubits[i] for i in indexes]
            axes = self.get_axes(affected_qubits)
            if pauli == 'X':
                self._x(exponent, axes[0], phase)
            elif pauli == 'Y':
                self._y(exponent, axes[0], phase)
            elif pauli == 'Z':
                self._z(exponent, axes[0], phase)
            elif pauli == 'H':
                self._h(exponent, axes[0], phase)
            elif pauli == 'CZ':
                self._cz(exponent, axes[0], axes[1], phase)
            elif pauli == 'CX':
                self._cx(exponent, axes[0], axes[1], phase)
            else:
                assert False
        return True

    def _strat_act_on_stabilizer_ch_form_from_single_qubit_decompose(
        self, val: Any, qubits: Sequence['cirq.Qid']
    ) -> bool:
        if num_qubits(val) == 1:
            if not has_unitary(val):
                return NotImplemented
            u = unitary(val)
            clifford_gate = SingleQubitCliffordGate.from_unitary(u)
            if clifford_gate is not None:
                # Gather the effective unitary applied so as to correct for the
                # global phase later.
                final_unitary = np.eye(2)
                for axis, quarter_turns in clifford_gate.decompose_rotation():
                    gate = None  # type: Optional[cirq.Gate]
                    if axis == pauli_gates.X:
                        gate = common_gates.XPowGate(exponent=quarter_turns / 2)
                    elif axis == pauli_gates.Y:
                        gate = common_gates.YPowGate(exponent=quarter_turns / 2)
                    elif axis == pauli_gates.Z:
                        gate = common_gates.ZPowGate(exponent=quarter_turns / 2)
                    assert gate is not None
                    self._strat_apply_to_ch_form(gate, qubits)

                    final_unitary = np.matmul(unitary(gate), final_unitary)

                # Find the entry with the largest magnitude in the input unitary.
                k = max(np.ndindex(*u.shape), key=lambda t: abs(u[t]))
                # Correct the global phase that wasn't conserved in the above
                # decomposition.
                self.state.omega *= u[k] / final_unitary[k]
                return True

        return NotImplemented
