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

import abc
from typing import Any, Sequence, TYPE_CHECKING, Union

import numpy as np

from cirq import ops, protocols, linalg
from cirq.ops import common_gates, global_phase_op, matrix_gates, swap_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.act_on_args import ActOnArgs
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


class ActOnStabilizerArgs(ActOnArgs, metaclass=abc.ABCMeta):
    """Abstract wrapper around a stabilizer state for the act_on protocol."""

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> Union[bool, NotImplementedType]:
        strats = [
            self._strat_apply_gate,
            self._strat_apply_mixture,
        ]
        if allow_decompose:
            strats.append(self._strat_decompose)
            strats.append(self._strat_act_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, qubits)  # type: ignore
            if result is True:
                return True
            assert result is NotImplemented, str(result)

        return NotImplemented

    @abc.abstractmethod
    def _x(self, g: common_gates.XPowGate, axis: int):
        """Apply an X gate"""

    @abc.abstractmethod
    def _y(self, g: common_gates.YPowGate, axis: int):
        """Apply a Y gate"""

    @abc.abstractmethod
    def _z(self, g: common_gates.ZPowGate, axis: int):
        """Apply a Z gate"""

    @abc.abstractmethod
    def _h(self, g: common_gates.HPowGate, axis: int):
        """Apply an H gate"""

    @abc.abstractmethod
    def _cz(self, g: common_gates.CZPowGate, control_axis: int, target_axis: int):
        """Apply a CZ gate"""

    @abc.abstractmethod
    def _cx(self, g: common_gates.CXPowGate, control_axis: int, target_axis: int):
        """Apply a CX gate"""

    @abc.abstractmethod
    def _global_phase(self, g: global_phase_op.GlobalPhaseGate):
        """Apply global phase"""

    def _swap(self, g: swap_gates.SwapPowGate, control_axis: int, target_axis: int):
        """Apply a SWAP gate"""
        if g.exponent % 1 != 0:
            raise ValueError('Swap exponent must be integer')  # coverage: ignore
        self._cx(common_gates.CX, control_axis, target_axis)
        self._cx(
            common_gates.CXPowGate(exponent=g.exponent, global_shift=g.global_shift),
            target_axis,
            control_axis,
        )
        self._cx(common_gates.CX, control_axis, target_axis)

    def _strat_apply_gate(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
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
        elif isinstance(gate, global_phase_op.GlobalPhaseGate):
            self._global_phase(gate)
        elif isinstance(gate, swap_gates.SwapPowGate):
            self._swap(gate, axes[0], axes[1])
        else:
            return NotImplemented
        return True

    def _strat_apply_mixture(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
        mixture = protocols.mixture(val, None)
        if mixture is None:
            return NotImplemented
        if not all(linalg.is_unitary(m) for _, m in mixture):
            return NotImplemented
        probabilities, unitaries = zip(*mixture)
        index = self.prng.choice(len(unitaries), p=probabilities)
        return self._strat_act_from_single_qubit_decompose(
            matrix_gates.MatrixGate(unitaries[index]), qubits
        )

    def _strat_act_from_single_qubit_decompose(
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
                    gate = axis ** (quarter_turns / 2)
                    self._strat_apply_gate(gate, qubits)
                    final_unitary = np.matmul(unitary(gate), final_unitary)

                # Find the entry with the largest magnitude in the input unitary.
                k = max(np.ndindex(*u.shape), key=lambda t: abs(u[t]))
                # Correct the global phase that wasn't conserved in the above
                # decomposition.
                self._global_phase(global_phase_op.GlobalPhaseGate(u[k] / final_unitary[k]))
                return True

        return NotImplemented

    def _strat_decompose(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
        gate = val.gate if isinstance(val, ops.Operation) else val
        operations = protocols.decompose_once_with_qubits(gate, qubits, None)
        if operations is None or not all(protocols.has_stabilizer_effect(op) for op in operations):
            return NotImplemented
        for op in operations:
            protocols.act_on(op, self)
        return True
