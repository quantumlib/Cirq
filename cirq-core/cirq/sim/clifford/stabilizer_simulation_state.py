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
from typing import Any, cast, Generic, Optional, Sequence, TYPE_CHECKING, TypeVar, Union

import numpy as np
import sympy

from cirq import linalg, ops, protocols
from cirq.ops import common_gates, global_phase_op, matrix_gates, swap_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.simulation_state import SimulationState
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


TStabilizerState = TypeVar('TStabilizerState', bound='cirq.StabilizerState')


class StabilizerSimulationState(
    SimulationState[TStabilizerState], Generic[TStabilizerState], metaclass=abc.ABCMeta
):
    """Abstract wrapper around a stabilizer state for the act_on protocol."""

    def __init__(
        self,
        *,
        state: TStabilizerState,
        prng: Optional[np.random.RandomState] = None,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        classical_data: Optional['cirq.ClassicalDataStore'] = None,
    ):
        """Initializes the StabilizerSimulationState.

        Args:
            state: The quantum stabilizer state to use in the simulation or
                act_on invocation.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            classical_data: The shared classical data container for this
                simulation.
        """
        super().__init__(state=state, prng=prng, qubits=qubits, classical_data=classical_data)

    @property
    def state(self) -> TStabilizerState:
        return self._state

    def _act_on_fallback_(
        self, action: Any, qubits: Sequence['cirq.Qid'], allow_decompose: bool = True
    ) -> Union[bool, NotImplementedType]:
        strats = [self._strat_apply_gate, self._strat_apply_mixture]
        if allow_decompose:
            strats.append(self._strat_decompose)
            strats.append(self._strat_act_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, qubits)
            if result is True:
                return True
            assert result is NotImplemented, str(result)

        return NotImplemented

    def _swap(
        self, control_axis: int, target_axis: int, exponent: float = 1, global_shift: float = 0
    ):
        """Apply a SWAP gate"""
        if exponent % 1 != 0:
            raise ValueError('Swap exponent must be integer')  # pragma: no cover
        self._state.apply_cx(control_axis, target_axis)
        self._state.apply_cx(target_axis, control_axis, exponent, global_shift)
        self._state.apply_cx(control_axis, target_axis)

    def _strat_apply_gate(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
        if not protocols.has_stabilizer_effect(val):
            return NotImplemented
        gate = val.gate if isinstance(val, ops.Operation) else val
        axes = self.get_axes(qubits)
        exponent = cast(float, getattr(gate, 'exponent', None))
        if isinstance(gate, common_gates.XPowGate):
            self._state.apply_x(axes[0], exponent, gate.global_shift)
        elif isinstance(gate, common_gates.YPowGate):
            self._state.apply_y(axes[0], exponent, gate.global_shift)
        elif isinstance(gate, common_gates.ZPowGate):
            self._state.apply_z(axes[0], exponent, gate.global_shift)
        elif isinstance(gate, common_gates.HPowGate):
            self._state.apply_h(axes[0], exponent, gate.global_shift)
        elif isinstance(gate, common_gates.CXPowGate):
            self._state.apply_cx(axes[0], axes[1], exponent, gate.global_shift)
        elif isinstance(gate, common_gates.CZPowGate):
            self._state.apply_cz(axes[0], axes[1], exponent, gate.global_shift)
        elif isinstance(gate, global_phase_op.GlobalPhaseGate):
            if isinstance(gate.coefficient, sympy.Expr):
                return NotImplemented
            self._state.apply_global_phase(gate.coefficient)
        elif isinstance(gate, swap_gates.SwapPowGate):
            self._swap(axes[0], axes[1], exponent, gate.global_shift)
        else:
            return NotImplemented
        return True

    def _strat_apply_mixture(self, val: Any, qubits: Sequence['cirq.Qid']) -> bool:
        mixture = protocols.mixture(val, None)
        if mixture is None:
            return NotImplemented
        if not all(linalg.is_unitary(m) for _, m in mixture):
            return NotImplemented  # pragma: no cover
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
            gate_and_phase = SingleQubitCliffordGate.from_unitary_with_global_phase(u)
            if gate_and_phase is not None:
                clifford_gate, global_phase = gate_and_phase
                # Apply gates.
                for gate in clifford_gate.decompose_gate():
                    self._strat_apply_gate(gate, qubits)
                # Apply global phase.
                self._state.apply_global_phase(global_phase)
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
