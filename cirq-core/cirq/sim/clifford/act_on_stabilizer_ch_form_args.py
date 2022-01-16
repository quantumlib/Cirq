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

from typing import Any, Dict, List, Optional, Sequence, Union
from cirq.sim.clifford.clifford_simulator import CliffordState

import numpy as np

import cirq
from cirq import value, ops, protocols
from cirq.ops import common_gates, pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.act_on_args import ActOnArgs
from cirq.type_workarounds import NotImplementedType


class ActOnStabilizerCHFormArgs(ActOnArgs):
    """Wrapper around a stabilizer state in CH form for the act_on protocol.

    To act on this object, directly edit the `state` property, which is
    storing the stabilizer state of the quantum system with one axis per qubit.
    """

    def __init__(
        self,
        state: Optional['cirq.StabilizerStateChForm'] = None,
        initial_state: int = 0,
        prng: Optional[np.random.RandomState] = None,
        log_of_measurement_results: Optional[Dict[str, Any]] = None,
        qubits: Sequence['cirq.Qid'] = None,
    ):
        """Initializes with the given state and the axes for the operation.
        Args:
            state: The StabilizerStateChForm to act on. Operations are expected
                to perform inplace edits of this object.
            initial_state: The initial state for the simulation in the
                computational basis. Only used when `state` is None.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
        """
        super().__init__(prng, qubits, log_of_measurement_results)
        if state is None:
            qubit_map = {q: i for i, q in enumerate(self.qubits)}
            self.state = CliffordState(qubit_map, initial_state=initial_state).ch_form
        else:
            self.state = state

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> Union[bool, NotImplementedType]:
        strats = []
        if allow_decompose:
            strats.append(_strat_act_on_stabilizer_ch_form_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, self, qubits)
            if result is True:
                return True
            assert result is NotImplemented, str(result)

        return NotImplemented

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Returns the measurement from the stabilizer state form."""
        return [self.state._measure(self.qubit_map[q], self.prng) for q in qubits]

    def _on_copy(self, target: 'ActOnStabilizerCHFormArgs', deep_copy_buffers: bool = True):
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
            ch_form_args = ActOnStabilizerCHFormArgs(
                state=state, prng=prng, log_of_measurement_results=measurements, qubits=self.qubits
            )
            protocols.act_on(op, ch_form_args)
        return np.array(list(measurements.values()), dtype=bool)


def _strat_act_on_stabilizer_ch_form_from_single_qubit_decompose(
    val: Any, args: 'cirq.ActOnStabilizerCHFormArgs', qubits: Sequence['cirq.Qid']
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
                gate: Optional[cirq.Gate] = None
                if axis == pauli_gates.X:
                    gate = common_gates.XPowGate(exponent=quarter_turns / 2)
                    assert gate._act_on_(args, qubits)
                elif axis == pauli_gates.Y:
                    gate = common_gates.YPowGate(exponent=quarter_turns / 2)
                    assert gate._act_on_(args, qubits)
                else:
                    assert axis == pauli_gates.Z
                    gate = common_gates.ZPowGate(exponent=quarter_turns / 2)
                    assert gate._act_on_(args, qubits)

                final_unitary = np.matmul(unitary(gate), final_unitary)

            # Find the entry with the largest magnitude in the input unitary.
            k = max(np.ndindex(*u.shape), key=lambda t: abs(u[t]))
            # Correct the global phase that wasn't conserved in the above
            # decomposition.
            args.state.omega *= u[k] / final_unitary[k]
            return True

    return NotImplemented
