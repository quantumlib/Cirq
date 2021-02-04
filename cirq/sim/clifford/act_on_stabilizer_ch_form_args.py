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

from typing import Any, Dict, Iterable, TYPE_CHECKING

import numpy as np

from cirq.ops import common_gates, pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.clifford.stabilizer_state_ch_form import StabilizerStateChForm

if TYPE_CHECKING:
    import cirq
    from typing import Optional


class ActOnStabilizerCHFormArgs:
    """Wrapper around a stabilizer state in CH form for the act_on protocol.

    To act on this object, directly edit the `state` property, which is
    storing the stabilizer state of the quantum system with one axis per qubit.
    """

    def __init__(
        self,
        state: StabilizerStateChForm,
        axes: Iterable[int],
        prng: np.random.RandomState,
        log_of_measurement_results: Dict[str, Any],
    ):
        """Initializes with the given state and the axes for the operation.
        Args:
            state: The StabilizerStateChForm to act on. Operations are expected
                to perform inplace edits of this object.
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into. Edit it easily by calling
                `ActOnStabilizerCHFormArgs.record_measurement_result`.
        """
        self.state = state
        self.axes = tuple(axes)
        self.prng = prng
        self.log_of_measurement_results = log_of_measurement_results

    def record_measurement_result(self, key: str, value: Any):
        """Adds a measurement result to the log.
        Args:
            key: The key the measurement result should be logged under. Note
                that operations should only store results under keys they have
                declared in a `_measurement_keys_` method.
            value: The value to log for the measurement.
        """
        if key in self.log_of_measurement_results:
            raise ValueError(f"Measurement already logged to key {key!r}")
        self.log_of_measurement_results[key] = value

    def _act_on_fallback_(self, action: Any, allow_decompose: bool):
        strats = []
        if allow_decompose:
            strats.append(_strat_act_on_stabilizer_ch_form_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, self)
            if result is True:
                return True
            assert result is NotImplemented, str(result)

        return NotImplemented


def _strat_act_on_stabilizer_ch_form_from_single_qubit_decompose(
    val: Any, args: 'cirq.ActOnStabilizerCHFormArgs'
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
                    assert gate._act_on_(args)
                elif axis == pauli_gates.Y:
                    gate = common_gates.YPowGate(exponent=quarter_turns / 2)
                    assert gate._act_on_(args)
                else:
                    assert axis == pauli_gates.Z
                    gate = common_gates.ZPowGate(exponent=quarter_turns / 2)
                    assert gate._act_on_(args)

                final_unitary = np.matmul(unitary(gate), final_unitary)

            # Find the entry with the largest magnitude in the input unitary.
            k = max(np.ndindex(*u.shape), key=lambda t: abs(u[t]))
            # Correct the global phase that wasn't conserved in the above
            # decomposition.
            args.state.omega *= u[k] / final_unitary[k]
            return True

    return NotImplemented
