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

from typing import Any, Dict, Iterable, TYPE_CHECKING

import numpy as np

from cirq.ops import common_gates
from cirq.ops import pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.sim.clifford.clifford_tableau import CliffordTableau

if TYPE_CHECKING:
    import cirq


class ActOnCliffordTableauArgs:
    """State and context for an operation acting on a clifford tableau.
    There are two common ways to act on this object:
    1. Directly edit the `tableau` property, which is storing the clifford
        tableau of the quantum system with one axis per qubit.
    2. Call `record_measurement_result(key, val)` to log a measurement result.
    """

    def __init__(self, tableau: CliffordTableau, axes: Iterable[int],
                 prng: np.random.RandomState,
                 log_of_measurement_results: Dict[str, Any]):
        """
        Args:
            tableau: The CliffordTableau to act on. Operations are expected to
                perform inplace edits of this object.
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into. Edit it easily by calling
                `ActOnCliffordTableauArgs.record_measurement_result`.
        """
        self.tableau = tableau
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
            strats.append(
                _strat_act_on_clifford_tableau_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, self)
            if result is False:
                break  # coverage: ignore
            if result is True:
                return True
            assert result is NotImplemented, str(result)

        return NotImplemented


def _strat_act_on_clifford_tableau_from_single_qubit_decompose(
        val: Any, args: 'cirq.ActOnCliffordTableauArgs') -> bool:
    if num_qubits(val) == 1:
        if not has_unitary(val):
            return NotImplemented
        u = unitary(val)
        clifford_gate = SingleQubitCliffordGate.from_unitary(u)
        if clifford_gate is not None:
            for axis, quarter_turns in clifford_gate.decompose_rotation():
                if axis == pauli_gates.X:
                    common_gates.XPowGate(exponent=quarter_turns /
                                          2)._act_on_(args)
                elif axis == pauli_gates.Y:
                    common_gates.YPowGate(exponent=quarter_turns /
                                          2)._act_on_(args)
                else:
                    assert axis == pauli_gates.Z
                    common_gates.ZPowGate(exponent=quarter_turns /
                                          2)._act_on_(args)
            return True

    return NotImplemented
