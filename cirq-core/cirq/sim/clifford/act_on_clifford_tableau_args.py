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

from typing import Any, Dict, TYPE_CHECKING, List, Sequence, Iterable, Union

import numpy as np

from cirq._compat import deprecated_parameter
from cirq.ops import common_gates
from cirq.ops import pauli_gates
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.protocols import has_unitary, num_qubits, unitary
from cirq.qis.clifford_tableau import CliffordTableau
from cirq.sim.act_on_args import ActOnArgs
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


def _rewrite_deprecated_args(args, kwargs):
    if len(args) > 2:
        kwargs['axes'] = args[2]
    if len(args) > 3:
        kwargs['prng'] = args[3]
    if len(args) > 4:
        kwargs['log_of_measurement_results'] = args[4]
    if len(args) > 5:
        kwargs['qubits'] = args[5]
    return args[:2], kwargs


class ActOnCliffordTableauArgs(ActOnArgs):
    """State and context for an operation acting on a clifford tableau.

    To act on this object, directly edit the `tableau` property, which is
    storing the density matrix of the quantum system with one axis per qubit.
    """

    @deprecated_parameter(
        deadline='v0.13',
        fix='No longer needed. `protocols.act_on` infers axes.',
        parameter_desc='axes',
        match=lambda args, kwargs: 'axes' in kwargs
        or ('prng' in kwargs and len(args) == 3)
        or (len(args) > 3 and isinstance(args[3], np.random.RandomState)),
        rewrite=_rewrite_deprecated_args,
    )
    def __init__(
        self,
        tableau: CliffordTableau,
        prng: np.random.RandomState,
        log_of_measurement_results: Dict[str, Any],
        qubits: Sequence['cirq.Qid'] = None,
        axes: Iterable[int] = None,
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
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
        """
        super().__init__(prng, qubits, axes, log_of_measurement_results)
        self.tableau = tableau

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> Union[bool, NotImplementedType]:
        strats = []
        if allow_decompose:
            strats.append(_strat_act_on_clifford_tableau_from_single_qubit_decompose)
        for strat in strats:
            result = strat(action, self, qubits)
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


def _strat_act_on_clifford_tableau_from_single_qubit_decompose(
    val: Any, args: 'cirq.ActOnCliffordTableauArgs', qubits: Sequence['cirq.Qid']
) -> bool:
    if num_qubits(val) == 1:
        if not has_unitary(val):
            return NotImplemented
        u = unitary(val)
        clifford_gate = SingleQubitCliffordGate.from_unitary(u)
        if clifford_gate is not None:
            for axis, quarter_turns in clifford_gate.decompose_rotation():
                if axis == pauli_gates.X:
                    common_gates.XPowGate(exponent=quarter_turns / 2)._act_on_(args, qubits)
                elif axis == pauli_gates.Y:
                    common_gates.YPowGate(exponent=quarter_turns / 2)._act_on_(args, qubits)
                else:
                    assert axis == pauli_gates.Z
                    common_gates.ZPowGate(exponent=quarter_turns / 2)._act_on_(args, qubits)
            return True

    return NotImplemented
