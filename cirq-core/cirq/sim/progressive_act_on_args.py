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
import enum
from typing import List, Sequence, Union, TypeVar, Generic, TYPE_CHECKING

import numpy as np

from cirq import sim, ops, protocols, qis

if TYPE_CHECKING:
    import cirq


class PureActOnArgs(sim.ActOnArgs):
    def __init__(self, initial_state: Union[int, Sequence[bool]], prng, qubits, logs):
        super().__init__(
            prng=prng,
            qubits=qubits,
            log_of_measurement_results=logs,
        )
        self.b = list(
            initial_state
            if isinstance(initial_state, Sequence)
            else reversed([bool(initial_state & (1 << n)) for n in range(len(qubits))])
        )

    def as_int(self):
        return sum(1 << i for i, v in enumerate(reversed(self.b)) if v)

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        return [1 if self.b[i] else 0 for i in self.get_axes(qubits)]

    def _on_copy(self, target: 'PureActOnArgs'):
        target.b = self.b.copy()

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        gate = action if isinstance(action, ops.Gate) else action.gate
        qubits = action.qubits if isinstance(action, ops.Operation) else qubits
        if isinstance(gate, ops.XPowGate) and gate.exponent % 2 == 1:
            for i in self.get_axes(qubits):
                self.b[i] = not self.b[i]
            return True
        if isinstance(gate, ops.CXPowGate) and gate.exponent % 2 == 1:
            axes = self.get_axes(qubits)
            self.b[axes[1]] ^= self.b[axes[0]]
            return True
        if isinstance(gate, ops.EigenGate) and gate.exponent % 2 == 0:
            return True
        elif isinstance(gate, ops.ResetChannel):
            for i in self.get_axes(qubits):
                self.b[i] = False
            return True
        return False

    def sample(self, qubits, repetitions=1, seed=None):
        measurements: List[List[int]] = []
        for _ in range(repetitions):
            measurements.append(self._perform_measurement(qubits))

        return np.array(measurements, dtype=int)

    def can_factor(self, qubits: Sequence['cirq.Qid']):
        return True

    def _on_factor(
        self,
        qubits: Sequence['cirq.Qid'],
        extracted: 'PureActOnArgs',
        remainder: 'PureActOnArgs',
        validate=True,
        atol=1e-07,
    ):
        axes = self.get_axes(qubits)
        extracted.b = [self.b[i] for i in axes]
        remainder.b = [self.b[i] for i in range(len(self.qubits)) if i not in axes]

    def _on_kronecker_product(self, other: 'PureActOnArgs', target: 'PureActOnArgs'):
        target.b = list(self.b) + list(other.b)

    def _on_transpose_to_qubit_order(self, qubits: Sequence['cirq.Qid'], target: 'PureActOnArgs'):
        target.b = [self.b[i] for i in self.get_axes(qubits)]


TActOnArgs = TypeVar('TActOnArgs', bound='cirq.ActOnArgs')


class Progression(enum.IntEnum):
    PURE = 1
    CH_FORM = 2
    STATE_VECTOR = 3
    DENSITY_MATRIX = 4


def _upgrade(args: sim.ActOnArgs, to: Progression) -> sim.ActOnArgs:
    if isinstance(args, PureActOnArgs) and to > Progression.PURE:
        ch = sim.StabilizerStateChForm(len(args.qubits), args.as_int())
        args = sim.ActOnStabilizerCHFormArgs(
            ch, args.prng, args.log_of_measurement_results, args.qubits
        )
    if isinstance(args, sim.ActOnStabilizerCHFormArgs) and to > Progression.CH_FORM:
        sv = args.state.state_vector().reshape([q.dimension for q in args.qubits])
        args = sim.ActOnStateVectorArgs(
            sv, np.empty_like(sv), args.prng, args.log_of_measurement_results, args.qubits
        )
    if isinstance(args, sim.ActOnStateVectorArgs) and to > Progression.STATE_VECTOR:
        dm = qis.density_matrix_from_state_vector(args.target_tensor).reshape(
            [q.dimension for q in args.qubits] * 2
        )
        args = sim.ActOnDensityMatrixArgs(
            dm,
            [np.empty_like(dm) for _ in range(3)],
            tuple(q.dimension for q in args.qubits),
            args.prng,
            args.log_of_measurement_results,
            args.qubits,
        )
    return args


def _progression(args: sim.ActOnArgs) -> Progression:
    if isinstance(args, PureActOnArgs):
        return Progression.PURE
    if isinstance(args, sim.ActOnStabilizerCHFormArgs):
        return Progression.CH_FORM
    if isinstance(args, sim.ActOnStateVectorArgs):
        return Progression.STATE_VECTOR
    return Progression.DENSITY_MATRIX


class ProgressiveActOnArgs(Generic[TActOnArgs], sim.ActOnArgs[TActOnArgs]):
    def __init__(
        self,
        args: 'cirq.ActOnArgs',
    ):
        initial_progression = _progression(args)
        super().__init__(
            prng=args.prng,
            qubits=args.qubits,
            log_of_measurement_results=args.log_of_measurement_results,
        )
        self.args = args
        self._run_progressively = initial_progression == Progression.PURE

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        return self.args._perform_measurement(qubits)

    def _on_copy(self, target):
        target.args = self.args.copy()
        target.args._log_of_measurement_results = self._log_of_measurement_results

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        if isinstance(action, ops.Operation):
            qubits = None  # type: ignore
        if isinstance(self.args, PureActOnArgs):
            if self.args._act_on_fallback_(action, qubits, allow_decompose=allow_decompose):
                return True
            self.args = _upgrade(self.args, Progression.CH_FORM)
        if isinstance(self.args, sim.ActOnStabilizerCHFormArgs):
            if protocols.has_stabilizer_effect(action) and (
                protocols.has_unitary(action) or protocols.is_measurement(action)
            ):
                protocols.act_on(action, self.args, qubits, allow_decompose=allow_decompose)
                return True
            self.args = _upgrade(self.args, Progression.STATE_VECTOR)
        if isinstance(self.args, sim.ActOnStateVectorArgs):
            if protocols.has_unitary(action) or protocols.is_measurement(action):
                protocols.act_on(action, self.args, qubits, allow_decompose=allow_decompose)
                return True
            self.args = _upgrade(self.args, Progression.DENSITY_MATRIX)
        protocols.act_on(action, self.args, qubits, allow_decompose=allow_decompose)
        return True

    def sample(self, qubits, repetitions=1, seed=None):
        return self.args.sample(qubits, repetitions, seed)

    def _on_swap(self, q1: 'cirq.Qid', q2: 'cirq.Qid', target):
        target.args = self.args.swap(q1, q2)

    def _on_rename(self, q1: 'cirq.Qid', q2: 'cirq.Qid', target):
        target.args = self.args.rename(q1, q2)

    def _on_kronecker_product(self, other, target):
        self_progression = _progression(self.args)
        other_progression = _progression(other.args)
        progression = max(self_progression, other_progression)
        self_args = _upgrade(self.args, progression)
        other_args = _upgrade(other.args, progression)
        target.args = self_args.kronecker_product(other_args, inplace=self is target)

    def can_factor(self, q):
        return self.args.can_factor(q)

    def _on_factor(self, qubits, extracted, remainder, validate=False, atol=1e-07):
        extracted.args, remainder.args = self.args.factor(qubits, validate=validate)
        if self._run_progressively and len(qubits) == 1:
            state = [x != 0 for x in extracted.args._perform_measurement(qubits)]
            extracted.args = PureActOnArgs(
                state, self.prng, extracted.args.qubits, extracted.args.log_of_measurement_results
            )

    def _on_transpose_to_qubit_order(self, qubits: Sequence['cirq.Qid'], target):
        target.args = self.args.transpose_to_qubit_order(qubits)

    def create_merged_state(self):
        args = _upgrade(self.args, Progression.DENSITY_MATRIX)
        return args.create_merged_state()
