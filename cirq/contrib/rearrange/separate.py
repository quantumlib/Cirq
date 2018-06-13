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

from cirq import ops, Circuit, Moment, InsertStrategy

from cirq.contrib.rearrange.axis import Axis
from cirq.contrib.rearrange.clifford_pauli_gate import CliffordPauliGate
from cirq.contrib.rearrange.interaction_gate import InteractionGate
from cirq.contrib.rearrange.pauli_string import PauliString
from cirq.contrib.rearrange.non_clifford_gate import NonCliffordGate


def _convert_from_regular_op(op, tolerance: float = 1e-5):
    # TODO: Support conversion in a more general way
    conversions = (  # All assumed to be EigenGate subclasses
        (ops.RotXGate, (0,)),
        (ops.RotYGate, (1,)),
        (ops.RotZGate, (2,)),
        (ops.Rot11Gate, (2, 2)),
        (ops.CNotGate, (2, 0)),
        (InteractionGate, '?'),
    )
    for cls, axes in conversions:
        if isinstance(op.gate, cls):
            if axes == '?':
                assert abs(op.gate.half_turns % 2.0 - 1.0) < tolerance
                yield InteractionGate(op.gate.axis0, op.gate.axis1)(*op.qubits)
            assert len(op.qubits) == len(axes)
            half_turns = op.gate._exponent  # TODO: Don't access private value
            if len(axes) == 1:
                qubit, = op.qubits
                axis = Axis(axes[0])
                if abs((half_turns + 1.0) % 2.0 - 1.0) < tolerance:
                    return  # No gate
                elif abs(half_turns % 2.0 - 1.0) < tolerance:
                    # Whole Pauli, half turn
                    yield CliffordPauliGate(axis, False)(qubit)
                elif abs(half_turns % 2.0 - 0.5) < tolerance:
                    # Half Pauli, quarter turn
                    yield CliffordPauliGate(axis, True)(qubit)
                elif abs(half_turns % 2.0 - 1.5) < tolerance:
                    # Reverse half Pauli, -quarter turn
                    yield CliffordPauliGate(axis.negative(), True)(qubit)
                else:
                    # Non-clifford rotation
                    yield NonCliffordGate.op_from_single(axis, half_turns, qubit)
                return
            elif len(axes) == 2:
                assert abs(half_turns % 2.0 - 1.0) < tolerance
                yield InteractionGate(Axis(axes[0]), Axis(axes[1]))(*op.qubits)
                return
            else:
                break
    raise TypeError('Gate cannot be converted to interaction: {}'.format(op.gate))

def _convert_from_regular_ops(ops):
    for op in ops:
        yield from _convert_from_regular_op(op)

def convert_circuit(circuit):
    return Circuit.from_ops(_convert_from_regular_ops(circuit.iter_ops()))

def transform_pauli_string_left(pauli_string, ops_right_to_left):
    for op in ops_right_to_left:
        if isinstance(op.gate, NonCliffordGate):
            # Skip because this gate will already have been moved out
            continue
        pauli_string.pass_op_over(op)

def _iter_ops_range(circuit, moment_start, moment_end, reverse=False):
    iter_i = range(moment_start, moment_end)
    iter_i = reversed(iter_i) if reverse else iter_i
    for i in iter_i:
        moment = circuit[i]
        for op in moment.operations:
            yield op

def _pull_non_clifford_left(circuit):
    for i, moment in enumerate(circuit):
        for op in moment.operations:
            if isinstance(op.gate, NonCliffordGate):
                pauli_string = op.gate.pauli_string.copy()
                ops_to_cross = _iter_ops_range(circuit, 0, i, reverse=True)
                transform_pauli_string_left(pauli_string, ops_to_cross)
                pauli_string.clean_axes()
                yield NonCliffordGate(pauli_string, half_turns=op.gate.half_turns).updated_op()

def non_clifford_half(circuit):
    ops = _pull_non_clifford_left(circuit)
    return Circuit.from_ops(ops, strategy=InsertStrategy.NEW)

def clifford_half(circuit):
    return Circuit(
            (Moment((op for op in moment.operations if not isinstance(op.gate, NonCliffordGate)))
             for moment in circuit))
