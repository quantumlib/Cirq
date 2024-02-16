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

from typing import Any, Optional

from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.ops.dense_pauli_string import DensePauliString
from cirq._import import LazyLoader
import cirq.protocols.unitary_protocol as unitary_protocol
import cirq.protocols.has_unitary_protocol as has_unitary_protocol
import cirq.protocols.qid_shape_protocol as qid_shape_protocol
import cirq.protocols.decompose_protocol as decompose_protocol

pauli_string_decomposition = LazyLoader(
    "pauli_string_decomposition",
    globals(),
    "cirq.transformers.analytical_decompositions.pauli_string_decomposition",
)


def has_stabilizer_effect(val: Any) -> bool:
    """Returns whether the input has a stabilizer effect.

    For 1-qubit gates always returns correct result. For other operations relies
    on the operation to define whether it has stabilizer effect.
    """
    strats = [
        _strat_has_stabilizer_effect_from_has_stabilizer_effect,
        _strat_has_stabilizer_effect_from_gate,
        _strat_has_stabilizer_effect_from_unitary,
        _strat_has_stabilizer_effect_from_decompose,
    ]
    for strat in strats:
        result = strat(val)
        if result is not None:
            return result

    # If you can't determine if it has stabilizer effect,  it does not.
    return False


def _strat_has_stabilizer_effect_from_has_stabilizer_effect(val: Any) -> Optional[bool]:
    """Infer whether val has stabilizer effect via its `_has_stabilizer_effect_` method."""
    if hasattr(val, '_has_stabilizer_effect_'):
        result = val._has_stabilizer_effect_()
        if result is not NotImplemented and result is not None:
            return result
    return None


def _strat_has_stabilizer_effect_from_gate(val: Any) -> Optional[bool]:
    """Infer whether val's gate has stabilizer effect via the _has_stabilizer_effect_ method."""
    if hasattr(val, 'gate'):
        return _strat_has_stabilizer_effect_from_has_stabilizer_effect(val.gate)
    return None


def _strat_has_stabilizer_effect_from_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer whether val has stabilizer effect from its unitary.

    Returns whether unitary of `val` normalizes the Pauli group. Works only for
    2x2 unitaries.
    """
    # Do not try this strategy if there is no unitary or if the number of
    # qubits is greater than 3 since that would be expensive.
    qid_shape = qid_shape_protocol.qid_shape(val, default=None)
    if (
        qid_shape is None
        or len(qid_shape) > 3
        or qid_shape != (2,) * len(qid_shape)
        or not has_unitary_protocol.has_unitary(val)
    ):
        return None
    unitary = unitary_protocol.unitary(val)
    if len(qid_shape) == 1:
        return SingleQubitCliffordGate.from_unitary(unitary) is not None

    # Check if the action of the unitary on each single qubit pauli string leads to a pauli product.
    # Source: https://quantumcomputing.stackexchange.com/a/13158
    for q_idx in range(len(qid_shape)):
        for g in 'XZ':
            pauli_string = ['I'] * len(qid_shape)
            pauli_string[q_idx] = g
            ps = DensePauliString(pauli_string)
            p = ps._unitary_()
            if not pauli_string_decomposition.unitary_to_pauli_string(
                (unitary @ p @ unitary.T.conj())
            ):
                return False
    return True


def _strat_has_stabilizer_effect_from_decompose(val: Any) -> Optional[bool]:
    decomposition, _, _ = decompose_protocol._try_decompose_into_operations_and_qubits(val)
    if decomposition is None:
        return None
    for op in decomposition:
        res = has_stabilizer_effect(op)
        if not res:
            return res
    return True
