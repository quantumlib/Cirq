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

from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union
import numpy as np

from cirq import ops, protocols, value
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


# Tag for gates to which noise must be applied.
PHYSICAL_GATE_TAG = 'physical_gate'


@value.value_equality(distinct_child_types=True)
class OpIdentifier:
    """Identifies an operation by gate and (optionally) target qubits."""

    def __init__(self, gate_type: Type['cirq.Gate'], *qubits: 'cirq.Qid'):
        self._gate_type = gate_type
        self._gate_family = ops.GateFamily(gate_type)
        self._qubits: Tuple['cirq.Qid', ...] = tuple(qubits)

    @property
    def gate_type(self) -> Type['cirq.Gate']:
        # set to a type during initialization, never modified
        return self._gate_type

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self._qubits

    def _predicate(self, *args, **kwargs):
        return self._gate_family._predicate(*args, **kwargs)

    def is_proper_subtype_of(self, op_id: 'OpIdentifier'):
        """Returns true if this is contained within op_id, but not equal to it.

        If this returns true, (x in self) implies (x in op_id), but the reverse
        implication does not hold. op_id must be more general than self (either
        by accepting any qubits or having a more general gate type) for this
        to return true.
        """
        more_specific_qubits = self.qubits and not op_id.qubits
        more_specific_gate = self.gate_type != op_id.gate_type and issubclass(
            self.gate_type, op_id.gate_type
        )
        if more_specific_qubits:
            return more_specific_gate or self.gate_type == op_id.gate_type
        elif more_specific_gate:
            return more_specific_qubits or self.qubits == op_id.qubits
        else:
            return False

    def __contains__(self, item: Union[ops.Gate, ops.Operation]) -> bool:
        if isinstance(item, ops.Gate):
            return (not self._qubits) and self._predicate(item)
        return (
            (not self.qubits or (item.qubits == self._qubits))
            and item.gate is not None
            and self._predicate(item.gate)
        )

    def __str__(self):
        return f'{self.gate_type}{self.qubits}'

    def __repr__(self) -> str:
        qubits = ', '.join(map(repr, self.qubits))
        return f'cirq.devices.noise_utils.OpIdentifier({proper_repr(self.gate_type)}, {qubits})'

    def _value_equality_values_(self) -> Any:
        return (self.gate_type, self.qubits)

    def _json_dict_(self) -> Dict[str, Any]:
        if hasattr(self.gate_type, '__name__'):
            return {'gate_type': protocols.json_cirq_type(self._gate_type), 'qubits': self._qubits}
        return {'gate_type': self._gate_type, 'qubits': self._qubits}

    @classmethod
    def _from_json_dict_(cls, gate_type, qubits, **kwargs) -> 'OpIdentifier':
        if isinstance(gate_type, str):
            gate_type = protocols.cirq_type_from_json(gate_type)
        return cls(gate_type, *qubits)


# TODO: expose all from top-level cirq?
def decay_constant_to_xeb_fidelity(decay_constant: float, num_qubits: int = 2) -> float:
    """Calculates the XEB fidelity from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant.
        num_qubits: Number of qubits.

    Returns:
        Calculated XEB fidelity.
    """
    N = 2**num_qubits
    return 1 - ((1 - decay_constant) * (1 - 1 / N))


def decay_constant_to_pauli_error(decay_constant: float, num_qubits: int = 1) -> float:
    """Calculates pauli error from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant.
        num_qubits: Number of qubits.

    Returns:
        Calculated Pauli error.
    """
    N = 2**num_qubits
    return (1 - decay_constant) * (1 - 1 / N / N)


def pauli_error_to_decay_constant(pauli_error: float, num_qubits: int = 1) -> float:
    """Calculates depolarization decay constant from pauli error.

    Args:
        pauli_error: The pauli error.
        num_qubits: Number of qubits.

    Returns:
        Calculated depolarization decay constant.
    """
    N = 2**num_qubits
    return 1 - (pauli_error / (1 - 1 / N / N))


def xeb_fidelity_to_decay_constant(xeb_fidelity: float, num_qubits: int = 2) -> float:
    """Calculates the depolarization decay constant from XEB fidelity.

    Args:
        xeb_fidelity: The XEB fidelity.
        num_qubits: Number of qubits.

    Returns:
        Calculated depolarization decay constant.
    """
    N = 2**num_qubits
    return 1 - (1 - xeb_fidelity) / (1 - 1 / N)


def pauli_error_from_t1(t_ns: float, t1_ns: float) -> float:
    """Calculates the pauli error from T1 decay constant.

    This computes error for a specific duration, `t`.

    Args:
        t_ns: The duration of the gate in ns.
        t1_ns: The T1 decay constant in ns.

    Returns:
        Calculated Pauli error resulting from T1 decay.
    """
    t2 = 2 * t1_ns
    return (1 - np.exp(-t_ns / t2)) / 2 + (1 - np.exp(-t_ns / t1_ns)) / 4


def average_error(decay_constant: float, num_qubits: int = 1) -> float:
    """Calculates the average error from the depolarization decay constant.

    Args:
        decay_constant: Depolarization decay constant.
        num_qubits: Number of qubits.

    Returns:
        Calculated average error.
    """
    N = 2**num_qubits
    return (1 - decay_constant) * (1 - 1 / N)


def decoherence_pauli_error(t1_ns: float, tphi_ns: float, gate_time_ns: float) -> float:
    """The component of Pauli error caused by decoherence on a single qubit.

    Args:
        t1_ns: T1 time in nanoseconds.
        tphi_ns: Tphi time in nanoseconds.
        gate_time_ns: Duration in nanoseconds of the gate affected by this error.

    Returns:
        Calculated Pauli error resulting from decoherence.
    """
    gamma_2 = (1 / (2 * t1_ns)) + 1 / tphi_ns

    exp1 = np.exp(-gate_time_ns / t1_ns)
    exp2 = np.exp(-gate_time_ns * gamma_2)
    px = 0.25 * (1 - exp1)
    py = px
    pz = 0.5 * (1 - exp2) - px
    return px + py + pz
