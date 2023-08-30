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

from typing import Union, TYPE_CHECKING

import abc

from cirq import circuits, devices, ops
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate, SwapNetworkGate
from cirq.contrib.acquaintance.bipartite import BipartiteSwapNetworkGate
from cirq.contrib.acquaintance.shift_swap_network import ShiftSwapNetworkGate
from cirq.contrib.acquaintance.permutation import PermutationGate

if TYPE_CHECKING:
    import cirq


class AcquaintanceDevice(devices.Device, metaclass=abc.ABCMeta):
    """A device that contains only acquaintance and permutation gates."""

    gate_types = (AcquaintanceOpportunityGate, PermutationGate)

    def validate_operation(self, operation: 'cirq.Operation') -> None:
        if not (
            isinstance(operation, ops.GateOperation) and isinstance(operation.gate, self.gate_types)
        ):
            raise ValueError(
                f'not (isinstance({operation!r}, {ops.Operation!r}) and '
                f'ininstance({operation!r}.gate, {self.gate_types!r})'
            )


def get_acquaintance_size(obj: Union[circuits.Circuit, ops.Operation]) -> int:
    """The maximum number of qubits to be acquainted with each other."""
    if isinstance(obj, circuits.Circuit):
        return max(tuple(get_acquaintance_size(op) for op in obj.all_operations()) or (0,))
    if not isinstance(obj, ops.Operation):
        raise TypeError('not isinstance(obj, (Circuit, Operation))')
    if not isinstance(obj, ops.GateOperation):
        return 0
    if isinstance(obj.gate, AcquaintanceOpportunityGate):
        return len(obj.qubits)
    if isinstance(obj.gate, BipartiteSwapNetworkGate):
        return 2
    if isinstance(obj.gate, ShiftSwapNetworkGate):
        return obj.gate.acquaintance_size()
    if isinstance(obj.gate, SwapNetworkGate):
        if obj.gate.acquaintance_size is None:
            return sum(sorted(obj.gate.part_lens)[-2:])
        if (obj.gate.acquaintance_size - 1) in obj.gate.part_lens:
            return obj.gate.acquaintance_size
    sizer = getattr(obj.gate, '_acquaintance_size_', None)
    return 0 if sizer is None else sizer(len(obj.qubits))


class _UnconstrainedAcquaintanceDevice(AcquaintanceDevice):
    """An acquaintance device with no constraints other than of the gate types."""

    def __repr__(self) -> str:
        return 'UnconstrainedAcquaintanceDevice'  # pragma: no cover


UnconstrainedAcquaintanceDevice = _UnconstrainedAcquaintanceDevice()
