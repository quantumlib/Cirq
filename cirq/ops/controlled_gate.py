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

from typing import Optional, TypeVar, Type, Union, Sequence

import numpy as np

from cirq import linalg, extension, protocols
from cirq.ops import raw_types, gate_features

T_DESIRED = TypeVar('T_DESIRED')

POTENTIALLY_EXPOSED_SUB_TYPES = (
    gate_features.TextDiagrammable,
)


class ControlledGate(raw_types.Gate,
                     extension.PotentialImplementation[Union[
                         gate_features.TextDiagrammable,
                     ]]):
    """Augments existing gates with a control qubit."""

    def __init__(self,
                 sub_gate: raw_types.Gate,
                 default_extensions: Optional[extension.Extensions] = None
                 ) -> None:
        """Initializes the controlled gate.

        Args:
            sub_gate: The gate to add a control qubit to.
            default_extensions: The extensions method that should be used when
                determining if the controlled gate supports certain gate
                features. For example, if this extensions instance is able to
                cast sub_gate to a ReversibleEffect then the controlled gate
                can also be cast to a ReversibleEffect. When this value is None,
                an empty extensions instance is used instead.
        """
        self.sub_gate = sub_gate
        self.default_extensions = default_extensions

    def validate_args(self, qubits) -> None:
        if len(qubits) < 1:
            raise ValueError('No control qubit specified.')
        self.sub_gate.validate_args(qubits[1:])

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.sub_gate == other.sub_gate

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ControlledGate, self.sub_gate))

    def _cast_sub_gate(self, desired_type: Type[T_DESIRED]) -> T_DESIRED:
        ext = self.default_extensions or extension.Extensions()
        cast_sub_gate = ext.try_cast(desired_type, self.sub_gate)
        if cast_sub_gate is None:
            raise TypeError('sub_gate is not a {}', desired_type)
        return cast_sub_gate

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> np.ndarray:
        control = axes[0]
        rest = axes[1:]
        active = linalg.slice_for_qubits_equal_to([control], 1)
        sub_axes = [r - int(r > control) for r in rest]
        target_view = target_tensor[active]
        buffer_view = available_buffer[active]
        result = protocols.apply_unitary_to_tensor(
            self.sub_gate,
            target_view,
            buffer_view,
            sub_axes,
            default=NotImplemented)

        if result is NotImplemented:
            return NotImplemented

        if result is target_view:
            return target_tensor

        if result is buffer_view:
            inactive = linalg.slice_for_qubits_equal_to([control], 0)
            available_buffer[inactive] = target_tensor[inactive]
            return available_buffer

        # HACK: assume they didn't somehow escape the slice view and edit the
        # rest of target_tensor.
        target_tensor[active] = result
        return target_tensor

    def try_cast_to(self, desired_type, ext):
        if desired_type in POTENTIALLY_EXPOSED_SUB_TYPES:
            cast_sub_gate = ext.try_cast(desired_type, self.sub_gate)
            if cast_sub_gate is None:
                return None
            return ControlledGate(cast_sub_gate, ext)
        return super().try_cast_to(desired_type, ext)

    def _unitary_(self) -> Union[np.ndarray, type(NotImplemented)]:
        sub_matrix = protocols.unitary(self.sub_gate, None)
        if sub_matrix is None:
            return NotImplemented
        return linalg.block_diag(np.eye(sub_matrix.shape[0]), sub_matrix)

    def __pow__(self, power: float) -> 'ControlledGate':
        new_sub_gate = protocols.extrapolate(self.sub_gate,
                                             power,
                                             NotImplemented)
        if new_sub_gate is NotImplemented:
            return NotImplemented
        return ControlledGate(new_sub_gate, self.default_extensions)

    def _is_parameterized_(self):
        return protocols.is_parameterized(self.sub_gate)

    def _resolve_parameters_(self, param_resolver):
        new_sub_gate = protocols.resolve_parameters(self.sub_gate,
                                                    param_resolver)
        return ControlledGate(new_sub_gate, self.default_extensions)

    def _trace_distance_bound_(self):
        return protocols.trace_distance_bound(self.sub_gate)

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        sub_info = protocols.circuit_diagram_info(self.sub_gate, args, None)
        if sub_info is None:
            return NotImplemented
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@',) + sub_info.wire_symbols,
            exponent=sub_info.exponent)

    def __str__(self):
        return 'C' + str(self.sub_gate)

    def __repr__(self):
        return 'cirq.ControlledGate(sub_gate={!r})'.format(self.sub_gate)
