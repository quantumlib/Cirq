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

from typing import Optional, TypeVar, Type, cast, Union

import numpy as np

from cirq import linalg, extension, protocols
from cirq.ops import raw_types, gate_features

T_DESIRED = TypeVar('T_DESIRED')

POTENTIALLY_EXPOSED_SUB_TYPES = (
    gate_features.BoundedEffect,
    gate_features.ExtrapolatableEffect,
    gate_features.ParameterizableEffect,
    gate_features.ReversibleEffect,
    gate_features.TextDiagrammable,
)


class ControlledGate(raw_types.Gate,
                     extension.PotentialImplementation[Union[
                         gate_features.BoundedEffect,
                         gate_features.ExtrapolatableEffect,
                         gate_features.ParameterizableEffect,
                         gate_features.ReversibleEffect,
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

    def extrapolate_effect(self, factor) -> 'ControlledGate':
        cast_sub_gate = self._cast_sub_gate(gate_features.ExtrapolatableEffect)
        new_sub_gate = cast_sub_gate.extrapolate_effect(factor)
        return ControlledGate(cast(raw_types.Gate, new_sub_gate),
                              self.default_extensions)

    def __pow__(self, power: float) -> 'ControlledGate':
        return self.extrapolate_effect(power)

    def inverse(self) -> 'ControlledGate':
        cast_sub_gate = self._cast_sub_gate(gate_features.ReversibleEffect)
        return ControlledGate(cast(raw_types.Gate, cast_sub_gate.inverse()),
                              self.default_extensions)

    def is_parameterized(self) -> bool:
        cast_sub_gate = self._cast_sub_gate(gate_features.ParameterizableEffect)
        return cast_sub_gate.is_parameterized()

    def with_parameters_resolved_by(self, param_resolver) -> 'ControlledGate':
        cast_sub_gate = self._cast_sub_gate(gate_features.ParameterizableEffect)
        new_sub_gate = cast_sub_gate.with_parameters_resolved_by(
            param_resolver)
        return ControlledGate(cast(raw_types.Gate, new_sub_gate),
                              self.default_extensions)

    def trace_distance_bound(self):
        cast_sub_gate = self._cast_sub_gate(gate_features.BoundedEffect)
        return cast_sub_gate.trace_distance_bound()

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        cast_sub_gate = self._cast_sub_gate(gate_features.TextDiagrammable)
        sub_info = cast_sub_gate.text_diagram_info(args)
        return gate_features.TextDiagramInfo(
            wire_symbols=('@',) + sub_info.wire_symbols,
            exponent=sub_info.exponent)

    def __str__(self):
        return 'C' + str(self.sub_gate)

    def __repr__(self):
        return 'cirq.ControlledGate(sub_gate={!r})'.format(self.sub_gate)
