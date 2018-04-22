# Copyright 2018 Google LLC
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

"""Partial reflection gate."""
from typing import Tuple, Union

import numpy as np

from cirq import abc
from cirq.extension import PotentialImplementation
from cirq.ops import gate_features
from cirq.value import Symbol


def _canonicalize_half_turns(
        half_turns: Union[Symbol, float]
) -> Union[Symbol, float]:
    if isinstance(half_turns, Symbol):
        return half_turns
    half_turns %= 2
    if half_turns > 1:
        half_turns -= 2
    return half_turns


class PartialReflectionGate(gate_features.BoundedEffectGate,
                            gate_features.TextDiagrammableGate,
                            PotentialImplementation):
    """A gate with two active eigenvalues.

    PartialReflectionGates have a direct sum decomposition I ⊕ U, where I is the
    identity and U has exactly two eigenvalues. Extrapolating the gate phases
    one eigenspace of U relative to the other, with half_turns=1 corresponding
    to the point where the relative phase factor is exactly -1.
    """

    def __init__(self,
                 *positional_args,
                 half_turns: Union[Symbol, float] = 1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    @abc.abstractmethod
    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True
                                  ) -> Tuple[str, ...]:
        pass

    def text_diagram_exponent(self):
        return self.half_turns

    def __pow__(self, power: float) -> 'PartialReflectionGate':
        return self.extrapolate_effect(power)

    def inverse(self) -> 'PartialReflectionGate':
        return self.extrapolate_effect(-1)

    def __repr__(self):
        base = ''.join(self.text_diagram_wire_symbols())
        if self.half_turns == 1:
            return base
        return '{}**{}'.format(base, repr(self.half_turns))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self), self.half_turns))

    def trace_distance_bound(self):
        if isinstance(self.half_turns, Symbol):
            return 1
        return abs(self.half_turns) * 3.5

    def try_cast_to(self, desired_type):
        if (desired_type in [gate_features.ExtrapolatableGate,
                             gate_features.ReversibleGate] and
                self.can_extrapolate_effect()):
            return self
        if desired_type is gate_features.KnownMatrixGate and self.has_matrix():
            return self
        return super().try_cast_to(desired_type)

    @abc.abstractmethod
    def _matrix_impl_assuming_unparameterized(self) -> np.ndarray:
        pass

    def has_matrix(self) -> bool:
        return not isinstance(self.half_turns, Symbol)

    def matrix(self) -> np.ndarray:
        if not self.has_matrix():
            raise ValueError("Parameterized. Don't have a known matrix.")
        return self._matrix_impl_assuming_unparameterized()

    def can_extrapolate_effect(self) -> bool:
        return not isinstance(self.half_turns, Symbol)

    def extrapolate_effect(self, factor) -> 'PartialReflectionGate':
        if not self.can_extrapolate_effect():
            raise ValueError("Parameterized. Don't have a known matrix.")
        return type(self)(half_turns=self.half_turns * factor)
