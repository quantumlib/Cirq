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

"""Partial reflection gate."""
from typing import Tuple, Union, cast

import numpy as np

from cirq import abc
from cirq.extension import PotentialImplementation
from cirq.linalg import reflection_matrix_pow
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
                            gate_features.ParameterizableGate,
                            gate_features.TextDiagrammableGate,
                            PotentialImplementation):
    """An interpolated reflection operation.

    A reflection operation is an operation that has exactly two eigenvalues
    which differ by 180 degrees (i.e. equal x and -x for some x).
    For an interpolated reflection operation, the eigenvalues differ by a
    relative phase not necessarily equal to 180 degrees.
    A PartialReflectionGate has a direct sum decomposition I ⊕ U or simply U,
    where I is the identity and U is an interpolated reflection operation.
    Extrapolating the gate phases one eigenspace of U relative to the other,
    with half_turns=1 corresponding to the point where U is a reflection
    operation (i.e., the relative phase is exactly -1).
    """
    def __init__(self,
                 *positional_args,
                 half_turns: Union[Symbol, float] = 1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    @abc.abstractmethod
    def _with_half_turns(self,
                         half_turns: Union[Symbol, float] = 1.0
                         ) -> 'PartialReflectionGate':
        """Initialize an instance by specifying the number of half-turns."""
        pass

    @abc.abstractmethod
    def text_diagram_wire_symbols(self,
                                  qubit_count = None,
                                  use_unicode_characters = True,
                                  precision = 3
                                  ) -> Tuple[str, ...]:
        pass

    def text_diagram_exponent(self):
        return self.half_turns

    def __pow__(self, power: float) -> 'PartialReflectionGate':
        return self.extrapolate_effect(power)

    def inverse(self) -> 'PartialReflectionGate':
        return self.extrapolate_effect(-1)

    def __str__(self):
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

    def try_cast_to(self, desired_type, ext):
        if (desired_type in [gate_features.ExtrapolatableGate,
                             gate_features.ReversibleGate] and
                not self.is_parameterized()):
            return self
        if (desired_type in [gate_features.SelfInverseGate] and
                not self.is_parameterized() and
                self.half_turns % 1 == 0):
            return self
        if (desired_type is gate_features.KnownMatrixGate and
                not self.is_parameterized()):
            return self
        return super().try_cast_to(desired_type, ext)

    @abc.abstractmethod
    def _reflection_matrix(self) -> np.ndarray:
        """The reflection matrix corresponding to half_turns=1."""
        pass

    def matrix(self) -> np.ndarray:
        if self.is_parameterized():
            raise ValueError("Parameterized. Don't have a known matrix.")
        return reflection_matrix_pow(
                self._reflection_matrix(), cast(float, self.half_turns))

    def extrapolate_effect(self, factor) -> 'PartialReflectionGate':
        if self.is_parameterized():
            raise ValueError("Parameterized. Don't know how to extrapolate.")
        return self._with_half_turns(half_turns=self.half_turns * factor)

    def is_parameterized(self) -> bool:
        return isinstance(self.half_turns, Symbol)

    def with_parameters_resolved_by(self,
                                    param_resolver) -> 'PartialReflectionGate':
        return self._with_half_turns(
                half_turns=param_resolver.value_of(self.half_turns))
