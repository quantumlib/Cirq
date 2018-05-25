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

from typing import Tuple, Union, List

import numpy as np

from cirq import abc
from cirq.extension import PotentialImplementation
from cirq.ops import gate_features
from cirq.value import Symbol


class EigenGate(gate_features.BoundedEffectGate,
                gate_features.ParameterizableGate,
                PotentialImplementation):
    """A gate with a known eigendecomposition.

    EigenGate is particularly useful when one wishes for different parts of
    the same eigenspace to be extrapolated differently. For example, if a gate
    has a 2-dimensional eigenspace with eigenvalue -1, but one wishes for the
    square root of the gate to split this eigenspace into a part with
    eigenvalue i and a part with eigenvalue -i, then EigenGate allows this
    functionality to be unambiguously specified via the _eigen_components
    method.
    """

    def __init__(self,
                 *positional_args,
                 exponent: Union[Symbol, float] = 1.0) -> None:
        assert not positional_args

        # Canonicalize the exponent.
        period = self._canonical_exponent_period()
        if period is not None and not isinstance(exponent, Symbol):
            exponent += period / 2
            exponent %= period
            exponent -= period / 2

        self.exponent = exponent

    @abc.abstractmethod
    def _with_exponent(self,
                       exponent: Union[Symbol, float] = 1.0
                       ) -> 'EigenGate':
        """Return the same kind of gate, but with a different exponent."""
        pass

    @abc.abstractmethod
    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        """A decomposition of the gate into (λ_half_turns, Σ|λ⟩⟨λ|) pieces.

        Returns:
            A list of tuples. Each tuple corresponds to an eigenspace of the
            gate. The first component of a tuple is how much that eigenspace
            should be phased by applying the gate, in half_turn units.
            The second component is the projection of the gate's matrix into
            that eigenspace.

            For example, the Pauli X gate's eigen components method would
            return [
                (0, np.array([[0.5, 0.5],
                              [0.5, 0.5]])),
                (1, np.array([[+0.5, -0.5],
                              [-0.5, +0.5]])),
            ].
        """
        pass

    @abc.abstractmethod
    def _canonical_exponent_period(self) -> Union[float, None]:
        """Determines how the exponent parameter is canonicalized.

        Returns:
            None if the exponent should not be canonicalized. Otherwise a float
            indicating the period of the exponent. If the period is p, then a
            given exponent will be shifted by p until it is in the range
            [-p, p) during initialization.
        """
        pass

    def __pow__(self, power: float) -> 'EigenGate':
        return self.extrapolate_effect(power)

    def inverse(self) -> 'EigenGate':
        return self.extrapolate_effect(-1)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.exponent == other.exponent

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self), self.exponent))

    def trace_distance_bound(self):
        if isinstance(self.exponent, Symbol):
            return 1

        angles = [half_turns for half_turns, _ in self._eigen_components()]
        min_angle = min(angles)
        max_angle = max(angles)
        return abs((max_angle - min_angle) * self.exponent * 3.5)

    def try_cast_to(self, desired_type):
        if (desired_type in [gate_features.ExtrapolatableGate,
                             gate_features.ReversibleGate] and
                not self.is_parameterized()):
            return self
        if (desired_type in [gate_features.SelfInverseGate] and
                not self.is_parameterized() and
                all(angle * self.exponent % 1 == 0
                    for angle, _ in self._eigen_components())):
            return self
        if (desired_type is gate_features.KnownMatrixGate and
                not self.is_parameterized()):
            return self
        return super().try_cast_to(desired_type)

    def matrix(self) -> np.ndarray:
        if self.is_parameterized():
            raise ValueError("Parameterized. Don't have a known matrix.")
        return np.sum(
            np.exp(1j * np.pi * half_turns * self.exponent) * component
            for half_turns, component in self._eigen_components()
        )

    def extrapolate_effect(self, factor) -> 'EigenGate':
        if self.is_parameterized():
            raise ValueError("Parameterized. Don't know how to extrapolate.")
        return self._with_exponent(exponent=self.exponent * factor)

    def is_parameterized(self) -> bool:
        return isinstance(self.exponent, Symbol)

    def with_parameters_resolved_by(self, param_resolver) -> 'EigenGate':
        return self._with_exponent(
                exponent=param_resolver.value_of(self.exponent))
