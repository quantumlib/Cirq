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

from typing import Tuple, Union, List, Optional, cast, TypeVar

import numpy as np

from cirq import abc, extension, value
from cirq.ops import gate_features, raw_types


TSelf = TypeVar('TSelf', bound='EigenGate')


class EigenGate(raw_types.Gate,
                gate_features.BoundedEffect,
                gate_features.ParameterizableEffect,
                extension.PotentialImplementation[Union[
                    gate_features.ExtrapolatableEffect,
                    gate_features.ReversibleEffect]]):
    """A gate with a known eigendecomposition.

    EigenGate is particularly useful when one wishes for different parts of
    the same eigenspace to be extrapolated differently. For example, if a gate
    has a 2-dimensional eigenspace with eigenvalue -1, but one wishes for the
    square root of the gate to split this eigenspace into a part with
    eigenvalue i and a part with eigenvalue -i, then EigenGate allows this
    functionality to be unambiguously specified via the _eigen_components
    method.
    """

    def __init__(self, *,  # Forces keyword args.
                 exponent: Union[value.Symbol, float] = 1.0) -> None:

        # Canonicalize the exponent.
        period = self._canonical_exponent_period()
        if period is not None and not isinstance(exponent, value.Symbol):
            # Shift into [-p/2, +p/2).
            exponent += period / 2
            exponent %= period
            exponent -= period / 2
            # Prefer (-p/2, +p/2] over [-p/2, +p/2).
            if exponent <= -period / 2:
                exponent += period

        self._exponent = exponent

    @abc.abstractmethod
    def _with_exponent(self: TSelf,
                       exponent: Union[value.Symbol, float]) -> TSelf:
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
    def _canonical_exponent_period(self) -> Optional[float]:
        """Determines how the exponent parameter is canonicalized.

        Returns:
            None if the exponent should not be canonicalized. Otherwise a float
            indicating the period of the exponent. If the period is p, then a
            given exponent will be shifted by p until it is in the range
            (-p/2, p/2] during initialization.
        """
        pass

    def __pow__(self: TSelf, power: float) -> TSelf:
        return self.extrapolate_effect(power)

    def inverse(self: TSelf) -> TSelf:
        return self.extrapolate_effect(-1)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._exponent == other._exponent

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self), self._exponent))

    def trace_distance_bound(self):
        if isinstance(self._exponent, value.Symbol):
            return 1

        angles = [half_turns for half_turns, _ in self._eigen_components()]
        min_angle = min(angles)
        max_angle = max(angles)
        return abs((max_angle - min_angle) * self._exponent * 3.5)

    def try_cast_to(self, desired_type, ext):
        if (desired_type in [gate_features.ExtrapolatableEffect,
                             gate_features.ReversibleEffect] and
                not self.is_parameterized()):
            return self
        return super().try_cast_to(desired_type, ext)

    def _maybe_unitary_effect_(self) -> Optional[np.ndarray]:
        if self.is_parameterized():
            return None
        e = cast(float, self._exponent)
        return np.sum(1j**(half_turns * e * 2) * component
                      for half_turns, component in self._eigen_components())

    def extrapolate_effect(self: TSelf,
                           factor: Union[float, value.Symbol]) -> TSelf:
        return self._with_exponent(
            exponent=self._exponent * factor)  # type: ignore

    def is_parameterized(self) -> bool:
        return isinstance(self._exponent, value.Symbol)

    def with_parameters_resolved_by(self: TSelf, param_resolver) -> TSelf:
        return self._with_exponent(
                exponent=param_resolver.value_of(self._exponent))
