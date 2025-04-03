# Copyright 2022 The Cirq Developers
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

import abc
from typing import List, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from typing_extensions import Self

from cirq import value

if TYPE_CHECKING:
    import cirq


class QuantumStateRepresentation(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def copy(self, deep_copy_buffers: bool = True) -> Self:
        """Creates a copy of the object.
        Args:
            deep_copy_buffers: If True, buffers will also be deep-copied.
            Otherwise the copy will share a reference to the original object's
            buffers.
        Returns:
            A copied instance.
        """

    @abc.abstractmethod
    def measure(
        self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None
    ) -> List[int]:
        """Measures the state.

        Args:
            axes: The axes to measure.
            seed: The random number seed to use.
        Returns:
            The measurements in order.
        """

    def sample(
        self,
        axes: Sequence[int],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        """Samples the state. Subclasses can override with more performant method.

        Args:
            axes: The axes to sample.
            repetitions: The number of samples to make.
            seed: The random number seed to use.
        Returns:
            The samples in order.
        """
        prng = value.parse_random_state(seed)
        measurements = []
        for _ in range(repetitions):
            state = self.copy()
            measurements.append(state.measure(axes, prng))
        return np.array(measurements, dtype=np.uint8)

    def kron(self, other: Self) -> Self:
        """Joins two state spaces together."""
        raise NotImplementedError()

    def factor(self, axes: Sequence[int], *, validate=True, atol=1e-07) -> Tuple[Self, Self]:
        """Splits two state spaces after a measurement or reset."""
        raise NotImplementedError()

    def reindex(self, axes: Sequence[int]) -> Self:
        """Physically reindexes the state by the new basis.
        Args:
            axes: The desired axis order.
        Returns:
            The state with qubit order transposed and underlying representation
            updated.
        """
        raise NotImplementedError()

    @property
    def supports_factor(self) -> bool:
        """Subclasses that allow factorization should override this."""
        return False

    @property
    def can_represent_mixed_states(self) -> bool:
        """Subclasses that can represent mixed states should override this."""
        return False
