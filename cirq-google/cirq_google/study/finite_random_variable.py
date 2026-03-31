# Copyright 2025 The Cirq Developers
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


import numbers
import random
from collections.abc import Iterator, Mapping
from typing import Any, cast

import cirq
from cirq.study.sweeps import SingleSweep


class FiniteRandomVariable(SingleSweep):
    """A sweep over randomly-sampled values from a finite distribution.

    This can generate a stream of random samples pulled from a finite
    distribution.  Some examples of possible finite distributions include
    a coin flip, die roll, or from {1, 0, -1}.

    This sweep uses Python's internal `random.choices` to generate samples
    from the requested distribution.  Given the same seed, this sequence
    will be identical across machines.

    Note: this is an experimental sweep and is not guaranteed to be
    future-compatible.

    Args:
        key:  symbol or string to sweep across.
        distribution: dictionary with keys of the potential values of the
            sweep.  The value of the dictionary is the weight of the value
            in the resulting statistical distribution.  This will be
            normalized to one.  For instance, specifying weights 1,2,1 will
            result in a 25%, 50%, 25% distribution for the respective keys.
        seed: A number to seed the pseudo-random number generator.
        length:  Number of samples which is the same as the number of
            sweep points.
    """

    def __init__(
        self,
        key: cirq.TParamKey,
        distribution: Mapping[float, float],
        seed: int,
        length: int,
        metadata: Any | None = None,
    ) -> None:
        """Creates a random sweep for a given key.

        Args:
            key: sympy.Symbol or equivalent to sweep across.
            distribution: A dictionary mapping values to their probabilities.
                Probabilities do not need to sum to 1; they will be
                normalized by `random.choices`.
            seed: A seed for the random number generator to ensure
                reproducibility.
            length: The number of points in the sweep.
            metadata: Optional metadata to attach to the sweep.
        """
        if not isinstance(distribution, dict) or not distribution:
            raise ValueError("distribution must be a non-empty dictionary.")
        if not all(isinstance(k, numbers.Real) for k in distribution.keys()):
            raise ValueError("distribution keys must be numbers.")
        if not all(
            isinstance(v, numbers.Real) and cast(float, v) >= 0.0 for v in distribution.values()
        ):
            raise ValueError("distribution values (weights) must be non-negative numbers.")
        if length <= 0:
            raise ValueError("length must be a positive integer.")
        super().__init__(key)
        self.distribution = distribution
        self.seed = seed
        self.length = length
        self.metadata = metadata

    def __len__(self) -> int:
        return self.length

    def _values(self) -> Iterator[float]:
        # Save the state of the RNG
        prev_state = random.getstate()
        # Generate self.length random values starting with the seed
        random.seed(self.seed)
        random_values = random.choices(
            list(self.distribution.keys()), list(self.distribution.values()), k=self.length
        )
        # Restore the RNG state
        random.setstate(prev_state)
        return iter(random_values)

    def _tuple(self):
        return (
            self.key,
            tuple(sorted(self.distribution.items())),
            self.seed,
            self.length,
            self.metadata,
        )

    def __repr__(self) -> str:
        metadata_repr = f', metadata={self.metadata!r}' if self.metadata is not None else ""
        return (
            f'cirq_google.study.FiniteRandomVariable({self.key!r}, '
            f'distribution={self.distribution!r}, '
            f'seed={self.seed!r}, length={self.length!r}{metadata_repr})'
        )

    @classmethod
    def _from_json_dict_(
        cls,
        *,
        key: cirq.TParamKey,
        distribution: dict[str, float],
        seed: int,
        length: int,
        metadata: Any | None = None,
        **kwargs,
    ) -> 'FiniteRandomVariable':
        # Convert json keys to floats
        fixed_distribution = {float(key): distribution[key] for key in distribution}
        return cls(
            key=key, distribution=fixed_distribution, seed=seed, length=length, metadata=metadata
        )

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["key", "distribution", "seed", "length", "metadata"])
