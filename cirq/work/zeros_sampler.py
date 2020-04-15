# Copyright 2020 The Cirq Developers
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

from typing import (
    List,
    Set,
)

import abc
import numpy as np

from cirq import work, study, protocols, ops, circuits


class ZerosSampler(work.Sampler, metaclass=abc.ABCMeta):
    """A dummy sampler for testing. Immediately returns zeroes."""

    def run_sweep(
            self,
            program: 'cirq.Circuit',
            params: study.Sweepable,
            repetitions: int = 1,
    ) -> List[study.TrialResult]:
        """Samples circuit as if every measurement resulted in zero.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
        zero_measurement = np.zeros((repetitions, 1), dtype=np.int8)
        measurements = {
            key: zero_measurement for key in _all_measurement_keys(program)
        }
        return [
            study.TrialResult.from_single_parameter_set(
                params=param_resolver, measurements=measurements)
            for param_resolver in study.to_resolvers(params)
        ]


def _all_measurement_keys(circuit: circuits.Circuit) -> Set[str]:
    result = set()  # type: Set[str]
    for op in ops.flatten_op_tree(iter(circuit)):
        key = protocols.measurement_key(op, default=None)
        if key is not None:
            result.add(key)
    return result
