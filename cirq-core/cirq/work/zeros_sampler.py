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

import abc
import collections
from typing import Dict, List, TYPE_CHECKING

import numpy as np

from cirq import devices, work, study, protocols

if TYPE_CHECKING:
    import cirq


class ZerosSampler(work.Sampler, metaclass=abc.ABCMeta):
    """A dummy sampler for testing. Immediately returns zeroes."""

    def __init__(self, device: devices.Device = None):
        """Construct a sampler that returns 0 for all measurements.

        Args:
            device: A device against which to validate the circuit. If None,
                no validation will be done.
        """
        self.device = device

    def run_sweep(
        self,
        program: 'cirq.AbstractCircuit',
        params: study.Sweepable,
        repetitions: int = 1,
    ) -> List[study.Result]:
        """Samples circuit as if every measurement resulted in zero.

        Args:
            program: The circuit to sample from.
            params: Parameters to run with the program.
            repetitions: The number of times to sample.

        Returns:
            Result list for this run; one for each possible parameter
            resolver.

        Raises:
            ValueError: circuit is not valid for the sampler, due to invalid
            repeated keys or incompatibility with the sampler's device.
        """
        if self.device:
            self.device.validate_circuit(program)
        num_qubits: Dict[str, int] = {}
        num_instances: Dict[str, int] = collections.Counter()
        for op in program.all_operations():
            key = protocols.measurement_key_name(op, default=None)
            if key is not None:
                n = len(op.qubits)
                prev_n = num_qubits.setdefault(key, n)
                if n != prev_n:
                    raise ValueError(
                        "Different num qubits for repeated measurement: "
                        f"key={key!r}, prev_n={prev_n}, n={n}"
                    )
                num_instances[key] += 1
        return [
            study.ResultDict(
                params=param_resolver,
                records={
                    k: np.zeros((repetitions, num_instances[k], n), dtype=int)
                    for k, n in num_qubits.items()
                },
            )
            for param_resolver in study.to_resolvers(params)
        ]
