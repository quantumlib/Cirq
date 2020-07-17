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

        Raises:
            ValueError if this sampler has a device and the circuit is not
            valid for the device.
        """
        if self.device:
            self.device.validate_circuit(program)
        measurements = {}  # type: Dict[str, np.ndarray]
        for op in program.all_operations():
            key = protocols.measurement_key(op, default=None)
            if key is not None:
                measurements[key] = np.zeros((repetitions, len(op.qubits)),
                                             dtype=int)
        return [
            study.TrialResult(params=param_resolver, measurements=measurements)
            for param_resolver in study.to_resolvers(params)
        ]
