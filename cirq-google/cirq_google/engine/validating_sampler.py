# Copyright 2021 The Cirq Developers
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
from typing import Callable, List, Optional, Sequence, Union

import cirq

VALIDATOR_TYPE = Callable[
    [Sequence[cirq.AbstractCircuit], Sequence[cirq.Sweepable], Union[int, List[int]]], None
]


class ValidatingSampler(cirq.Sampler):
    def __init__(
        self,
        *,
        device: Optional[cirq.Device] = None,
        validator: Optional[VALIDATOR_TYPE] = None,
        sampler: cirq.Sampler = cirq.Simulator(),
    ):
        """Wrapper around `cirq.Sampler` that performs device validation.

        This sampler will delegate to the wrapping sampler after
        performing validation on the circuit(s) given to the sampler.

        Args:
            device: `cirq.Device` that will validate_circuit before sampling.
            validator: A callable that will do any additional validation
               beyond the device.  For instance, this can perform serialization
               checks.  Note that this function takes a list of circuits and
               sweeps so that batch functionality can also be tested.
            sampler: sampler wrapped by this object.  After validating,
                samples will be returned by this enclosed `cirq.Sampler`.
        """
        self._device = device
        self._validator = validator
        self._sampler = sampler

    def _validate_circuit(
        self,
        circuits: Sequence[cirq.AbstractCircuit],
        sweeps: List[cirq.Sweepable],
        repetitions: Union[int, List[int]],
    ):
        if self._device:
            for circuit in circuits:
                self._device.validate_circuit(circuit)
        if self._validator:
            self._validator(circuits, sweeps, repetitions)

    def run_sweep(
        self,
        program: cirq.AbstractCircuit,
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> Sequence[cirq.Result]:
        self._validate_circuit([program], [params], repetitions)
        return self._sampler.run_sweep(program, params, repetitions)

    def run_batch(
        self,
        programs: Sequence[cirq.AbstractCircuit],
        params_list: Optional[List[cirq.Sweepable]] = None,
        repetitions: Union[int, List[int]] = 1,
    ) -> Sequence[Sequence[cirq.Result]]:
        if params_list is None:
            params_list = [None] * len(programs)
        self._validate_circuit(programs, params_list, repetitions)
        return self._sampler.run_batch(programs, params_list, repetitions)
