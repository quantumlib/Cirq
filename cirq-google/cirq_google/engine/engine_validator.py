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
from typing import Callable, List, Sequence, Union
from google.protobuf import any_pb2

import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer
from cirq_google.api import v2

MAX_MESSAGE_SIZE = 10_000_000
MAX_MOMENTS = 10000
MAX_TOTAL_REPETITIONS = 5_000_000

GATE_SET_VALIDATOR_TYPE = Callable[
    [Sequence[cirq.AbstractCircuit], List[cirq.Sweepable], int, 'Serializer'],
    None,
]


def _validate_depth(
    circuits: Sequence[cirq.AbstractCircuit],
    max_moments: int = MAX_MOMENTS,
) -> None:
    """Validate that the depth of the circuit is not too long (too many moments)."""
    for circuit in circuits:
        if len(circuit) > max_moments:
            raise RuntimeError(f'Provided circuit exceeds the limit of {max_moments} moments.')


def _verify_reps(
    sweeps: Sequence[cirq.Sweepable],
    repetitions: Union[int, List[int]],
    max_repetitions: int = MAX_TOTAL_REPETITIONS,
) -> None:
    """Verify that the total number of repetitions is under the limit."""
    total_reps = 0
    for idx, sweep in enumerate(sweeps):
        if isinstance(repetitions, List):
            total_reps += len(list(cirq.to_resolvers(sweep))) * repetitions[idx]
        else:
            total_reps += len(list(cirq.to_resolvers(sweep))) * repetitions
    if total_reps > max_repetitions:
        raise RuntimeError(
            f'No requested processors currently support the number of requested total repetitions.'
        )


def _verify_measurements(circuits):
    """Verify the circuit has appropriate measurements."""
    for circuit in circuits:
        has_measurement = any(
            isinstance(op.gate, cirq.MeasurementGate) for moment in circuit for op in moment
        )
        if not has_measurement:
            raise RuntimeError('Code must measure at least one qubit.')


def validate_gate_set(
    circuits: Sequence[cirq.AbstractCircuit],
    sweeps: List[cirq.Sweepable],
    repetitions: int,
    gate_set: Serializer,
    max_size: int = MAX_MESSAGE_SIZE,
) -> None:
    """Validate that the message size is below the maximum size limit."""
    batch = v2.batch_pb2.BatchProgram()
    packed = any_pb2.Any()
    for circuit in circuits:
        gate_set.serialize(circuit, msg=batch.programs.add())
    packed.Pack(batch)
    message_size = len(packed.SerializeToString())
    if message_size > max_size:
        raise RuntimeError("INVALID_PROGRAM: Program too long.")


def create_gate_set_validator(max_size: int = MAX_MESSAGE_SIZE) -> GATE_SET_VALIDATOR_TYPE:
    """Creates a Callanle gate set validator with a set message size."""
    return lambda c, s, r, g: validate_gate_set(c, s, r, g, max_size)


def validate_for_engine(
    circuits: Sequence[cirq.AbstractCircuit],
    sweeps: Sequence[cirq.Sweepable],
    repetitions: Union[int, List[int]],
    max_moments: int = MAX_MOMENTS,
    max_repetitions: int = MAX_TOTAL_REPETITIONS,
    max_duration_ns: int = 55000,
) -> None:
    """Validate a circuit and sweeps for sending to the Quantum Engine API."""
    _verify_reps(sweeps, repetitions, max_repetitions)
    _validate_depth(circuits, max_moments)
    _verify_measurements(circuits)


def create_engine_validator(
    max_moments: int = MAX_MOMENTS,
    max_repetitions: int = MAX_TOTAL_REPETITIONS,
    max_duration_ns: int = 55000,
) -> VALIDATOR_TYPE:
    """Creates a Callanle gate set validator with a set message size."""
    return lambda c, s, r: validate_for_engine(
        c, s, r, max_moments, max_repetitions, max_duration_ns
    )
