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
from typing import Callable, Sequence, Union
from google.protobuf import any_pb2

import cirq
from cirq_google.engine.validating_sampler import VALIDATOR_TYPE
from cirq_google.serialization.serializer import Serializer


MAX_MESSAGE_SIZE = 10_000_000
MAX_MOMENTS = 10000
MAX_TOTAL_REPETITIONS = 5_000_000

PROGRAM_VALIDATOR_TYPE = Callable[
    [Sequence[cirq.AbstractCircuit], Sequence[cirq.Sweepable], int, 'Serializer'], None
]


def _validate_depth(
    circuits: Sequence[cirq.AbstractCircuit], max_moments: int = MAX_MOMENTS
) -> None:
    """Validate that the depth of the circuit is not too long (too many moments)."""
    for circuit in circuits:
        if len(circuit) > max_moments:
            raise RuntimeError(f'Provided circuit exceeds the limit of {max_moments} moments.')


def _verify_reps(
    sweeps: Sequence[cirq.Sweepable],
    repetitions: Union[int, Sequence[int]],
    max_repetitions: int = MAX_TOTAL_REPETITIONS,
) -> None:
    """Verify that the total number of repetitions is under the limit."""
    total_reps = 0
    for idx, sweep in enumerate(sweeps):
        if not isinstance(repetitions, int):
            total_reps += len(list(cirq.to_resolvers(sweep))) * repetitions[idx]
        else:
            total_reps += len(list(cirq.to_resolvers(sweep))) * repetitions
    if total_reps > max_repetitions:
        raise RuntimeError(
            'No requested processors currently support the number of requested total repetitions.'
        )


def _verify_measurements(circuits):
    """Verify the circuit has appropriate measurements."""
    for circuit in circuits:
        has_measurement = any(
            isinstance(op.gate, cirq.MeasurementGate) for moment in circuit for op in moment
        )
        if not has_measurement:
            raise RuntimeError('Code must measure at least one qubit.')


# TODO(b/326267074): update validate methods to accept a `cirq.AbstractCircuit``
# instead of a `Sequence` since batching is no longer supported.
def validate_program(
    circuits: Sequence[cirq.AbstractCircuit],
    sweeps: Sequence[cirq.Sweepable],
    repetitions: int,
    serializer: Serializer,
    max_size: int = MAX_MESSAGE_SIZE,
) -> None:
    """Validate that the Program message size is below the maximum size limit.
    Args:
        circuits:  A sequence of  `cirq.Circuit` objects to validate.  For
          sweeps and runs, this will be a single circuit.  For batches,
          this will be a list of circuits.
        sweeps:  Parameters to run with each circuit.  The length of the
          sweeps sequence should be the same as the circuits argument.
        repetitions:  Number of repetitions to run with each sweep.
        serializer:  Serializer to use to serialize the circuits and sweeps.
        max_size:  proto size limit to check against.
    Raises:
        RuntimeError: if compiled proto is above the maximum size.
    """
    packed = any_pb2.Any()
    packed.Pack(serializer.serialize(circuits[0]))
    message_size = len(packed.SerializeToString())
    if message_size > max_size:
        raise RuntimeError("INVALID_PROGRAM: Program too long.")


def create_program_validator(max_size: int = MAX_MESSAGE_SIZE) -> PROGRAM_VALIDATOR_TYPE:
    """Creates a Callable program validator with a set message size.
    This validator can be used for a validator in `cg.ValidatingSampler`
    and can also be useful in generating 'engine emulators' by using
    `cg.SimulatedLocalProcessor` with this callable as a program_validator.
    Args:
        max_size:  proto size limit to check against.
    Returns: Callable to use in validation with the max_size already set.
    """

    def _validator(
        circuits: Sequence[cirq.AbstractCircuit],
        sweeps: Sequence[cirq.Sweepable],
        repetitions: int,
        serializer: Serializer,
    ):
        return validate_program(circuits, sweeps, repetitions, serializer, max_size)

    return _validator


def validate_for_engine(
    circuits: Sequence[cirq.AbstractCircuit],
    sweeps: Sequence[cirq.Sweepable],
    repetitions: Union[int, Sequence[int]],
    max_moments: int = MAX_MOMENTS,
    max_repetitions: int = MAX_TOTAL_REPETITIONS,
) -> None:
    """Validate a circuit and sweeps for sending to the Quantum Engine API.

    Args:
       circuits:  A sequence of  `cirq.Circuit` objects to validate.  For
          sweeps and runs, this will be a single circuit.
       sweeps:  Parameters to run with each circuit.  The length of the
          sweeps sequence should be the same as the circuits argument.
       repetitions:  Number of repetitions to run with each sweep.
       max_moments: Maximum number of moments to allow.
       max_repetitions: Maximum number of parameter sweep values allowed
           when summed across all sweeps.
       max_duration_ns:  Maximum duration of the circuit, in nanoseconds.
    """
    _verify_reps(sweeps, repetitions, max_repetitions)
    _validate_depth(circuits, max_moments)
    _verify_measurements(circuits)


def create_engine_validator(
    max_moments: int = MAX_MOMENTS, max_repetitions: int = MAX_TOTAL_REPETITIONS
) -> VALIDATOR_TYPE:
    """Creates a Callable gate set validator with a set message size.

    This validator can be used for a validator in `cg.ValidatingSampler`
    and can also be useful in generating 'engine emulators' by using
    `cg.SimulatedLocalProcessor` with this callable as a validator.

    Args:
        max_moments: Maximum number of moments to allow.
        max_repetitions: Maximum number of parameter sweep values allowed
            when summed across all sweeps and all batches.
    """

    def _validator(
        circuits: Sequence[cirq.AbstractCircuit],
        sweeps: Sequence[cirq.Sweepable],
        repetitions: Union[int, Sequence[int]],
    ):
        return validate_for_engine(circuits, sweeps, repetitions, max_moments, max_repetitions)

    return _validator
