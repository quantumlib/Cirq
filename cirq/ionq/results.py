# Copyright 2020 The Cirq Developers
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
"""Result types for the IonQ API."""

from typing import Dict

from cirq.value import digits


def _pretty_str_dict(value: dict, bit_count: int) -> str:
    """Pretty prints a dict, converting int dict values to bit strings."""
    strs = []
    for k, v in value.items():
        bits = ''.join(
            str(b)
            for b in digits.big_endian_int_to_bits(k, bit_count=bit_count))
        strs.append(f'{bits}: {v}')
    return '\n'.join(strs)


class QPUResult():
    """The results of running on an IonQ QPU."""

    def __init__(self, counts: dict, num_qubits: int):
        self._counts = counts
        self._num_qubits = num_qubits

    def num_qubits(self) -> int:
        """Returns the number of qubits the circuit was run on."""
        return self._num_qubits

    def repetitions(self) -> int:
        """Returns the number of times the circuit was run."""
        return sum(self._counts.values())

    def counts(self) -> Dict[int, int]:
        """Returns the raw counts of the measurement results.

        These are the results of measuring all qubits in the circuit.

        The key in this dictionary is the computational basis state measured.
        This is expressed in big-endian form, i.e. the bit string where the
        cirq.LineQubits are expressed in the order:
            (cirq.LineQubit(0), cirq.LineQubit(1), ..., cirq.LineQubit(n-1))

        The value is the number of times that corresponding bit string
        occurred.

        See `to_cirq_result` to convert to a `cirq.Result`.
        """
        return self._counts

    def __eq__(self, other):
        if not isinstance(other, QPUResult):
            return NotImplemented
        return (self._counts == other._counts and
                self._num_qubits == other._num_qubits)

    def __str__(self) -> str:
        return _pretty_str_dict(self._counts, self._num_qubits)

    # TODO: Convert his to cirq result objects.
    # https://github.com/quantumlib/Cirq/issues/3479


# TODO: Implement the sampler interface.
# https://github.com/quantumlib/Cirq/issues/3479
class SimulatorResult():
    """The results of running on an IonQ simulator.

    The IonQ simulator returns the probabilities of the different outcomes,
    not the raw wavefunction or samples.
    """

    def __init__(self, probabilities: dict, num_qubits: int):
        self._probabilities = probabilities
        self._num_qubits = num_qubits

    def num_qubits(self):
        """Returns the number of qubits the circuit was run on."""
        return self._num_qubits

    def probabilities(self):
        """Returns the probabilities of the measurement results.

        These are the probabilities arrising when measuring all qubits in the
        circuit.

        The key in this dictionary is the computational basis state measured.
        This is expressed in big-endian form, i.e. the bit string where the
        cirq.LineQubits are expressed in the order:
            (cirq.LineQubit(0), cirq.LineQubit(1), ..., cirq.LineQubit(n-1))

        The value is a float of the probability of this bit string occuring.
        """
        return self._probabilities

    def __eq__(self, other):
        if not isinstance(other, SimulatorResult):
            return NotImplemented
        return (self._probabilities == other._probabilities and
                self._num_qubits == other._num_qubits)

    def __str__(self) -> str:
        return _pretty_str_dict(self._probabilities, self._num_qubits)
