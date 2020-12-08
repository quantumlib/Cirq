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

import collections

from typing import Dict, Counter, Optional, Sequence

from cirq.value import digits


class QPUResult:
    """The results of running on an IonQ QPU."""

    def __init__(
        self, counts: Dict[int, int], num_qubits: int, measurement_dict: Dict[str, Sequence[int]]
    ):
        self._counts = counts
        self._num_qubits = num_qubits
        self._measurement_dict = measurement_dict

    def num_qubits(self) -> int:
        """Returns the number of qubits the circuit was run on."""
        return self._num_qubits

    def repetitions(self) -> int:
        """Returns the number of times the circuit was run."""
        return sum(self._counts.values())

    def counts(self, key: Optional[str] = None) -> Counter[int]:
        """Returns the raw counts of the measurement results.

        If a key parameter is supplied, these are the counts for the measurement results for
        the qubits measured by the measurement gate with that key.  If no key is given, these
        are the measurement results from measuring all qubits in the circuit.

        The key in the returned dictionary is the computational basis state measured for the
        qubits that have been measured.  This is expressed in big-endian form. For example, if
        no measurement key is supplied and all qubits are measured, the key in this returned dict
        has a bit string where the `cirq.LineQubit`s are expressed in the order:
            (cirq.LineQubit(0), cirq.LineQubit(1), ..., cirq.LineQubit(n-1))
        In the case where only `r` qubits are measured corresponding to targets t_0, t_1,...t_{r-1},
        the bit string corresponds to the order
            (cirq.LineQubit(t_0), cirq.LineQubit(t_1), ... cirq.LineQubit(t_{r-1}))

        The value is the number of times that corresponding bit string occurred.

        See `to_cirq_result` to convert to a `cirq.Result`.
        """
        if key is None:
            return collections.Counter(self._counts)
        if not key in self._measurement_dict:
            raise ValueError(
                f'Measurement key {key} is not a key for a measurement gate in the'
                'circuit that produced these results.'
            )
        targets = self._measurement_dict[key]
        result: Counter[int] = collections.Counter()
        for value, count in self._counts.items():
            bits = [(value >> (self.num_qubits() - target - 1)) & 1 for target in targets]
            bit_value = sum(bit * (1 << i) for i, bit in enumerate(bits[::-1]))
            result[bit_value] += count
        return result

    def measurement_dict(self) -> Dict[str, Sequence[int]]:
        """Returns a map from measurement keys to target qubit indices for this measurement."""
        return self._measurement_dict

    def __eq__(self, other):
        if not isinstance(other, QPUResult):
            return NotImplemented
        return (
            self._counts == other._counts
            and self._num_qubits == other._num_qubits
            and self._measurement_dict == other._measurement_dict
        )

    def __str__(self) -> str:
        return _pretty_str_dict(self._counts, self._num_qubits)

    # TODO: Convert his to cirq result objects.
    # https://github.com/quantumlib/Cirq/issues/3479


# TODO: Implement the sampler interface.
# https://github.com/quantumlib/Cirq/issues/3479
class SimulatorResult:
    """The results of running on an IonQ simulator.

    The IonQ simulator returns the probabilities of the different outcomes, not the raw state
    vector or samples.
    """

    def __init__(
        self,
        probabilities: Dict[int, float],
        num_qubits: int,
        measurement_dict: Dict[str, Sequence[int]],
    ):
        self._probabilities = probabilities
        self._num_qubits = num_qubits
        self._measurement_dict = measurement_dict

    def num_qubits(self) -> int:
        """Returns the number of qubits the circuit was run on."""
        return self._num_qubits

    def probabilities(self, key: Optional[str] = None) -> Dict[int, float]:
        """Returns the probabilities of the measurement results.

        If a key parameter is supplied, these are the probabilities for the measurement results for
        the qubits measured by the measurement gate with that key.  If no key is given, these
        are the measurement results from measuring all qubits in the circuit.

        The key in the returned dictionary is the computational basis state measured for the
        qubits that have been measured.  This is expressed in big-endian form. For example, if
        no measurement key is supplied and all qubits are measured, the key in this returned dict
        has a bit string where the `cirq.LineQubit`s are expressed in the order:
            (cirq.LineQubit(0), cirq.LineQubit(1), ..., cirq.LineQubit(n-1))
        In the case where only `r` qubits are measured corresponding to targets t_0, t_1,...t_{r-1},
        the bit string corresponds to the order
            (cirq.LineQubit(t_0), cirq.LineQubit(t_1), ... cirq.LineQubit(t_{r-1}))

        The value is the probability that the corresponding bit string occurred.

        See `to_cirq_result` to convert to a `cirq.Result`.
        """
        if key is None:
            return self._probabilities
        if not key in self._measurement_dict:
            raise ValueError(
                f'Measurement key {key} is not a key for a measurement gate in the'
                'circuit that produced these results.'
            )
        targets = self._measurement_dict[key]
        result: Dict[int, float] = dict()
        for value, probability in self._probabilities.items():
            bits = [(value >> (self.num_qubits() - target - 1)) & 1 for target in targets]
            bit_value = sum(bit * (1 << i) for i, bit in enumerate(bits[::-1]))
            if bit_value in result:
                result[bit_value] += probability
            else:
                result[bit_value] = probability
        return result

    def measurement_dict(self) -> Dict[str, Sequence[int]]:
        """Returns a map from measurement keys to target qubit indices for this measurement."""
        return self._measurement_dict

    def __eq__(self, other):
        if not isinstance(other, SimulatorResult):
            return NotImplemented
        return (
            self._probabilities == other._probabilities
            and self._num_qubits == other._num_qubits
            and self._measurement_dict == other._measurement_dict
        )

    def __str__(self) -> str:
        return _pretty_str_dict(self._probabilities, self._num_qubits)


def _pretty_str_dict(value: dict, bit_count: int) -> str:
    """Pretty prints a dict, converting int dict values to bit strings."""
    strs = []
    for k, v in value.items():
        bits = ''.join(str(b) for b in digits.big_endian_int_to_bits(k, bit_count=bit_count))
        strs.append(f'{bits}: {v}')
    return '\n'.join(strs)
