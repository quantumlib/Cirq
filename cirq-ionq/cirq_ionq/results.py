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
"""Result types for the IonQ API."""

import collections
from typing import Counter, Dict, List, Optional, Sequence

import numpy as np

import cirq


class QPUResult:
    """The results of running on an IonQ QPU."""

    def __init__(
        self, counts: Dict[int, int], num_qubits: int, measurement_dict: Dict[str, Sequence[int]]
    ):
        # We require a consistent ordering, and here we use bitvector as such.
        # OrderedDict can be removed in python 3.7, where it is part of the contract.
        self._counts = collections.OrderedDict(sorted(counts.items()))
        self._num_qubits = num_qubits
        self._measurement_dict = measurement_dict
        self._repetitions = sum(self._counts.values())

    def num_qubits(self) -> int:
        """Returns the number of qubits the circuit was run on."""
        return self._num_qubits

    def repetitions(self) -> int:
        """Returns the number of times the circuit was run."""
        return self._repetitions

    def ordered_results(self, key: Optional[str] = None) -> List[int]:
        """Returns a list of arbitrarily but consistently ordered results as big endian ints.

        If a key parameter is supplied, these are the counts for the measurement results for
        the qubits measured by the measurement gate with that key.  If no key is given, these
        are the measurement results from measuring all qubits in the circuit.

        The value in the returned list is the computational basis state measured for the
        qubits that have been measured.  This is expressed in big-endian form. For example, if
        no measurement key is supplied and all qubits are measured, each entry in this returned dict
        has a bit string where the `cirq.LineQubit`s are expressed in the order:
            (cirq.LineQubit(0), cirq.LineQubit(1), ..., cirq.LineQubit(n-1))
        In the case where only `r` qubits are measured corresponding to targets t_0, t_1,...t_{r-1},
        the bit string corresponds to the order
            (cirq.LineQubit(t_0), cirq.LineQubit(t_1), ... cirq.LineQubit(t_{r-1}))
        """

        if key is not None and not key in self._measurement_dict:
            raise ValueError(
                f'Measurement key {key} is not a key for a measurement gate in the'
                'circuit that produced these results.'
            )
        targets = self._measurement_dict[key] if key is not None else range(self.num_qubits())
        result: List[int] = []
        for value, count in self._counts.items():
            bits = [(value >> (self.num_qubits() - target - 1)) & 1 for target in targets]
            bit_value = sum(bit * (1 << i) for i, bit in enumerate(bits[::-1]))
            result.extend([bit_value] * count)
        return result

    def counts(self, key: Optional[str] = None) -> Counter[int]:
        """Returns the processed counts of the measurement results.

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
        result: Counter[int] = collections.Counter()
        result.update([bit_value for bit_value in self.ordered_results(key)])
        return result

    def measurement_dict(self) -> Dict[str, Sequence[int]]:
        """Returns a map from measurement keys to target qubit indices for this measurement."""
        return self._measurement_dict

    def to_cirq_result(self, params: Optional[cirq.ParamResolver] = None) -> cirq.Result:
        """Returns a `cirq.Result` for these results.

        `cirq.Result` contains a less dense representation of results than that returned by
        the IonQ API.  Typically these results are also ordered by when they were run, though
        that contract is implicit.  Because the IonQ API does not retain that ordering information,
        the order of these `cirq.Result` objects should *not* be interpetted as representing the
        order in which the circuit was repeated. Correlations between measurements keys are
        preserved.

        Args:
            params: The `cirq.ParamResolver` used to generate these results.

        Returns:
            The `cirq.Result` for these results.

        Raises:
            ValueError: If the circuit used to produce this result had no measurement gates
                (and hence no measurement keys).
        """
        if len(self.measurement_dict()) == 0:
            raise ValueError(
                'Can convert to cirq results only if the circuit had measurement gates '
                'with measurement keys.'
            )
        measurements = {}
        for key, targets in self.measurement_dict().items():
            qpu_results = self.ordered_results(key)
            measurements[key] = np.array(
                list(cirq.big_endian_int_to_bits(x, bit_count=len(targets)) for x in qpu_results)
            )
        return cirq.ResultDict(params=params or cirq.ParamResolver({}), measurements=measurements)

    def __eq__(self, other):
        if not isinstance(other, QPUResult):
            return NotImplemented
        return (
            self._counts == other._counts
            and self._num_qubits == other._num_qubits
            and self._measurement_dict == other._measurement_dict
            and self._repetitions == other._repetitions
        )

    def __str__(self) -> str:
        return _pretty_str_dict(self._counts, self._num_qubits)


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
        repetitions: int,
    ):
        self._probabilities = probabilities
        self._num_qubits = num_qubits
        self._measurement_dict = measurement_dict
        self._repetitions = repetitions

    def num_qubits(self) -> int:
        """Returns the number of qubits the circuit was run on."""
        return self._num_qubits

    def repetitions(self) -> int:
        """Returns the number of times the circuit was run.

        For IonQ API simulations this is used when generating `cirq.Result`s from `to_cirq_result`.
        The sampling is not done on the IonQ API but is done in `to_cirq_result`.
        """
        return self._repetitions

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

    def to_cirq_result(
        self,
        params: Optional[cirq.ParamResolver] = None,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        override_repetitions=None,
    ) -> cirq.Result:
        """Samples from the simulation probability result, producing a `cirq.Result`.

        The IonQ simulator returns the probabilities of different bitstrings. This converts such
        a representation to a randomly generated sample from the simulator. Note that it does this
        on every subsequent call of this method, so repeated calls do not produce the same
        `cirq.Result`s. When a job was created by the IonQ API, it had a number of repetitions and
        this is used, unless `override_repetitions` is set here.

        Args:
            params: Any parameters which were used to generated this result.
            seed: What to use for generating the randomness. If None, then `np.random` is used.
                If an integer, `np.random.RandomState(seed) is used. Otherwise if another
                randomness generator is used, it will be used.
            override_repetitions: Repetitions were supplied when the IonQ API ran the simulation,
                but different repetitions can be supplied here and will override.

        Returns:
            A `cirq.Result` corresponding to a sample from the probability distribution returned
            from the simulator.

        Raises:
            ValueError: If the circuit used to produce this result had no measurement gates
                (and hence no measurement keys).
        """
        if len(self.measurement_dict()) == 0:
            raise ValueError(
                'Can convert to cirq results only if the circuit had measurement gates '
                'with measurement keys.'
            )
        rand = cirq.value.parse_random_state(seed)
        measurements = {}
        values, weights = zip(*list(self.probabilities().items()))
        indices = rand.choice(
            range(len(values)), p=weights, size=override_repetitions or self.repetitions()
        )
        rand_values = np.array(values)[indices]
        for key, targets in self.measurement_dict().items():
            bits = [
                [(value >> (self.num_qubits() - target - 1)) & 1 for target in targets]
                for value in rand_values
            ]
            measurements[key] = np.array(bits)
        return cirq.ResultDict(params=params or cirq.ParamResolver({}), measurements=measurements)

    def __eq__(self, other):
        if not isinstance(other, SimulatorResult):
            return NotImplemented
        return (
            self._probabilities == other._probabilities
            and self._num_qubits == other._num_qubits
            and self._measurement_dict == other._measurement_dict
            and self._repetitions == other._repetitions
        )

    def __str__(self) -> str:
        return _pretty_str_dict(self._probabilities, self._num_qubits)


def _pretty_str_dict(value: dict, bit_count: int) -> str:
    """Pretty prints a dict, converting int dict values to bit strings."""
    strs = []
    for k, v in value.items():
        bits = ''.join(str(b) for b in cirq.big_endian_int_to_bits(k, bit_count=bit_count))
        strs.append(f'{bits}: {v}')
    return '\n'.join(strs)
