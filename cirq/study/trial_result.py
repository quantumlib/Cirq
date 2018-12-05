# Copyright 2018 The Cirq Developers
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

"""Defines trial results."""

from typing import Iterable, Callable, Tuple, TypeVar, Dict, Any

import collections
import numpy as np

from cirq import value
from cirq.study import resolver

T = TypeVar('T')


def _tuple_of_big_endian_int(bit_groups: Tuple[np.ndarray, ...]
                             ) -> Tuple[int, ...]:
    """Returns the big-endian integers specified by groups of bits.

    Args:
        bit_groups: Groups of descending bits, each specifying a big endian
            integer with the 1s bit at the end.

    Returns:
        A tuple containing the integer for each group.
    """
    return tuple(_big_endian_int(bits) for bits in bit_groups)


def _big_endian_int(bits: np.ndarray) -> int:
    """Returns the big-endian integer specified by the given bits.

    For example, [True, False, False, True, False] becomes binary 10010 which
    is 18 in decimal.

    Args:
        bits: Descending bits of the integer, with the 1s bit at the end.

    Returns:
        The integer.
    """
    result = 0
    for e in bits:
        result <<= 1
        if e:
            result |= 1
    return result


def _bitstring(vals: Iterable[Any]) -> str:
    return ''.join('1' if v else '0' for v in vals)


def _keyed_repeated_bitstrings(vals: Dict[str, np.ndarray]
                               ) -> str:
    keyed_bitstrings = []
    for key in sorted(vals.keys()):
        reps = vals[key]
        n = 0 if len(reps) == 0 else len(reps[0])
        all_bits = ', '.join([_bitstring(reps[:, i])
                              for i in range(n)])
        keyed_bitstrings.append('{}={}'.format(key, all_bits))
    return '\n'.join(keyed_bitstrings)


@value.value_equality(unhashable=True)
class TrialResult:
    """The results of multiple executions of a circuit with fixed parameters.

    Attributes:
        params: A ParamResolver of settings used when sampling result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are stored in a 2-dimensional
            numpy array, the first dimension corresponding to the repetition
            and the second to the actual boolean measurement results (ordered
            by the qubits being measured.)
        repetitions: The number of times a circuit was sampled to get these
            results.
    """

    def __init__(self, *,  # Forces keyword args.
                 params: resolver.ParamResolver,
                 measurements: Dict[str, np.ndarray],
                 repetitions: int) -> None:
        """
        Args:
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. The value for each key is a 2-D array of booleans,
                with the first index running over the repetitions, and the
                second index running over the qubits for the corresponding
                measurements.
            repetitions: The number of times the circuit was sampled.
        """
        self.params = params
        self.measurements = measurements
        self.repetitions = repetitions

    # Reason for 'type: ignore': https://github.com/python/mypy/issues/5273
    def multi_measurement_histogram(  # type: ignore
            self, *,  # Forces keyword args.
            keys: Iterable[str],
            fold_func: Callable[[Tuple[np.ndarray, ...]],
                                T] = _tuple_of_big_endian_int
    ) -> collections.Counter:
        """Counts the number of times combined measurement results occurred.

        This is a more general version of the 'histogram' method. Instead of
        only counting how often results occurred for one specific measurement,
        this method tensors multiple measurement results together and counts
        how often the combined results occurred.

        For example, suppose that:

            - fold_func is not specified
            - keys=['abc', 'd']
            - the measurement with key 'abc' measures qubits a, b, and c.
            - the measurement with key 'd' measures qubit d.
            - the circuit was sampled 3 times.
            - the sampled measurement values were:
                1. a=1 b=0 c=0 d=0
                2. a=0 b=1 c=0 d=1
                3. a=1 b=0 c=0 d=0

        Then the counter returned by this method will be:

            collections.Counter({
                (0b100, 0): 2,
                (0b010, 1): 1
            })


        Where '0b100' is binary for '4' and '0b010' is binary for '2'. Notice
        that the bits are combined in a big-endian way by default, with the
        first measured qubit determining the highest-value bit.

        Args:
            fold_func: A function used to convert sampled measurement results
                into countable values. The input is a tuple containing the
                list of bits measured by each measurement specified by the
                keys argument. If this argument is not specified, it defaults
                to returning tuples of integers, where each integer is the big
                endian interpretation of the bits a measurement sampled.
            keys: Keys of measurements to include in the histogram.

        Returns:
            A counter indicating how often measurements sampled various
            results.
        """
        fixed_keys = tuple(keys)
        samples = zip(*[self.measurements[sub_key]
                        for sub_key in fixed_keys])  # type: Iterable[Any]
        if len(fixed_keys) == 0:
            samples = [()] * self.repetitions
        c = collections.Counter()  # type: collections.Counter
        for sample in samples:
            c[fold_func(sample)] += 1
        return c

    # Reason for 'type: ignore': https://github.com/python/mypy/issues/5273
    def histogram(self,  # type: ignore
                  *,  # Forces keyword args.
                  key: str,
                  fold_func: Callable[[np.ndarray], T] = _big_endian_int
                  ) -> collections.Counter:
        """Counts the number of times a measurement result occurred.

        For example, suppose that:

            - fold_func is not specified
            - key='abc'
            - the measurement with key 'abc' measures qubits a, b, and c.
            - the circuit was sampled 3 times.
            - the sampled measurement values were:
                1. a=1 b=0 c=0
                2. a=0 b=1 c=0
                3. a=1 b=0 c=0

        Then the counter returned by this method will be:

            collections.Counter({
                0b100: 2,
                0b010: 1
            })

        Where '0b100' is binary for '4' and '0b010' is binary for '2'. Notice
        that the bits are combined in a big-endian way by default, with the
        first measured qubit determining the highest-value bit.

        Args:
            positional_args: Never specified. Forces keyword arguments.
            key: Keys of measurements to include in the histogram.
            fold_func: A function used to convert a sampled measurement result
                into a countable value. The input is a list of bits sampled
                together by a measurement. If this argument is not specified,
                it defaults to interpreting the bits as a big endian
                integer.

        Returns:
            A counter indicating how often a measurement sampled various
            results.
        """
        return self.multi_measurement_histogram(
            keys=[key],
            fold_func=lambda e: fold_func(e[0]))

    def __repr__(self):
        return ('cirq.TrialResult(params={!r}, '
                'repetitions={!r}, '
                'measurements={!r})').format(self.params,
                                             self.repetitions,
                                             self.measurements)

    def __str__(self):
        return _keyed_repeated_bitstrings(self.measurements)

    def _value_equality_values_(self):
        return self.measurements, self.repetitions, self.params
