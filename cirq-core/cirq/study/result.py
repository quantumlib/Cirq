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

import abc
import collections
import io
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver

if TYPE_CHECKING:
    import cirq

T = TypeVar('T')
TMeasurementKey = Union[str, 'cirq.Qid', Iterable['cirq.Qid']]


def _tuple_of_big_endian_int(bit_groups: Iterable[Any]) -> Tuple[int, ...]:
    """Returns the big-endian integers specified by groups of bits.

    Args:
        bit_groups: Groups of descending bits, each specifying a big endian
            integer with the 1s bit at the end.

    Returns:
        A tuple containing the integer for each group.
    """
    return tuple(value.big_endian_bits_to_int(bits) for bits in bit_groups)


def _bitstring(vals: Iterable[Any]) -> str:
    str_list = [str(int(v)) for v in vals]
    separator = '' if all(len(s) == 1 for s in str_list) else ' '
    return separator.join(str_list)


def _keyed_repeated_bitstrings(vals: Mapping[str, np.ndarray]) -> str:
    keyed_bitstrings = []
    for key in sorted(vals.keys()):
        reps = vals[key]
        n = 0 if len(reps) == 0 else len(reps[0])
        all_bits = ', '.join(_bitstring(reps[:, i]) for i in range(n))
        keyed_bitstrings.append(f'{key}={all_bits}')
    return '\n'.join(keyed_bitstrings)


def _key_to_str(key: TMeasurementKey) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, ops.Qid):
        return str(key)
    return ','.join(str(q) for q in key)


class Result(abc.ABC):
    """The results of multiple executions of a circuit with fixed parameters."""

    @property
    @abc.abstractmethod
    def params(self) -> 'cirq.ParamResolver':
        """A ParamResolver of settings used for this result."""

    @property
    @abc.abstractmethod
    def measurements(self) -> Mapping[str, np.ndarray]:
        """A mapping from measurement gate key to measurement results.

        The value for each key is a 2-D array of booleans, with the first index
        running over the repetitions, and the second index running over the
        qubits for the corresponding measurements.
        """

    @property
    @abc.abstractmethod
    def records(self) -> Mapping[str, np.ndarray]:
        """A mapping from measurement key to measurement records.

        The value for each key is a 3-D array of booleans, with the first index
        running over circuit repetitions, the second index running over instances
        of the measurement key in the circuit, and the third index running over
        the qubits for the corresponding measurements.
        """

    @property
    @abc.abstractmethod
    def data(self) -> pd.DataFrame:
        """Measurements converted to a pandas dataframe.

        The rows in the returned data frame correspond to repetitions of the
        circuit, and the columns correspond to measurement keys, where each
        element is a big-endian integer representation of measurement outcomes
        for the measurement key in that repetition. To convert these ints to
        bits see `cirq.big_endian_int_to_bits` and similar functions.
        """

    @staticmethod
    def dataframe_from_measurements(measurements: Mapping[str, np.ndarray]) -> pd.DataFrame:
        """Converts the given measurements to a pandas dataframe.

        This can be used by subclasses as a default implementation for the data
        property. Note that subclasses should typically memoize the result to
        avoid recomputing.
        """
        # Convert to a DataFrame with columns as measurement keys, rows as
        # repetitions and a big endian integer for individual measurements.
        converted_dict = {}
        for key, bitstrings in measurements.items():
            _, n = bitstrings.shape
            dtype = object if n > 63 else np.int64
            basis = 2 ** np.arange(n, dtype=dtype)[::-1]
            converted_dict[key] = np.sum(basis * bitstrings, axis=1)

        # Use objects to accomodate more than 64 qubits if needed.
        dtype = object if any(bs.shape[1] > 63 for _, bs in measurements.items()) else np.int64
        return pd.DataFrame(converted_dict, dtype=dtype)

    @property
    def repetitions(self) -> int:
        if not self.records:
            return 0
        # Get the length quickly from one of the keyed results.
        return len(next(iter(self.records.values())))

    def multi_measurement_histogram(
        self,
        *,  # Forces keyword args.
        keys: Iterable[TMeasurementKey],
        fold_func: Callable[[Tuple], T] = cast(Callable[[Tuple], T], _tuple_of_big_endian_int),
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
        fixed_keys = tuple(_key_to_str(key) for key in keys)
        samples: Iterable[Any] = zip(*(self.measurements[sub_key] for sub_key in fixed_keys))
        if len(fixed_keys) == 0:
            samples = [()] * self.repetitions
        c: collections.Counter = collections.Counter()
        for sample in samples:
            c[fold_func(sample)] += 1
        return c

    def histogram(
        self,
        *,  # Forces keyword args.
        key: TMeasurementKey,
        fold_func: Callable[[Tuple], T] = cast(Callable[[Tuple], T], value.big_endian_bits_to_int),
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
        return self.multi_measurement_histogram(keys=[key], fold_func=lambda e: fold_func(e[0]))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Result):
            return NotImplemented
        return (
            self.records.keys() == other.records.keys()
            and all(np.array_equal(self.records[k], other.records[k]) for k in self.records)
            and self.params == other.params
        )

    def __add__(self, other: 'cirq.Result') -> 'cirq.Result':
        if not isinstance(other, Result):
            return NotImplemented
        if self.params != other.params:
            raise ValueError(
                f'Cannot add results with different parameters: {self.params} != {other.params}'
            )
        shape = {k: v.shape[1:] for k, v in self.records.items()}
        other_shape = {k: v.shape[1:] for k, v in other.records.items()}
        if shape != other_shape:
            raise ValueError(
                f'Cannot add results with different measurement shapes: {shape} != {other_shape}'
            )
        all_records: Dict[str, np.ndarray] = {}
        for key in other.records:
            all_records[key] = np.append(self.records[key], other.records[key], axis=0)
        return ResultDict(params=self.params, records=all_records)


class ResultDict(Result):
    """A Result created from a dict mapping measurement keys to measured values.

    Stores results of executing a circuit for multiple repetitions with one
    fixed set of parameters. The values for each measurement key are stored as a
    2D numpy array. The first (row) index in each array is the repetition
    number, and the second (column) index is the qubit.

    Attributes:
        params: A ParamResolver of settings used when sampling result.
    """

    def __init__(
        self,
        *,  # Forces keyword args.
        params: Optional[resolver.ParamResolver] = None,
        measurements: Optional[Mapping[str, np.ndarray]] = None,
        records: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        """Inits Result.

        Args:
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. The value for each key is a 2-D array of booleans,
                with the first index running over the repetitions, and the
                second index running over the qubits for the corresponding
                measurements.
            records: A dictionary from measurement gate key to measurement
                results. The value for each key is a 3D array of booleans,
                with the first index running over the repetitions, the second
                index running over "instances" of that key in the circuit, and
                the last index running over the qubits for the corresponding
                measurements.
        """
        if params is None:
            params = resolver.ParamResolver({})
        if measurements is None and records is None:
            # For backwards compatibility, allow constructing with None.
            measurements = {}
            records = {}
        self._params = params
        self._measurements = measurements
        self._records = records
        self._data: Optional[pd.DataFrame] = None

    @property
    def params(self) -> 'cirq.ParamResolver':
        return self._params

    @property
    def measurements(self) -> Mapping[str, np.ndarray]:
        if self._measurements is None:
            assert self._records is not None
            self._measurements = {}
            for key, data in self._records.items():
                reps, instances, qubits = data.shape
                if instances != 1:
                    raise ValueError('Cannot extract 2D measurements for repeated keys')
                self._measurements[key] = data.reshape((reps, qubits))
        return self._measurements

    @property
    def records(self) -> Mapping[str, np.ndarray]:
        if self._records is None:
            assert self._measurements is not None
            self._records = {
                key: data[:, np.newaxis, :] for key, data in self._measurements.items()
            }
        return self._records

    @property
    def repetitions(self) -> int:
        if self._records is not None:
            if not self._records:
                return 0
            # Get the length quickly from one of the keyed results.
            return len(next(iter(self._records.values())))
        else:
            if not self._measurements:
                return 0
            # Get the length quickly from one of the keyed results.
            return len(next(iter(self._measurements.values())))

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            self._data = self.dataframe_from_measurements(self.measurements)
        return self._data

    def _record_dict_repr(self):
        """Helper function for use in __repr__ to display the records field."""
        return '{' + ', '.join(f'{k!r}: {proper_repr(v)}' for k, v in self.records.items()) + '}'

    def __repr__(self) -> str:
        return f'cirq.ResultDict(params={self.params!r}, records={self._record_dict_repr()})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Output to show in ipython and Jupyter notebooks."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('ResultDict(...)')
        else:
            p.text(str(self))

    def __str__(self) -> str:
        return _keyed_repeated_bitstrings(self.measurements)

    def _json_dict_(self):
        packed_records = {}
        for key, digits in self.records.items():
            packed_digits, binary = _pack_digits(digits)
            packed_records[key] = {
                'packed_digits': packed_digits,
                'binary': binary,
                'dtype': digits.dtype.name,
                'shape': digits.shape,
            }
        return {'params': self.params, 'records': packed_records}

    @classmethod
    def _from_packed_records(cls, records, **kwargs):
        """Helper function for `_from_json_dict_` to construct from packed records."""
        return cls(records={key: _unpack_digits(**val) for key, val in records.items()}, **kwargs)

    @classmethod
    def _from_json_dict_(cls, params, **kwargs):
        if 'measurements' in kwargs:
            measurements = kwargs['measurements']
            return cls(
                params=params,
                measurements={key: _unpack_digits(**val) for key, val in measurements.items()},
            )
        return cls._from_packed_records(params=params, records=kwargs['records'])


def _pack_digits(digits: np.ndarray, pack_bits: str = 'auto') -> Tuple[str, bool]:
    """Returns a string of packed digits and a boolean indicating whether the
    digits were packed as binary values.

    Args:
        digits: A numpy array.
        pack_bits: If 'auto' (the default), automatically pack binary digits
            using `np.packbits` to save space. If 'never', do not pack binary
            digits. If 'force', use `np.packbits` without checking for
            compatibility.

    Raises:
        ValueError: If `pack_bits` is not `auto`, `force`, or `never`.
    """
    # If digits are binary, pack them better to save space

    if pack_bits == 'force':
        return _pack_bits(digits), True
    if pack_bits not in ['auto', 'never']:
        raise ValueError("Please set `pack_bits` to 'auto', 'force', or 'never'.")
        # Do error checking here, otherwise the following logic will work
        # for both "auto" and "never".

    if pack_bits == 'auto' and np.array_equal(digits, digits.astype(np.bool_)):
        return _pack_bits(digits.astype(np.bool_)), True

    buffer = io.BytesIO()
    np.save(buffer, digits, allow_pickle=False)
    buffer.seek(0)
    packed_digits = buffer.read().hex()
    buffer.close()
    return packed_digits, False


def _pack_bits(bits: np.ndarray) -> str:
    return np.packbits(bits).tobytes().hex()


def _unpack_digits(
    packed_digits: str, binary: bool, dtype: Union[None, str], shape: Union[None, Sequence[int]]
) -> np.ndarray:
    """The opposite of `_pack_digits`.

    Args:
        packed_digits: The hex-encoded string representing a numpy array of
            digits. This is the first return value of `_pack_digits`.
        binary: Whether the digits have been packed as binary. This is the
            second return value of `_pack_digits`.
        dtype: If `binary` is True, you must also provide the datatype of the
            array. Otherwise, dtype information is contained within the hex
            string.
        shape: If `binary` is True, you must also provide the shape of the
            array. Otherwise, shape information is contained within the hex
            string.
    """
    if binary:
        dtype = cast(str, dtype)
        shape = cast(Sequence[int], shape)
        return _unpack_bits(packed_digits, dtype, shape)

    buffer = io.BytesIO()
    buffer.write(bytes.fromhex(packed_digits))
    buffer.seek(0)
    digits = np.load(buffer, allow_pickle=False)
    buffer.close()
    return digits


def _unpack_bits(packed_bits: str, dtype: str, shape: Sequence[int]) -> np.ndarray:
    bits_bytes = bytes.fromhex(packed_bits)
    bits = np.unpackbits(np.frombuffer(bits_bytes, dtype=np.uint8))
    return bits[: np.prod(shape).item()].reshape(shape).astype(dtype)
