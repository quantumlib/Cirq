# Copyright 2018 Google LLC
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

"""Gates that can be directly described to the API, without decomposition."""

import abc
from typing import Union

from cirq.api.google.v1 import operations_pb2
from cirq.ops import gate_features, raw_types


class NativeGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate with a known mechanism for encoding into API protos."""

    @abc.abstractmethod
    def to_proto(self, *qubits) -> operations_pb2.Operation:
        raise NotImplementedError()


class ParameterizedValue:
    """A constant plus the runtime value of a parameter with a given key."""

    def __new__(cls, key: str = '', val: float = 0):
        """Constructs a ParameterizedValue representing val + param(key).

        Args:
            val: A constant offset.
            key: The name of a parameter. If this is the empty string, then no
                parameter will be used.

        Returns:
            Just val if key is empty, otherwise a new ParameterizedValue.
        """
        if key == '':
            return val
        return super().__new__(cls)

    def __init__(self, key: str = '', val: float = 0):
        """Initializes a ParameterizedValue representing val + param(key).

        Args:
            val: A constant offset.
            key: The name of a parameter. Because of the implementation of new,
                this will never be the empty string.
        """
        self.val = val
        self.key = key

    def __str__(self):
        if self.key == '':
            return repr(self.val)
        if self.val == 0:
            return 'param({})'.format(repr(self.key))
        return '{} + param({})'.format(repr(self.val), repr(self.key))

    def __repr__(self):
        return 'ParameterizedValue({}, {})'.format(repr(self.val),
                                                   repr(self.key))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.key == other.key and self.val == other.val

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ParameterizedValue, self.val, self.key))

    def __add__(self, other: float) -> 'ParameterizedValue':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return ParameterizedValue(self.key, self.val + other)

    def __radd__(self, other: float) -> 'ParameterizedValue':
        return self.__add__(other)

    def __sub__(self, other: float) -> 'ParameterizedValue':
        if not isinstance(other, (int, float)):
            return NotImplemented
        return ParameterizedValue(self.key, self.val - other)

    @staticmethod
    def val_of(val: Union['ParameterizedValue', float]):
        if isinstance(val, ParameterizedValue):
            return float(val.val)
        return float(val)

    @staticmethod
    def key_of(val: Union['ParameterizedValue', float]):
        if isinstance(val, ParameterizedValue):
            return val.key
        return ''


class MeasurementGate(NativeGate, gate_features.AsciiDiagrammableGate):
    """Indicates that a qubit should be measured, and where the result goes."""

    def __init__(self, key: str = ''):
        self.key = key

    def to_proto(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')

        q = qubits[0]
        op = operations_pb2.Operation()
        op.measurement.target.x = q.x
        op.measurement.target.y = q.y
        op.measurement.key = self.key
        return op

    def ascii_wire_symbols(self):
        return 'M',

    def __repr__(self):
        return 'MeasurementGate({})'.format(repr(self.key))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.key == other.key

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((MeasurementGate, self.key))


class Exp11Gate(NativeGate, gate_features.PhaseableGate):
    """A two-qubit interaction that phases the amplitude of the 11 state."""

    def __init__(self, *positional_args,
                 half_turns: Union[ParameterizedValue, float]=1):
        assert not positional_args
        self.half_turns = _canonicalize_turns(half_turns)

    def phase_by(self, phase_turns, qubit_index):
        return self

    def to_proto(self, *qubits):
        if len(qubits) != 2:
            raise ValueError('Wrong number of qubits.')

        p, q = qubits
        op = operations_pb2.Operation()
        op.exp_11.target1.x = p.x
        op.exp_11.target1.y = p.y
        op.exp_11.target2.x = q.x
        op.exp_11.target2.y = q.y
        op.exp_11.half_turns.raw = ParameterizedValue.val_of(self.half_turns)
        op.exp_11.half_turns.parameter_key = ParameterizedValue.key_of(
            self.half_turns)
        return op

    def ascii_wire_symbols(self):
        return 'Z', 'Z'

    def ascii_exponent(self):
        return self.half_turns

    def __repr__(self):
        return 'ParameterizedCZGate(half_turns={})'.format(
            repr(self.half_turns))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((Exp11Gate, self.half_turns))


class ExpWGate(NativeGate,
               gate_features.AsciiDiagrammableGate,
               gate_features.PhaseableGate):
    """A rotation around an axis in the XY plane of the Bloch sphere."""

    def __init__(self, *positional_args,
                 half_turns: Union[ParameterizedValue, float]=1,
                 axis_half_turns: Union[ParameterizedValue, float]=0):
        assert not positional_args
        self.half_turns = _canonicalize_turns(half_turns)
        self.axis_half_turns = _canonicalize_turns(axis_half_turns)

        if (not isinstance(self.half_turns, ParameterizedValue) and
                not isinstance(self.axis_half_turns, ParameterizedValue) and
                not 0 <= self.axis_half_turns < 1):
            # Canonicalize to negative rotation around positive axis.
            self.half_turns = _canonicalize_turns(-self.half_turns)
            self.axis_half_turns = _canonicalize_turns(
                self.axis_half_turns + 1)

    def to_proto(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')

        q = qubits[0]
        op = operations_pb2.Operation()
        op.exp_w.target.x = q.x
        op.exp_w.target.y = q.y
        op.exp_w.axis_half_turns.raw = ParameterizedValue.val_of(
            self.axis_half_turns)
        op.exp_w.axis_half_turns.parameter_key = ParameterizedValue.key_of(
            self.axis_half_turns)
        op.exp_w.half_turns.raw = ParameterizedValue.val_of(self.half_turns)
        op.exp_w.half_turns.parameter_key = ParameterizedValue.key_of(
            self.half_turns)
        return op

    def phase_by(self, phase_turns, qubit_index):
        return ExpWGate(
            half_turns=self.half_turns,
            axis_half_turns=self.axis_half_turns + phase_turns * 2)

    def ascii_wire_symbols(self):
        if self.axis_half_turns == 0:
            return 'X',
        if self.axis_half_turns == 0.5:
            return 'Y',
        return 'W({})'.format(self.axis_half_turns),

    def ascii_exponent(self):
        return self.half_turns

    def __repr__(self):
        return ('ExpWGate(half_turns={}, axis_half_turns={})'.format(
                    repr(self.half_turns),
                    repr(self.axis_half_turns)))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self.half_turns == other.half_turns and
                self.axis_half_turns == other.axis_half_turns)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ExpWGate, self.half_turns, self.axis_half_turns))


class ExpZGate(NativeGate,
               gate_features.AsciiDiagrammableGate,
               gate_features.PhaseableGate):
    """A rotation around the Z axis of the Bloch sphere."""

    def __init__(self, *positional_args,
                 half_turns: Union[ParameterizedValue, float]=1):
        assert not positional_args
        self.half_turns = _canonicalize_turns(half_turns)

    def ascii_wire_symbols(self):
        return 'Z',

    def ascii_exponent(self):
        return self.half_turns

    def phase_by(self, phase_turns, qubit_index):
        return self

    def to_proto(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')

        q = qubits[0]
        op = operations_pb2.Operation()
        op.exp_z.target.x = q.x
        op.exp_z.target.y = q.y
        op.exp_z.half_turns.raw = ParameterizedValue.val_of(self.half_turns)
        op.exp_z.half_turns.parameter_key = ParameterizedValue.key_of(
            self.half_turns)
        return op

    def __repr__(self):
        return 'ExpZGate(half_turns={})'.format(
            repr(self.half_turns))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ExpZGate, self.half_turns))


def _canonicalize_turns(
        val: Union[ParameterizedValue, float]
) -> Union[ParameterizedValue, float]:
    v = ParameterizedValue.val_of(val)
    v %= 2
    if v > 1:
        v -= 2
    return ParameterizedValue(ParameterizedValue.key_of(val), v)
