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

import numpy as np

from cirq import ops
from cirq.api.google.v1 import operations_pb2
from cirq.extension import PotentialImplementation
from cirq.study.parameterized_value import ParameterizedValue


class XmonGate(ops.Gate, metaclass=abc.ABCMeta):
    """A gate with a known mechanism for encoding into google API protos."""

    @abc.abstractmethod
    def to_proto(self, *qubits) -> operations_pb2.Operation:
        raise NotImplementedError()


class XmonMeasurementGate(XmonGate, ops.MeasurementGate):
    """Indicates that a qubit should be measured, and where the result goes."""

    def to_proto(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')

        q = qubits[0]
        op = operations_pb2.Operation()
        op.measurement.target.x = q.x
        op.measurement.target.y = q.y
        op.measurement.key = self.key
        return op


class Exp11Gate(XmonGate,
                ops.AsciiDiagrammableGate,
                ops.InterchangeableQubitsGate,
                ops.PhaseableGate,
                PotentialImplementation):
    """A two-qubit interaction that phases the amplitude of the 11 state."""

    def __init__(self, *positional_args,
                 half_turns: Union[ParameterizedValue, float]=1):
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

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

    def try_cast_to(self, desired_type):
        if desired_type is ops.KnownMatrixGate and self.has_matrix():
            return self
        return super().try_cast_to(desired_type)

    def has_matrix(self):
        return not isinstance(self.half_turns, ParameterizedValue)

    def matrix(self):
        if not self.has_matrix():
            raise ValueError("Don't have a known matrix.")
        return ops.RotZGate(half_turns=self.half_turns).matrix()

    def ascii_wire_symbols(self):
        return 'Z', 'Z'

    def ascii_exponent(self):
        return self.half_turns

    def __str__(self):
        if self.half_turns == 1:
            return 'CZ'
        return self.__repr__()

    def __repr__(self):
        return 'Exp11Gate(half_turns={})'.format(
            repr(self.half_turns))

    def __eq__(self, other):
        if not isinstance(other, (ops.Rot11Gate, type(self))):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((ops.Rot11Gate, self.half_turns))


class ExpWGate(XmonGate,
               ops.AsciiDiagrammableGate,
               ops.PhaseableGate,
               ops.BoundedEffectGate,
               PotentialImplementation):
    """A rotation around an axis in the XY plane of the Bloch sphere."""

    def __init__(self, *positional_args,
                 half_turns: Union[ParameterizedValue, float]=1,
                 axis_half_turns: Union[ParameterizedValue, float]=0):
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)
        self.axis_half_turns = _canonicalize_half_turns(axis_half_turns)

        if (not isinstance(self.half_turns, ParameterizedValue) and
                not isinstance(self.axis_half_turns, ParameterizedValue) and
                not 0 <= self.axis_half_turns < 1):
            # Canonicalize to negative rotation around positive axis.
            self.half_turns = _canonicalize_half_turns(-self.half_turns)
            self.axis_half_turns = _canonicalize_half_turns(
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

    def try_cast_to(self, desired_type):
        if desired_type is ops.KnownMatrixGate and self.has_matrix():
            return self
        if desired_type is ops.ReversibleGate and self.has_inverse():
            return self
        return super().try_cast_to(desired_type)

    def has_inverse(self):
        return not isinstance(self.half_turns, ParameterizedValue)

    def inverse(self):
        if not self.has_inverse():
            raise ValueError("Don't have a known inverse.")
        return ExpWGate(half_turns=-self.half_turns,
                        axis_half_turns=self.axis_half_turns)

    def has_matrix(self):
        return (not isinstance(self.half_turns, ParameterizedValue) and
                not isinstance(self.axis_half_turns, ParameterizedValue))

    def matrix(self):
        if not self.has_matrix():
            raise ValueError("Don't have a known matrix.")
        phase = ops.RotZGate(half_turns=self.axis_half_turns).matrix()
        c = np.exp(1j * np.pi * self.half_turns)
        rot = np.array([[1 + c, 1 - c], [1 - c, 1 + c]]) / 2
        return phase.dot(rot).dot(np.conj(phase))

    def phase_by(self, phase_turns, qubit_index):
        return ExpWGate(
            half_turns=self.half_turns,
            axis_half_turns=self.axis_half_turns + phase_turns * 2)

    def trace_distance_bound(self):
        return abs(self.half_turns) * 3.5

    def ascii_wire_symbols(self):
        if self.axis_half_turns == 0:
            return 'X',
        if self.axis_half_turns == 0.5:
            return 'Y',
        return 'W({})'.format(self.axis_half_turns),

    def ascii_exponent(self):
        return self.half_turns

    def __str__(self):
        base = self.ascii_wire_symbols()[0]
        if self.half_turns == 1:
            return base
        return '{}^{}'.format(base, self.half_turns)

    def __repr__(self):
        return ('ExpWGate(half_turns={}, axis_half_turns={})'.format(
                    repr(self.half_turns),
                    repr(self.axis_half_turns)))

    def __eq__(self, other):
        if isinstance(other, ops.RotXGate):
            return (self.axis_half_turns == 0 and
                    self.half_turns == other.half_turns)
        if isinstance(other, ops.RotYGate):
            return (self.axis_half_turns == 0.5 and
                    self.half_turns == other.half_turns)
        if isinstance(other, type(self)):
            return (self.half_turns == other.half_turns and
                    self.axis_half_turns == other.axis_half_turns)
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        if self.axis_half_turns == 0:
            return hash((ops.RotXGate, self.half_turns))
        if self.axis_half_turns == 0.5:
            return hash((ops.RotYGate, self.half_turns))
        return hash((ExpWGate, self.half_turns, self.axis_half_turns))


class ExpZGate(XmonGate,
               ops.AsciiDiagrammableGate,
               ops.PhaseableGate,
               PotentialImplementation):
    """A rotation around the Z axis of the Bloch sphere."""

    def __init__(self, *positional_args,
                 half_turns: Union[ParameterizedValue, float]=1):
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    def ascii_wire_symbols(self):
        if self.half_turns in [-0.25, 0.25]:
            return 'T'
        if self.half_turns in [-0.5, 0.5]:
            return 'S'
        return 'Z',

    def ascii_exponent(self):
        if self.half_turns in [-0.5, -0.25, 0.25, 0.5]:
            return 1
        return self.half_turns

    def try_cast_to(self, desired_type):
        if desired_type is ops.KnownMatrixGate and self.has_matrix():
            return self
        if desired_type is ops.ReversibleGate and self.has_inverse():
            return self
        return super().try_cast_to(desired_type)

    def has_inverse(self):
        return not isinstance(self.half_turns, ParameterizedValue)

    def inverse(self):
        if not self.has_inverse():
            raise ValueError("Don't have a known inverse.")
        return ExpZGate(half_turns=-self.half_turns)

    def has_matrix(self):
        return not isinstance(self.half_turns, ParameterizedValue)

    def matrix(self):
        if not self.has_matrix():
            raise ValueError("Don't have a known matrix.")
        return ops.RotZGate(half_turns=self.half_turns).matrix()

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

    def __str__(self):
        if self.half_turns == 0.5:
            return 'S'
        if self.half_turns == 0.25:
            return 'T'
        if self.half_turns == -0.5:
            return 'S^-1'
        if self.half_turns == -0.25:
            return 'T^-1'
        return 'Z^{}'.format(self.half_turns)

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


def _canonicalize_half_turns(
        half_turns: Union[ParameterizedValue, float]
) -> Union[ParameterizedValue, float]:
    v = ParameterizedValue.val_of(half_turns)
    v %= 2
    if v > 1:
        v -= 2
    return ParameterizedValue(ParameterizedValue.key_of(half_turns), v)
