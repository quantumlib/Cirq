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

"""Gates that can be directly described to the API, without decomposition."""

from typing import Union, Optional

import numpy as np

from cirq import abc, ops, value
from cirq.api.google.v1 import operations_pb2
from cirq.extension import PotentialImplementation
from cirq.google.xmon_qubit import XmonQubit


class XmonGate(ops.Gate, metaclass=abc.ABCMeta):
    """A gate with a known mechanism for encoding into google API protos."""

    @abc.abstractmethod
    def to_proto(self, *qubits) -> operations_pb2.Operation:
        raise NotImplementedError()

    @staticmethod
    def from_proto(op: operations_pb2.Operation) -> ops.Operation:
        param = XmonGate.parameterized_value_from_proto
        qubit = XmonQubit.from_proto
        which = op.WhichOneof('operation')
        if which == 'exp_w':
            exp_w = op.exp_w
            return ExpWGate(
                half_turns=param(exp_w.half_turns),
                axis_half_turns=param(exp_w.axis_half_turns),
            ).on(qubit(exp_w.target))
        elif which == 'exp_z':
            exp_z = op.exp_z
            return ExpZGate(
                half_turns=param(exp_z.half_turns)
            ).on(qubit(exp_z.target))
        elif which == 'exp_11':
            exp_11 = op.exp_11
            return Exp11Gate(
                half_turns=param(exp_11.half_turns)
            ).on(qubit(exp_11.target1), qubit(exp_11.target2))
        elif which == 'measurement':
            meas = op.measurement
            return XmonMeasurementGate(
                key=meas.key
            ).on(*[qubit(q) for q in meas.targets])
        else:
            raise ValueError('invalid operation: {}'.format(op))

    @staticmethod
    def parameterized_value_from_proto(
        message: operations_pb2.ParameterizedFloat
    ) -> Union[value.Symbol, float]:
        which = message.WhichOneof('value')
        if which == 'raw':
            return message.raw
        elif which == 'parameter_key':
            return value.Symbol(message.parameter_key)
        else:
            raise ValueError('No value specified for parameterized float.')

    @staticmethod
    def parameterized_value_to_proto(
        param: Union[value.Symbol, float],
        out: operations_pb2.ParameterizedFloat = None
    ) -> operations_pb2.ParameterizedFloat:
        if out is None:
            out = operations_pb2.ParameterizedFloat()
        if isinstance(param, value.Symbol):
            out.parameter_key = param.name
        else:
            out.raw = float(param)
        return out


class XmonMeasurementGate(XmonGate, ops.MeasurementGate):
    """Indicates that qubits should be measured, and where the result goes.

    This measurement is done in the computational basis.
    """

    def to_proto(self, *qubits):
        if len(qubits) == 0:
            raise ValueError('Measurement gate on no qubits.')

        op = operations_pb2.Operation()
        for q in qubits:
            q.to_proto(op.measurement.targets.add())
        op.measurement.key = self.key
        return op

    def __repr__(self):
        return 'XmonMeasurementGate({})'.format(repr(self.key))


class Exp11Gate(XmonGate,
                ops.TextDiagrammableGate,
                ops.InterchangeableQubitsGate,
                ops.PhaseableGate,
                ops.ParameterizableGate,
                PotentialImplementation):
    """A two-qubit interaction that phases the amplitude of the 11 state.

    This gate is exp(i * pi * |11><11|  * half_turn).

    Note that this half_turn parameter is such that a full turn is the
    identity matrix, in contrast to the single qubit gates, where a full
    turn is minus identity. The single qubit half-turn gates are defined
    so that a full turn corresponds to a rotation on the Bloch sphere of a
    360 degree rotation. For two qubit gates, there isn't a Bloch sphere,
    so the half_turn corresponds to half of a full rotation in U(4).
    """

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: The amount of phasing of the 11 state, in half_turns.
            rads: The amount of phasing of the 11 state, in radians.
            degs: The amount of phasing of the 11 state, in degrees.
        """
        self.half_turns = value.chosen_angle_to_canonical_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs)

    def phase_by(self, phase_turns, qubit_index):
        return self

    def to_proto(self, *qubits):
        if len(qubits) != 2:
            raise ValueError('Wrong number of qubits.')

        p, q = qubits
        op = operations_pb2.Operation()
        p.to_proto(op.exp_11.target1)
        q.to_proto(op.exp_11.target2)
        self.parameterized_value_to_proto(self.half_turns,
                                          op.exp_11.half_turns)
        return op

    def try_cast_to(self, desired_type, ext):
        if desired_type is ops.KnownMatrixGate and self.has_matrix():
            return self
        return super().try_cast_to(desired_type, ext)

    def has_matrix(self):
        return not isinstance(self.half_turns, value.Symbol)

    def matrix(self):
        if not self.has_matrix():
            raise ValueError("Don't have a known matrix.")
        return ops.Rot11Gate(half_turns=self.half_turns).matrix()

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '@', 'Z'

    def text_diagram_exponent(self):
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
        return hash((Exp11Gate, self.half_turns))

    def is_parameterized(self) -> bool:
        return isinstance(self.half_turns, value.Symbol)

    def with_parameters_resolved_by(self, param_resolver) -> 'Exp11Gate':
        return Exp11Gate(half_turns=param_resolver.value_of(self.half_turns))


class ExpWGate(XmonGate,
               ops.SingleQubitGate,
               ops.TextDiagrammableGate,
               ops.PhaseableGate,
               ops.BoundedEffectGate,
               ops.ParameterizableGate,
               PotentialImplementation):
    """A rotation around an axis in the XY plane of the Bloch sphere.

    This gate is exp(-i * pi * W(axis_half_turn) * half_turn / 2) where
        W(theta) = cos(pi * theta) X + sin(pi * theta) Y
     or in matrix form
       W(theta) = [[0, cos(pi * theta) - i sin(pi * theta)],
                   [cos(pi * theta) + i sin(pi * theta), 0]]

    Note the half_turn nomenclature here comes from viewing this as a rotation
    on the Bloch sphere. Two half_turns correspond to a rotation in the
    bloch sphere of 360 degrees. Note that this is minus identity, not
    just identity.  Similarly the axis_half_turns refers thinking of rotating
    the Bloch operator, starting with the operator pointing along the X
    direction. An axis_half_turn of 1 corresponds to the operator pointing
    along the -X direction while an axis_half_turn of 0.5 correspond to
    an operator pointing along the Y direction.
    """

    def __init__(self, *,  # Forces keyword args.
                 axis_half_turns: Optional[Union[value.Symbol, float]] = None,
                 axis_rads: Optional[float] = None,
                 axis_degs: Optional[float] = None,
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one rotation angle argument may be specified. At most one axis
        angle argument may be specified. If more are specified, the result is
        considered ambiguous and an error is thrown. If no angle argument is
        given, the default value of one half turn is used.

        The axis angle determines the rotation axis in the XY plane, with 0
        being positive-ward along X and 90 degrees being positive-ward along Y.

        Args:
            axis_half_turns: The axis angle in the XY plane, in half_turns.
            axis_rads: The axis angle in the XY plane, in radians.
            axis_degs: The axis angle in the XY plane, in degrees.
            half_turns: The amount to rotate, in half_turns.
            rads: The amount to rotate, in radians.
            degs: The amount to rotate, in degrees.
        """
        self.half_turns = value.chosen_angle_to_canonical_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs)
        self.axis_half_turns = value.chosen_angle_to_canonical_half_turns(
            half_turns=axis_half_turns,
            rads=axis_rads,
            degs=axis_degs,
            default=0.0)

        if (not isinstance(self.half_turns, value.Symbol) and
                not isinstance(self.axis_half_turns, value.Symbol) and
                not 0 <= self.axis_half_turns < 1):
            # Canonicalize to negative rotation around positive axis.
            self.half_turns = value.canonicalize_half_turns(-self.half_turns)
            self.axis_half_turns = value.canonicalize_half_turns(
                self.axis_half_turns + 1)

    def to_proto(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')

        q = qubits[0]
        op = operations_pb2.Operation()
        q.to_proto(op.exp_w.target)
        self.parameterized_value_to_proto(self.axis_half_turns,
                                          op.exp_w.axis_half_turns)
        self.parameterized_value_to_proto(self.half_turns, op.exp_w.half_turns)
        return op

    def try_cast_to(self, desired_type, ext):
        if desired_type is ops.KnownMatrixGate and self.has_matrix():
            return self
        if desired_type is ops.ReversibleGate and self.has_inverse():
            return self
        return super().try_cast_to(desired_type, ext)

    def has_inverse(self):
        return not isinstance(self.half_turns, value.Symbol)

    def inverse(self):
        if not self.has_inverse():
            raise ValueError("Don't have a known inverse.")
        return ExpWGate(half_turns=-self.half_turns,
                        axis_half_turns=self.axis_half_turns)

    def has_matrix(self):
        return (not isinstance(self.half_turns, value.Symbol) and
                not isinstance(self.axis_half_turns, value.Symbol))

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
        if isinstance(self.half_turns, value.Symbol):
            return 1
        return abs(self.half_turns) * 3.5

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        e = 0 if precision is None else 10**-precision
        if abs(self.axis_half_turns) <= e:
            return 'X',
        if abs(self.axis_half_turns - 0.5) <= e:
            return 'Y',
        if precision is not None:
            return 'W({{:.{}}})'.format(precision).format(
                self.axis_half_turns),
        else:
            return 'W({})'.format(self.axis_half_turns),

    def text_diagram_exponent(self):
        return self.half_turns

    def __str__(self):
        base = self.text_diagram_wire_symbols()[0]
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

    def is_parameterized(self) -> bool:
        return (isinstance(self.half_turns, value.Symbol) or
                isinstance(self.axis_half_turns, value.Symbol))

    def with_parameters_resolved_by(self, param_resolver) -> 'ExpWGate':
        return ExpWGate(
                half_turns=param_resolver.value_of(self.half_turns),
                axis_half_turns=param_resolver.value_of(self.axis_half_turns))


class ExpZGate(XmonGate,
               ops.SingleQubitGate,
               ops.TextDiagrammableGate,
               ops.ParameterizableGate,
               PotentialImplementation):
    """A rotation around the Z axis of the Bloch sphere.

    This gate is exp(-i * pi * Z * half_turns / 2) where Z is the Z matrix
        Z = [[1, 0],
             [0, -1]]

    Note the half_turn nomenclature here comes from viewing this as a rotation
    on the Bloch sphere. Two half_turns correspond to a rotation in the
    bloch sphere of 360 degrees.
    """

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: The relative phasing of Z's eigenstates, in half_turns.
            rads: The relative phasing of Z's eigenstates, in radians.
            degs: The relative phasing of Z's eigenstates, in degrees.
        """
        self.half_turns = value.chosen_angle_to_canonical_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        if self.half_turns in [-0.25, 0.25]:
            return 'T'
        if self.half_turns in [-0.5, 0.5]:
            return 'S'
        return 'Z',

    def text_diagram_exponent(self):
        if self.half_turns in [0.25, 0.5]:
            return 1
        if self.half_turns in [-0.5, -0.25]:
            return -1
        return self.half_turns

    def try_cast_to(self, desired_type, ext):
        if desired_type is ops.KnownMatrixGate and self.has_matrix():
            return self
        if desired_type is ops.ReversibleGate and self.has_inverse():
            return self
        return super().try_cast_to(desired_type, ext)

    def has_inverse(self):
        return not isinstance(self.half_turns, value.Symbol)

    def inverse(self):
        if not self.has_inverse():
            raise ValueError("Don't have a known inverse.")
        return ExpZGate(half_turns=-self.half_turns)

    def has_matrix(self):
        return not isinstance(self.half_turns, value.Symbol)

    def matrix(self):
        if not self.has_matrix():
            raise ValueError("Don't have a known matrix.")
        return np.diag([(-1j)**self.half_turns, 1j**self.half_turns])

    def trace_distance_bound(self):
        if isinstance(self.half_turns, value.Symbol):
            return 1
        return abs(self.half_turns) * 3.5

    def to_proto(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')

        q = qubits[0]
        op = operations_pb2.Operation()
        q.to_proto(op.exp_z.target)
        self.parameterized_value_to_proto(self.half_turns, op.exp_z.half_turns)
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

    def is_parameterized(self) -> bool:
        return isinstance(self.half_turns, value.Symbol)

    def with_parameters_resolved_by(self, param_resolver) -> 'ExpZGate':
        return ExpZGate(half_turns=param_resolver.value_of(self.half_turns))
