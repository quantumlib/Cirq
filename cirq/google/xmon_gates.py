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

from typing import Dict, Optional, Union, Tuple, Any, cast

import abc
import json

import numpy as np

from cirq import ops, value, protocols
from cirq.devices.grid_qubit import GridQubit

from cirq.type_workarounds import NotImplementedType


class XmonGate(ops.Gate, metaclass=abc.ABCMeta):
    """A gate with a known mechanism for encoding into google API protos."""

    @abc.abstractmethod
    def to_proto_dict(self, *qubits) -> Dict:
        """Returns a dictionary representing the proto.

        For definitions of the protos see api/google/v1/operations.proto
        """
        raise NotImplementedError()

    @staticmethod
    def is_supported_op(op: ops.Operation) -> bool:
        if (isinstance(op, ops.GateOperation) and
                isinstance(op.gate, (ops.Rot11Gate, ops.MeasurementGate))):
            return True
        return XmonGate.try_get_xmon_gate(op) is not None

    @staticmethod
    def try_get_xmon_gate(op: ops.Operation) -> Optional['XmonGate']:
        if (isinstance(op, ops.GateOperation) and
                isinstance(op.gate, XmonGate)):
            return op.gate
        return None

    @staticmethod
    def from_proto_dict(proto_dict: Dict) -> ops.Operation:
        """Convert the proto dictionary to the corresponding operation.

        See protos in api/google/v1 for specification of the protos.

        Args:
            proto_dict: Dictionary representing the proto. Keys are always
                strings, but values may be types correspond to a raw proto type
                or another dictionary (for messages).

        Returns:
            The operation.

        Raises:
            ValueError if the dictionary does not contain required values
            corresponding to the proto.
        """

        def raise_missing_fields(gate_name: str):
            raise ValueError(
                '{} missing required fields: {}'.format(gate_name, proto_dict))
        param = XmonGate.parameterized_value_from_proto_dict
        qubit = GridQubit.from_proto_dict
        if 'exp_w' in proto_dict:
            exp_w = proto_dict['exp_w']
            if ('half_turns' not in exp_w or 'axis_half_turns' not in exp_w
                    or 'target' not in exp_w):
                raise_missing_fields('ExpW')
            return ExpWGate(
                half_turns=param(exp_w['half_turns']),
                axis_half_turns=param(exp_w['axis_half_turns']),
            ).on(qubit(exp_w['target']))
        elif 'exp_z' in proto_dict:
            exp_z = proto_dict['exp_z']
            if 'half_turns' not in exp_z or 'target' not in exp_z:
                raise_missing_fields('ExpZ')
            return ExpZGate(
                half_turns=param(exp_z['half_turns'])
            ).on(qubit(exp_z['target']))
        elif 'exp_11' in proto_dict:
            exp_11 = proto_dict['exp_11']
            if ('half_turns' not in exp_11 or 'target1' not in exp_11
                    or 'target2' not in exp_11):
                raise_missing_fields('Exp11')
            return ops.CZ(qubit(exp_11['target1']),
                          qubit(exp_11['target2']))**param(exp_11['half_turns'])
        elif 'measurement' in proto_dict:
            meas = proto_dict['measurement']
            invert_mask = cast(Tuple[Any, ...], ())
            if 'invert_mask' in meas:
                invert_mask = tuple(json.loads(x) for x in meas['invert_mask'])
            if 'key' not in meas or 'targets' not in meas:
                raise_missing_fields('Measurement')
            return ops.MeasurementGate(
                key=meas['key'],
                invert_mask=invert_mask
            ).on(*[qubit(q) for q in meas['targets']])
        else:
            raise ValueError('invalid operation: {}'.format(proto_dict))

    @staticmethod
    def parameterized_value_from_proto_dict(message: Dict) -> Union[
        value.Symbol, float]:
        if 'raw' in message:
            return message['raw']
        if 'parameter_key' in message:
            return value.Symbol(message['parameter_key'])
        raise ValueError('No value specified for parameterized float.')


    @staticmethod
    def parameterized_value_to_proto_dict(
        param: Union[value.Symbol, float]) -> Dict:
        out = {}  # type: Dict
        if isinstance(param, value.Symbol):
            out['parameter_key'] = param.name
        else:
            out['raw'] = float(param)
        return out


class ExpWGate(XmonGate,
               ops.SingleQubitGate,
               ops.TextDiagrammable):
    """A rotation around an axis in the XY plane of the Bloch sphere.

    This gate is a "phased X rotation". Specifically:
        ───W(axis)^t─── = ───Z^-axis───X^t───Z^axis───

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

    def to_proto_dict(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')

        q = qubits[0]
        exp_w = {
            'target': q.to_proto_dict(),
            'axis_half_turns': self.parameterized_value_to_proto_dict(
                self.axis_half_turns),
            'half_turns': self.parameterized_value_to_proto_dict(
                self.half_turns)
        }
        return {'exp_w': exp_w}

    def __pow__(self, exponent: Any):
        if self._is_parameterized_() and exponent != 1:
            return NotImplemented
        if self.half_turns != 1 and isinstance(exponent, value.Symbol):
            return NotImplemented
        return ExpWGate(half_turns=self.half_turns * exponent,
                        axis_half_turns=self.axis_half_turns)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        if (isinstance(self.half_turns, value.Symbol) or
                isinstance(self.axis_half_turns, value.Symbol)):
            return NotImplemented

        phase = protocols.unitary(
            ops.RotZGate(half_turns=self.axis_half_turns))
        c = np.exp(1j * np.pi * self.half_turns)
        rot = np.array([[1 + c, 1 - c], [1 - c, 1 + c]]) / 2
        return np.dot(np.dot(phase, rot), np.conj(phase))

    def _phase_by_(self, phase_turns, qubit_index):
        return ExpWGate(
            half_turns=self.half_turns,
            axis_half_turns=self.axis_half_turns + phase_turns * 2)

    def _trace_distance_bound_(self):
        if isinstance(self.half_turns, value.Symbol):
            return 1
        return abs(self.half_turns) * 3.5

    def text_diagram_info(self, args: ops.TextDiagramInfoArgs
                          ) -> ops.TextDiagramInfo:
        e = 0 if args.precision is None else 10**-args.precision
        half_turns = self.half_turns
        if isinstance(self.axis_half_turns, value.Symbol):
            s = 'W({})'.format(self.axis_half_turns)
        elif abs(self.axis_half_turns) <= e:
            s = 'X'
        elif (abs(self.axis_half_turns - 1) <= e and
              isinstance(half_turns, float)):
            s = 'X'
            half_turns = -half_turns
        elif abs(self.axis_half_turns - 0.5) <= e:
            s = 'Y'
        elif args.precision is not None:
            s = 'W({{:.{}}})'.format(args.precision).format(
                self.axis_half_turns)
        else:
            s = 'W({})'.format(self.axis_half_turns)
        return ops.TextDiagramInfo((s,), half_turns)

    def __str__(self):
        info = self.text_diagram_info(
            ops.TextDiagramInfoArgs.UNINFORMED_DEFAULT)
        if info.exponent == 1:
            return info.wire_symbols[0]
        return '{}^{}'.format(info.wire_symbols[0], info.exponent)

    def __repr__(self):
        return (
            'cirq.google.ExpWGate(half_turns={}, axis_half_turns={})'.format(
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

    def _is_parameterized_(self) -> bool:
        return (isinstance(self.half_turns, value.Symbol) or
                isinstance(self.axis_half_turns, value.Symbol))

    def _resolve_parameters_(self, param_resolver) -> 'ExpWGate':
        return ExpWGate(
                half_turns=param_resolver.value_of(self.half_turns),
                axis_half_turns=param_resolver.value_of(self.axis_half_turns))


class ExpZGate(XmonGate, ops.RotZGate):
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
        super().__init__(half_turns=half_turns,
                         rads=rads,
                         degs=degs,
                         global_shift_in_half_turns=-0.5)

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'ExpZGate':
        return ExpZGate(half_turns=exponent)

    def to_proto_dict(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')
        q = qubits[0]
        exp_z = {'target': q.to_proto_dict(),
                 'half_turns': self.parameterized_value_to_proto_dict(
                     self.half_turns),
                 }
        return {'exp_z': exp_z}

    def __repr__(self):
        return 'cirq.google.ExpZGate(half_turns={!r})'.format(self.half_turns)

    def _is_parameterized_(self) -> bool:
        return isinstance(self.half_turns, value.Symbol)

    def _resolve_parameters_(self, param_resolver) -> 'ExpZGate':
        return ExpZGate(half_turns=param_resolver.value_of(self.half_turns))
