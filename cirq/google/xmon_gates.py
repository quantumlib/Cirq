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
                isinstance(op.gate, (ops.CZPowGate,
                                     ops.MeasurementGate,
                                     ops.ZPowGate))):
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
                exponent=param(exp_w['half_turns']),
                phase_exponent=param(exp_w['axis_half_turns']),
            ).on(qubit(exp_w['target']))
        elif 'exp_z' in proto_dict:
            exp_z = proto_dict['exp_z']
            if 'half_turns' not in exp_z or 'target' not in exp_z:
                raise_missing_fields('ExpZ')
            return ops.Z(qubit(exp_z['target']))**param(exp_z['half_turns'])
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


class ExpWGate(XmonGate, ops.SingleQubitGate):
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
                 phase_exponent: Union[value.Symbol, float] = 0.0,
                 exponent: Union[value.Symbol, float] = 1.0
                 ) -> None:
        """Initializes the gate.

        At most one rotation angle argument may be specified. At most one axis
        angle argument may be specified. If more are specified, the result is
        considered ambiguous and an error is thrown. If no angle argument is
        given, the default value of one half turn is used.

        The axis angle determines the rotation axis in the XY plane, with 0
        being positive-ward along X and 90 degrees being positive-ward along Y.

        Args:
            phase_exponent: The axis angle in the XY plane, in half_turns.
            exponent: The amount to rotate, in half_turns.
        """
        self.exponent = value.canonicalize_half_turns(exponent)
        self.phase_exponent = value.canonicalize_half_turns(phase_exponent)

        if (not isinstance(self.exponent, value.Symbol) and
                not isinstance(self.phase_exponent, value.Symbol) and
                not 0 <= self.phase_exponent < 1):
            # Canonicalize to negative rotation around positive axis.
            self.exponent = value.canonicalize_half_turns(-self.exponent)
            self.phase_exponent = value.canonicalize_half_turns(
                self.phase_exponent + 1)

    def to_proto_dict(self, *qubits):
        if len(qubits) != 1:
            raise ValueError('Wrong number of qubits.')

        q = qubits[0]
        exp_w = {
            'target': q.to_proto_dict(),
            'axis_half_turns': self.parameterized_value_to_proto_dict(
                self.phase_exponent),
            'half_turns': self.parameterized_value_to_proto_dict(
                self.exponent)
        }
        return {'exp_w': exp_w}

    def __pow__(self, exponent: Any) -> 'ExpWGate':
        new_exponent = protocols.mul(self.exponent, exponent, NotImplemented)
        if new_exponent is NotImplemented:
            return NotImplemented
        return ExpWGate(exponent=new_exponent,
                        phase_exponent=self.phase_exponent)

    def _has_unitary_(self) -> bool:
        return not (isinstance(self.exponent, value.Symbol) or
                    isinstance(self.phase_exponent, value.Symbol))

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        if not self._has_unitary_():
            return NotImplemented

        phase = protocols.unitary(ops.Z**self.phase_exponent)
        c = np.exp(1j * np.pi * self.exponent)
        rot = np.array([[1 + c, 1 - c], [1 - c, 1 + c]]) / 2
        return np.dot(np.dot(phase, rot), np.conj(phase))

    def _phase_by_(self, phase_turns, qubit_index):
        return ExpWGate(
            exponent=self.exponent,
            phase_exponent=self.phase_exponent + phase_turns * 2)

    def _trace_distance_bound_(self):
        if isinstance(self.exponent, value.Symbol):
            return 1
        return abs(self.exponent) * 3.5

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        e = 0 if args.precision is None else 10**-args.precision
        half_turns = self.exponent
        if isinstance(self.phase_exponent, value.Symbol):
            s = 'W({})'.format(self.phase_exponent)
        elif abs(self.phase_exponent) <= e:
            s = 'X'
        elif (abs(self.phase_exponent - 1) <= e and
              isinstance(half_turns, float)):
            s = 'X'
            half_turns = -half_turns
        elif abs(self.phase_exponent - 0.5) <= e:
            s = 'Y'
        elif args.precision is not None:
            s = 'W({{:.{}}})'.format(args.precision).format(
                self.phase_exponent)
        else:
            s = 'W({})'.format(self.phase_exponent)
        return protocols.CircuitDiagramInfo((s,), half_turns)

    def __str__(self):
        info = protocols.circuit_diagram_info(self)
        if info.exponent == 1:
            return info.wire_symbols[0]
        return '{}^{}'.format(info.wire_symbols[0], info.exponent)

    def __repr__(self):
        return (
            'cirq.google.ExpWGate(exponent={}, phase_exponent={})'.format(
                repr(self.exponent),
                repr(self.phase_exponent)))

    def __eq__(self, other):
        if isinstance(other, ops.XPowGate):
            return (self.phase_exponent == 0 and
                    self.exponent == other.exponent)
        if isinstance(other, ops.YPowGate):
            return (self.phase_exponent == 0.5 and
                    self.exponent == other.exponent)
        if isinstance(other, type(self)):
            return (self.exponent == other.exponent and
                    self.phase_exponent == other.phase_exponent)
        return NotImplemented

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        if self.phase_exponent == 0:
            return hash((ops.XPowGate, self.exponent))
        if self.phase_exponent == 0.5:
            return hash((ops.YPowGate, self.exponent))
        return hash((ExpWGate, self.exponent, self.phase_exponent))

    def _is_parameterized_(self) -> bool:
        return (isinstance(self.exponent, value.Symbol) or
                isinstance(self.phase_exponent, value.Symbol))

    def _resolve_parameters_(self, param_resolver) -> 'ExpWGate':
        return ExpWGate(
                exponent=param_resolver.value_of(self.exponent),
                phase_exponent=param_resolver.value_of(self.phase_exponent))
