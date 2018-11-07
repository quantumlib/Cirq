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

from typing import Dict, Union, Tuple, Any, cast

import json

from cirq import ops, value
from cirq.devices.grid_qubit import GridQubit


def is_native_xmon_op(op: ops.Operation) -> bool:
    return (isinstance(op, ops.GateOperation) and
            isinstance(op.gate, (ops.CZPowGate,
                                 ops.MeasurementGate,
                                 ops.PhasedXPowGate,
                                 ops.XPowGate,
                                 ops.YPowGate,
                                 ops.ZPowGate)))


def xmon_op_from_proto_dict(proto_dict: Dict) -> ops.Operation:
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
    param = _parameterized_value_from_proto_dict
    qubit = GridQubit.from_proto_dict
    if 'exp_w' in proto_dict:
        exp_w = proto_dict['exp_w']
        if ('half_turns' not in exp_w or 'axis_half_turns' not in exp_w
                or 'target' not in exp_w):
            raise_missing_fields('ExpW')
        return ops.PhasedXPowGate(
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


def _parameterized_value_from_proto_dict(message: Dict
                                         ) -> Union[value.Symbol, float]:
    if 'raw' in message:
        return message['raw']
    if 'parameter_key' in message:
        return value.Symbol(message['parameter_key'])
    raise ValueError('No value specified for parameterized float. '
                     'Expected "raw" or "parameter_key" to be set. '
                     'message: {!r}'.format(message))


def _parameterized_value_to_proto_dict(param: Union[value.Symbol, float]
                                       ) -> Dict:
    out = {}  # type: Dict
    if isinstance(param, value.Symbol):
        out['parameter_key'] = param.name
    else:
        out['raw'] = float(param)
    return out
