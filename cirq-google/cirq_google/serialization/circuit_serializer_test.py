# Copyright 2021 The Cirq Developers
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

from typing import Any, Dict, List, Optional

import attrs
import numpy as np
import pytest
import sympy
import tunits.units
from google.protobuf import json_format

import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.serialization.circuit_serializer import _SERIALIZER_NAME
from cirq_google.serialization.op_deserializer import OpDeserializer
from cirq_google.serialization.op_serializer import OpSerializer
from cirq_google.serialization.tag_deserializer import TagDeserializer
from cirq_google.serialization.tag_serializer import TagSerializer


class FakeDevice(cirq.Device):
    def __init__(self):
        pass


def op_proto(json: Dict) -> v2.program_pb2.Operation:
    op = v2.program_pb2.Operation()
    json_format.ParseDict(json, op)
    return op


def circuit_proto(json: Dict, qubits: List[str]):
    constants = [v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id=q)) for q in qubits]
    return v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[v2.program_pb2.Moment(operations=[op_proto(json)])],
        ),
        constants=constants,
    )


Q0 = cirq.GridQubit(2, 4)
Q1 = cirq.GridQubit(2, 5)


X_PROTO = op_proto({'xpowgate': {'exponent': {'float_value': 1.0}}, 'qubit_constant_index': [0]})

# TODO(#5758): Add support for numpy types to `TParamVal`.
OPERATIONS = [
    (cirq.X(Q0), X_PROTO),
    (
        cg.InternalGate(gate_name='g', gate_module='test', num_qubits=1)(Q0),
        op_proto(
            {
                'internalgate': {'name': 'g', 'module': 'test', 'num_qubits': 1},
                'qubit_constant_index': [0],
            }
        ),
    ),
    (
        cirq.Y(Q0),
        op_proto({'ypowgate': {'exponent': {'float_value': 1.0}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.Z(Q0),
        op_proto({'zpowgate': {'exponent': {'float_value': 1.0}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.XPowGate(exponent=0.125)(Q1),
        op_proto({'xpowgate': {'exponent': {'float_value': 0.125}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.XPowGate(exponent=np.double(0.125))(Q1),
        op_proto({'xpowgate': {'exponent': {'float_value': 0.125}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.XPowGate(exponent=np.short(1))(Q1),
        op_proto({'xpowgate': {'exponent': {'float_value': 1.0}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.XPowGate(exponent=sympy.Symbol('a'))(Q1),
        op_proto({'xpowgate': {'exponent': {'symbol': 'a'}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.XPowGate(exponent=0.25 + sympy.Symbol('t'))(Q1),
        op_proto(
            {
                'xpowgate': {
                    'exponent': {
                        'func': {
                            'type': 'add',
                            'args': [{'arg_value': {'float_value': 0.25}}, {'symbol': 't'}],
                        }
                    }
                },
                'qubit_constant_index': [0],
            }
        ),
    ),
    (
        cirq.XPowGate(exponent=2 * sympy.Symbol('a'))(Q1),
        op_proto(
            {
                'xpowgate': {
                    'exponent': {
                        'func': {
                            'type': 'mul',
                            'args': [{'arg_value': {'float_value': 2.00}}, {'symbol': 'a'}],
                        }
                    }
                },
                'qubit_constant_index': [0],
            }
        ),
    ),
    (
        cirq.XPowGate(exponent=2 ** sympy.Symbol('a'))(Q1),
        op_proto(
            {
                'xpowgate': {
                    'exponent': {
                        'func': {
                            'type': 'pow',
                            'args': [{'arg_value': {'float_value': 2.00}}, {'symbol': 'a'}],
                        }
                    }
                },
                'qubit_constant_index': [0],
            }
        ),
    ),
    (
        cirq.YPowGate(exponent=0.25)(Q0),
        op_proto({'ypowgate': {'exponent': {'float_value': 0.25}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.ZPowGate(exponent=0.5)(Q0),
        op_proto({'zpowgate': {'exponent': {'float_value': 0.5}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.ZPowGate(exponent=0.5)(Q0).with_tags(cg.PhysicalZTag()),
        op_proto(
            {
                'zpowgate': {'exponent': {'float_value': 0.5}, 'is_physical_z': True},
                'qubit_constant_index': [0],
                'tag_indices': [1],
            }
        ),
    ),
    (
        cirq.PhasedXPowGate(phase_exponent=0.125, exponent=0.5)(Q0),
        op_proto(
            {
                'phasedxpowgate': {
                    'phase_exponent': {'float_value': 0.125},
                    'exponent': {'float_value': 0.5},
                },
                'qubit_constant_index': [0],
            }
        ),
    ),
    (
        cirq.PhasedXZGate(x_exponent=0.125, z_exponent=0.5, axis_phase_exponent=0.25)(Q0),
        op_proto(
            {
                'phasedxzgate': {
                    'x_exponent': {'float_value': 0.125},
                    'z_exponent': {'float_value': 0.5},
                    'axis_phase_exponent': {'float_value': 0.25},
                },
                'qubit_constant_index': [0],
            }
        ),
    ),
    (
        cirq.CZ(Q0, Q1),
        op_proto({'czpowgate': {'exponent': {'float_value': 1.0}}, 'qubit_constant_index': [0, 1]}),
    ),
    (
        cirq.CZPowGate(exponent=0.5)(Q0, Q1),
        op_proto({'czpowgate': {'exponent': {'float_value': 0.5}}, 'qubit_constant_index': [0, 1]}),
    ),
    (
        cirq.ISwapPowGate(exponent=0.5)(Q0, Q1),
        op_proto(
            {'iswappowgate': {'exponent': {'float_value': 0.5}}, 'qubit_constant_index': [0, 1]}
        ),
    ),
    (
        cirq.FSimGate(theta=2 + sympy.Symbol('a'), phi=1)(Q0, Q1),
        op_proto(
            {
                'fsimgate': {
                    'theta': {
                        'func': {
                            'type': 'add',
                            'args': [{'arg_value': {'float_value': 2.00}}, {'symbol': 'a'}],
                        }
                    },
                    'phi': {'float_value': 1.0},
                },
                'qubit_constant_index': [0, 1],
            }
        ),
    ),
    (
        cirq.FSimGate(theta=0.5, phi=0.25)(Q0, Q1),
        op_proto(
            {
                'fsimgate': {'theta': {'float_value': 0.5}, 'phi': {'float_value': 0.25}},
                'qubit_constant_index': [0, 1],
            }
        ),
    ),
    (
        cirq.FSimGate(theta=0.5, phi=0.0)(Q0, Q1),
        op_proto(
            {
                'fsimgate': {'theta': {'float_value': 0.5}, 'phi': {'float_value': 0.0}},
                'qubit_constant_index': [0, 1],
            }
        ),
    ),
    (
        cirq.FSimGate(theta=2, phi=1)(Q0, Q1),
        op_proto(
            {
                'fsimgate': {'theta': {'float_value': 2.0}, 'phi': {'float_value': 1.0}},
                'qubit_constant_index': [0, 1],
            }
        ),
    ),
    (
        cirq.FSimGate(theta=2, phi=1)(Q0, Q1).with_tags(cg.FSimViaModelTag()),
        op_proto(
            {
                'fsimgate': {
                    'theta': {'float_value': 2.0},
                    'phi': {'float_value': 1.0},
                    'translate_via_model': True,
                },
                'qubit_constant_index': [0, 1],
                'tag_indices': [2],
            }
        ),
    ),
    (
        cirq.WaitGate(duration=cirq.Duration(nanos=15))(Q0),
        op_proto(
            {'waitgate': {'duration_nanos': {'float_value': 15}}, 'qubit_constant_index': [0]}
        ),
    ),
    (
        cirq.WaitGate(duration=cirq.Duration(nanos=15), num_qubits=2)(Q0, Q1),
        op_proto(
            {'waitgate': {'duration_nanos': {'float_value': 15}}, 'qubit_constant_index': [0, 1]}
        ),
    ),
    (
        cirq.R(Q0),
        op_proto(
            {
                'resetgate': {'arguments': {'dimension': {'arg_value': {'float_value': 2}}}},
                'qubit_constant_index': [0],
            }
        ),
    ),
    (
        cirq.MeasurementGate(num_qubits=2, key='iron', invert_mask=(True, False))(Q0, Q1),
        op_proto(
            {
                'measurementgate': {
                    'key': {'arg_value': {'string_value': 'iron'}},
                    'invert_mask': {'arg_value': {'bool_values': {'values': [True, False]}}},
                },
                'qubit_constant_index': [0, 1],
            }
        ),
    ),
    (
        cg.experimental.CouplerPulse(
            hold_time=cirq.Duration(picos=1),
            rise_time=cirq.Duration(nanos=3),
            padding_time=cirq.Duration(micros=8),
            coupling_mhz=4.0,
            q0_detune_mhz=sympy.Symbol('x'),
            q1_detune_mhz=sympy.Symbol('y'),
        )(Q0, Q1),
        op_proto(
            {
                'couplerpulsegate': {
                    'hold_time_ps': {'float_value': 1.0},
                    'rise_time_ps': {'float_value': 3e3},
                    'padding_time_ps': {'float_value': 8e6},
                    'coupling_mhz': {'float_value': 4.0},
                    'q0_detune_mhz': {'symbol': 'x'},
                    'q1_detune_mhz': {'symbol': 'y'},
                },
                'qubit_constant_index': [0, 1],
            }
        ),
    ),
    (
        cirq.ops.SingleQubitCliffordGate.X(Q0),
        op_proto(
            {
                'singlequbitcliffordgate': {
                    'tableau': {
                        'num_qubits': 1,
                        'initial_state': 0,
                        'rs': [False, True],
                        'xs': [True, False],
                        'zs': [False, True],
                    }
                },
                'qubit_constant_index': [0],
            }
        ),
    ),
    (
        cirq.H(Q0),
        op_proto({'hpowgate': {'exponent': {'float_value': 1.0}}, 'qubit_constant_index': [0]}),
    ),
    (
        cirq.H(Q0).with_classical_controls('a'),
        op_proto(
            {
                'hpowgate': {'exponent': {'float_value': 1.0}},
                'qubit_constant_index': [0],
                'conditioned_on': [{'measurement_key': {'string_key': 'a', 'index': -1}}],
            }
        ),
    ),
    (
        cirq.H(Q0).with_classical_controls(
            cirq.SympyCondition(sympy.Eq(sympy.Symbol('a'), sympy.Symbol('b')))
        ),
        op_proto(
            {
                'hpowgate': {'exponent': {'float_value': 1.0}},
                'qubit_constant_index': [0],
                'conditioned_on': [
                    {'func': {'type': '==', 'args': [{'symbol': 'a'}, {'symbol': 'b'}]}}
                ],
            }
        ),
    ),
    (
        cirq.H(Q0).with_classical_controls(
            cirq.BitMaskKeyCondition('a', bitmask=13, target_value=9, equal_target=False)
        ),
        op_proto(
            {
                'hpowgate': {'exponent': {'float_value': 1.0}},
                'qubit_constant_index': [0],
                'conditioned_on': [
                    {
                        'func': {
                            'type': 'bitmask!=',
                            'args': [
                                {'measurement_key': {'string_key': 'a', 'index': -1}},
                                {'arg_value': {'float_value': 9}},
                                {'arg_value': {'float_value': 13}},
                            ],
                        }
                    }
                ],
            }
        ),
    ),
    (cirq.I(Q0), op_proto({'identitygate': {'qid_shape': [2]}, 'qubit_constant_index': [0]})),
]


@pytest.mark.parametrize(('op', 'op_proto'), OPERATIONS)
def test_serialize_deserialize_ops(op, op_proto):
    serializer = cg.CircuitSerializer()

    constants = []

    for q in op.qubits:
        constants.append(v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id=f'{q.row}_{q.col}')))
    for tag in op.tags:
        constants.append(v2.program_pb2.Constant(tag_value=tag.to_proto()))
    constants.append(v2.program_pb2.Constant(operation_value=op_proto))
    op_index = len(constants) - 1
    constants.append(
        v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[op_index]))
    )

    # Serialize / Deserializer circuit with single operation
    circuit = cirq.Circuit(op)
    circuit_proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moment_indices=[len(constants) - 1],
        ),
        constants=constants,
    )
    assert circuit_proto == serializer.serialize(circuit)
    assert serializer.deserialize(circuit_proto) == circuit


def test_serialize_deserialize_circuit():
    serializer = cg.CircuitSerializer()
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.X(q1), cirq.X(q0))

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moment_indices=[4, 5]
        ),
        constants=[
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_1')),
            v2.program_pb2.Constant(
                operation_value=v2.program_pb2.Operation(
                    xpowgate=v2.program_pb2.XPowGate(
                        exponent=v2.program_pb2.FloatArg(float_value=1.0)
                    ),
                    qubit_constant_index=[0],
                )
            ),
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_2')),
            v2.program_pb2.Constant(
                operation_value=v2.program_pb2.Operation(
                    xpowgate=v2.program_pb2.XPowGate(
                        exponent=v2.program_pb2.FloatArg(float_value=1.0)
                    ),
                    qubit_constant_index=[2],
                )
            ),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[1, 3])),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[1])),
        ],
    )
    assert proto == serializer.serialize(circuit)
    assert serializer.deserialize(proto) == circuit


def test_serialize_deserialize_circuit_with_constants_table():
    serializer = cg.CircuitSerializer()
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.X(q1), cirq.X(q0))
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moment_indices=[4, 5]
        ),
        constants=[
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_1')),
            v2.program_pb2.Constant(
                operation_value=v2.program_pb2.Operation(
                    xpowgate=v2.program_pb2.XPowGate(
                        exponent=v2.program_pb2.FloatArg(float_value=1.0)
                    ),
                    qubit_constant_index=[0],
                )
            ),
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_2')),
            v2.program_pb2.Constant(
                operation_value=v2.program_pb2.Operation(
                    xpowgate=v2.program_pb2.XPowGate(
                        exponent=v2.program_pb2.FloatArg(float_value=1.0)
                    ),
                    qubit_constant_index=[2],
                )
            ),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[1, 3])),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[1])),
        ],
    )
    assert proto == serializer.serialize(circuit)
    assert serializer.deserialize(proto) == circuit


def test_deserialize_circuit_with_mixed_moments_and_indicies_not_allowed():
    serializer = cg.CircuitSerializer()
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moment_indices=[4],
            moments=[
                v2.program_pb2.Moment(
                    operations=[
                        v2.program_pb2.Operation(
                            xpowgate=v2.program_pb2.XPowGate(
                                exponent=v2.program_pb2.FloatArg(float_value=1.0)
                            ),
                            qubit_constant_index=[0],
                        )
                    ]
                )
            ],
        ),
        constants=[
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_1')),
            v2.program_pb2.Constant(
                operation_value=v2.program_pb2.Operation(
                    xpowgate=v2.program_pb2.XPowGate(
                        exponent=v2.program_pb2.FloatArg(float_value=1.0)
                    ),
                    qubit_constant_index=[0],
                )
            ),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[1])),
        ],
    )
    with pytest.raises(ValueError, match="set at the same time"):
        _ = serializer.deserialize(proto)


def test_serialize_deserialize_circuit_with_duplicate_moments():
    q = cirq.GridQubit(4, 3)
    circuit = cirq.Circuit(cirq.X(q), cirq.Z(q), cirq.X(q), cirq.Z(q))
    serializer = cg.CircuitSerializer()
    proto = serializer.serialize(circuit)
    deserialized_circuit = serializer.deserialize(proto)
    assert deserialized_circuit == circuit
    assert deserialized_circuit[0] is deserialized_circuit[2]
    assert deserialized_circuit[1] is deserialized_circuit[3]


def test_serialize_deserialize_circuit_with_tokens():
    serializer = cg.CircuitSerializer()
    tag1 = cg.CalibrationTag('abc123')
    tag2 = cg.CalibrationTag('def456')
    circuit = cirq.Circuit(
        cirq.X(Q0).with_tags(tag1),
        cirq.X(Q1).with_tags(tag2),
        cirq.X(Q0).with_tags(tag2),
        cirq.X(Q0),
    )

    op_q0_tag1 = v2.program_pb2.Operation()
    op_q0_tag1.xpowgate.exponent.float_value = 1.0
    op_q0_tag1.qubit_constant_index.append(0)
    op_q0_tag1.token_constant_index = 1

    op_q1_tag2 = v2.program_pb2.Operation()
    op_q1_tag2.xpowgate.exponent.float_value = 1.0
    op_q1_tag2.qubit_constant_index.append(3)
    op_q1_tag2.token_constant_index = 4

    # Test repeated tag uses existing constant entey
    op_q0_tag2 = v2.program_pb2.Operation()
    op_q0_tag2.xpowgate.exponent.float_value = 1.0
    op_q0_tag2.qubit_constant_index.append(0)
    op_q0_tag2.token_constant_index = 4

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moment_indices=[6, 8, 10]
        ),
        constants=[
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_4')),
            v2.program_pb2.Constant(string_value='abc123'),
            v2.program_pb2.Constant(operation_value=op_q0_tag1),
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_5')),
            v2.program_pb2.Constant(string_value='def456'),
            v2.program_pb2.Constant(operation_value=op_q1_tag2),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[2, 5])),
            v2.program_pb2.Constant(operation_value=op_q0_tag2),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[7])),
            v2.program_pb2.Constant(operation_value=X_PROTO),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[9])),
        ],
    )
    assert proto == serializer.serialize(circuit)
    assert serializer.deserialize(proto) == circuit


def test_deserialize_circuit_with_token_strings():
    """Supporting token strings for backwards compatibility."""
    serializer = cg.CircuitSerializer()
    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moment_indices=[2]
        ),
        constants=[
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_4')),
            v2.program_pb2.Constant(
                operation_value=v2.program_pb2.Operation(
                    xpowgate=v2.program_pb2.XPowGate(
                        exponent=v2.program_pb2.FloatArg(float_value=1.0)
                    ),
                    token_value='abc123',
                    qubit_constant_index=[0],
                )
            ),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[1])),
        ],
    )
    tag = cg.CalibrationTag('abc123')
    circuit = cirq.Circuit(cirq.X(Q0).with_tags(tag))
    assert serializer.deserialize(proto) == circuit


def test_serialize_deserialize_circuit_with_subcircuit():
    serializer = cg.CircuitSerializer()
    tag1 = cg.CalibrationTag('abc123')
    fcircuit = cirq.FrozenCircuit(cirq.XPowGate(exponent=2 * sympy.Symbol('t'))(Q0))
    circuit = cirq.Circuit(
        cirq.X(Q1).with_tags(tag1),
        cirq.CircuitOperation(fcircuit, use_repetition_ids=True).repeat(repetition_ids=['a', 'b']),
        cirq.CircuitOperation(fcircuit, use_repetition_ids=True).with_qubit_mapping({Q0: Q1}),
        cirq.X(Q0),
    )

    op_x = v2.program_pb2.Operation()
    op_x.xpowgate.exponent.float_value = 1.0
    op_x.qubit_constant_index.append(3)
    op_tag = v2.program_pb2.Operation()
    op_tag.xpowgate.exponent.float_value = 1.0
    op_tag.qubit_constant_index.append(0)
    op_tag.token_constant_index = 1
    op_symbol = v2.program_pb2.Operation()
    op_symbol.xpowgate.exponent.func.type = 'mul'
    op_symbol.xpowgate.exponent.func.args.add().arg_value.float_value = 2.0
    op_symbol.xpowgate.exponent.func.args.add().symbol = 't'
    op_symbol.qubit_constant_index.append(3)

    c_op1 = v2.program_pb2.CircuitOperation()
    c_op1.circuit_constant_index = 6
    c_op1.use_repetition_ids = True
    rep_spec = c_op1.repetition_specification
    rep_spec.repetition_count = 2
    rep_spec.repetition_ids.ids.extend(['a', 'b'])

    c_op2 = v2.program_pb2.CircuitOperation()
    c_op2.circuit_constant_index = 6
    c_op2.use_repetition_ids = True
    c_op2.repetition_specification.repetition_count = 1
    qmap = c_op2.qubit_map.entries.add()
    qmap.key.id = '2_4'
    qmap.value.id = '2_5'

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moment_indices=[7, 9]
        ),
        constants=[
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_5')),
            v2.program_pb2.Constant(string_value='abc123'),
            v2.program_pb2.Constant(operation_value=op_tag),
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='2_4')),
            v2.program_pb2.Constant(operation_value=op_symbol),
            v2.program_pb2.Constant(moment_value=v2.program_pb2.Moment(operation_indices=[4])),
            v2.program_pb2.Constant(
                circuit_value=v2.program_pb2.Circuit(
                    scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moment_indices=[5]
                )
            ),
            v2.program_pb2.Constant(
                moment_value=v2.program_pb2.Moment(
                    operation_indices=[2], circuit_operations=[c_op1]
                )
            ),
            v2.program_pb2.Constant(operation_value=op_x),
            v2.program_pb2.Constant(
                moment_value=v2.program_pb2.Moment(
                    operation_indices=[8], circuit_operations=[c_op2]
                )
            ),
        ],
    )
    assert str(proto) == str(serializer.serialize(circuit))
    assert proto == serializer.serialize(circuit)
    assert serializer.deserialize(proto) == circuit


def test_circuit_operation_with_classical_controls():
    serializer = cg.CircuitSerializer()
    fcircuit = cirq.FrozenCircuit(cirq.X(Q1) ** 0.5, cirq.measure(Q1, key='a'))
    fcircuit2 = cirq.FrozenCircuit(cirq.X(Q1) ** 0.5)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            fcircuit,
            use_repetition_ids=False,
            repeat_until=cirq.KeyCondition(cirq.MeasurementKey('a')),
        ),
        cirq.CircuitOperation(fcircuit2, use_repetition_ids=False).with_classical_controls(
            cirq.SympyCondition(sympy.Symbol('a') >= 1)
        ),
    )
    proto = serializer.serialize(circuit)
    assert serializer.deserialize(proto) == circuit


def test_serialize_deserialize_empty_circuit():
    serializer = cg.CircuitSerializer()
    circuit = cirq.Circuit()

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT, moments=[]
        ),
    )
    assert proto == serializer.serialize(circuit)
    assert serializer.deserialize(proto) == circuit


def test_deserialize_empty_moment():
    serializer = cg.CircuitSerializer()
    circuit = cirq.Circuit([cirq.Moment()])

    proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[v2.program_pb2.Moment()],
        ),
    )
    assert serializer.deserialize(proto) == circuit


def test_circuit_serializer_name():
    serializer = cg.CircuitSerializer()
    assert serializer.name == _SERIALIZER_NAME
    assert cg.serialization.circuit_serializer.CIRCUIT_SERIALIZER.name == 'v2_5'


def test_serialize_unrecognized():
    serializer = cg.CircuitSerializer()
    with pytest.raises(NotImplementedError, match='program type'):
        serializer.serialize("not quite right")


def default_circuit():
    return cirq.FrozenCircuit(
        cirq.X(cirq.GridQubit(1, 1)) ** sympy.Symbol('k'),
        cirq.measure(cirq.GridQubit(1, 1), key='m'),
    )


def test_serialize_circuit_op_errors():
    serializer = cg.CircuitSerializer()
    constants = [serializer.serialize(default_circuit()).circuit]
    raw_constants = {default_circuit(): 0}

    op = cirq.CircuitOperation(default_circuit())
    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer._serialize_circuit_op(op)

    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer._serialize_circuit_op(op, constants=constants)

    with pytest.raises(ValueError, match='CircuitOp serialization requires a constants list'):
        serializer._serialize_circuit_op(op, raw_constants=raw_constants)


def test_serialize_op_unsupported_type():
    serializer = cg.CircuitSerializer()
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    with pytest.raises(ValueError, match='CNOT'):
        serializer.serialize(cirq.Circuit(cirq.CNOT(q0, q1)))


def test_serialize_op_bad_operation():
    serializer = cg.CircuitSerializer()

    class NullOperation(cirq.Operation):
        @property
        def qubits(self):
            return tuple()  # pragma: no cover

        def with_qubits(self, *qubits):
            return self  # pragma: no cover

    null_op = NullOperation()
    with pytest.raises(ValueError, match='Cannot serialize op'):
        serializer.serialize(cirq.Circuit(null_op))


def test_deserialize_fsim_missing_parameters():
    serializer = cg.CircuitSerializer()
    proto = circuit_proto(
        {'fsimgate': {'theta': {'float_value': 3.0}}, 'qubit_constant_index': [0, 1]},
        ['1_1', '1_2'],
    )
    with pytest.raises(ValueError, match='theta and phi must be specified'):
        serializer.deserialize(proto)


def test_deserialize_wrong_types():
    serializer = cg.CircuitSerializer()
    proto = circuit_proto(
        {
            'measurementgate': {
                'key': {'arg_value': {'float_value': 3.0}},
                'invert_mask': {'arg_value': {'bool_values': {'values': [True, False]}}},
            },
            'qubit_constant_index': [0],
        },
        ['1_1'],
    )
    with pytest.raises(ValueError, match='Incorrect types for measurement gate'):
        serializer.deserialize(proto)


def test_no_constants_table():
    serializer = cg.CircuitSerializer()
    op = op_proto(
        {
            'xpowgate': {'exponent': {'float_value': 1.0}},
            'qubits': [{'id': '1_2'}],
            'token_constant_index': 0,
        }
    )

    with pytest.raises(ValueError, match='Proto has references to constants table'):
        serializer._deserialize_gate_op(op)


def test_measurement_gate_deserialize() -> None:
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q) ** 0.5, cirq.measure(q))
    msg = cg.CIRCUIT_SERIALIZER.serialize(circuit)

    assert cg.CIRCUIT_SERIALIZER.deserialize(msg) == circuit


def test_circuit_with_cliffords():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(
        g(q) for g in cirq.ops.SingleQubitCliffordGate.all_single_qubit_cliffords
    )
    msg = cg.CIRCUIT_SERIALIZER.serialize(circuit)
    assert cg.CIRCUIT_SERIALIZER.deserialize(msg) == circuit


def test_circuit_with_couplerpulse():
    circuit = cirq.Circuit(cg.experimental.CouplerPulse(cirq.Duration(nanos=1), 2)(Q0, Q1))
    msg = cg.CIRCUIT_SERIALIZER.serialize(circuit)
    assert cg.CIRCUIT_SERIALIZER.deserialize(msg) == circuit


@pytest.mark.parametrize(
    'tag',
    [
        cg.ops.DynamicalDecouplingTag('X'),
        cg.FSimViaModelTag(),
        cg.PhysicalZTag(),
        cg.InternalTag(name='abc', package='xyz'),
    ],
)
def test_circuit_with_tag(tag):
    c = cirq.Circuit(cirq.X(cirq.q(0)).with_tags(tag), cirq.Z(cirq.q(0)).with_tags(tag))
    msg = cg.CIRCUIT_SERIALIZER.serialize(c)
    nc = cg.CIRCUIT_SERIALIZER.deserialize(msg)
    assert c == nc
    assert nc[0].operations[0].tags == (tag,)


@pytest.mark.filterwarnings('ignore:Unrecognized Tag .*DingDongTag')
def test_unrecognized_tag_is_ignored():
    class DingDongTag:
        pass

    c = cirq.Circuit(cirq.X(cirq.q(0)).with_tags(DingDongTag()))
    msg = cg.CIRCUIT_SERIALIZER.serialize(c)
    nc = cg.CIRCUIT_SERIALIZER.deserialize(msg)
    assert cirq.Circuit(cirq.X(cirq.q(0))) == nc


@pytest.mark.filterwarnings('ignore:Unknown tag msg=phase_match')
def test_unknown_tag_is_ignored():
    op_tag = v2.program_pb2.Operation()
    op_tag.xpowgate.exponent.float_value = 1.0
    op_tag.qubit_constant_index.append(0)
    op_tag.tag_indices.append(1)
    tag = v2.program_pb2.Tag()
    tag.phase_match.SetInParent()
    circuit_proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[v2.program_pb2.Moment(operations=[op_tag])],
        ),
        constants=[
            v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_1')),
            v2.program_pb2.Constant(tag_value=tag),
        ],
    )
    expected_circuit_no_tag = cirq.Circuit(cirq.X(cirq.GridQubit(1, 1)))
    assert cg.CIRCUIT_SERIALIZER.deserialize(circuit_proto) == expected_circuit_no_tag


def test_backwards_compatibility_with_old_tags():
    op_tag = v2.program_pb2.Operation()
    op_tag.xpowgate.exponent.float_value = 1.0
    op_tag.qubit_constant_index.append(0)
    tag = v2.program_pb2.Tag()
    tag.dynamical_decoupling.protocol = "X"
    op_tag.tags.append(tag)
    circuit_proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[v2.program_pb2.Moment(operations=[op_tag])],
        ),
        constants=[v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_1'))],
    )
    expected_circuit = cirq.Circuit(
        cirq.X(cirq.GridQubit(1, 1)).with_tags(cg.ops.DynamicalDecouplingTag(protocol='X'))
    )
    assert cg.CIRCUIT_SERIALIZER.deserialize(circuit_proto) == expected_circuit


def test_circuit_with_units():
    c = cirq.Circuit(
        cg.InternalGate(
            gate_module='test', gate_name='test', parameter_with_unit=3.14 * tunits.units.ns
        )(cirq.q(0, 0))
    )
    msg = cg.CIRCUIT_SERIALIZER.serialize(c)
    assert c == cg.CIRCUIT_SERIALIZER.deserialize(msg)


class BingBongGate(cirq.Gate):

    def __init__(self, param: float):
        self.param = param

    def _num_qubits_(self) -> int:
        return 1


class BingBongSerializer(OpSerializer):
    """Describes how to serialize CircuitOperations."""

    def can_serialize_operation(self, op):
        return isinstance(op.gate, BingBongGate)

    def to_proto(
        self,
        op: cirq.Operation,
        msg: Optional[v2.program_pb2.CircuitOperation] = None,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> v2.program_pb2.CircuitOperation:
        assert isinstance(op.gate, BingBongGate)
        if msg is None:
            msg = v2.program_pb2.Operation()  # pragma: no cover
        msg.internalgate.name = 'bingbong'
        msg.internalgate.module = 'test'
        msg.internalgate.num_qubits = 1
        msg.internalgate.gate_args['param'].arg_value.float_value = op.gate.param

        for qubit in op.qubits:
            if qubit not in raw_constants:
                constants.append(
                    v2.program_pb2.Constant(
                        qubit=v2.program_pb2.Qubit(id=v2.qubit_to_proto_id(qubit))
                    )
                )
                raw_constants[qubit] = len(constants) - 1
            msg.qubit_constant_index.append(raw_constants[qubit])
        return msg


class BingBongDeserializer(OpDeserializer):
    """Describes how to serialize CircuitOperations."""

    def can_deserialize_proto(self, proto):
        return (
            isinstance(proto, v2.program_pb2.Operation)
            and proto.WhichOneof("gate_value") == "internalgate"
            and proto.internalgate.name == 'bingbong'
            and proto.internalgate.module == 'test'
        )

    def from_proto(
        self,
        proto: v2.program_pb2.Operation,
        *,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> cirq.Operation:
        return BingBongGate(param=proto.internalgate.gate_args["param"].arg_value.float_value).on(
            deserialized_constants[proto.qubit_constant_index[0]]
        )


def test_serdes_preserves_syc():
    serializer = cg.CircuitSerializer()
    c = cirq.Circuit(cg.SYC(cirq.q(0, 0), cirq.q(0, 1)))
    msg = serializer.serialize(c)
    deserialized_circuit = serializer.deserialize(msg)
    assert deserialized_circuit == c
    assert isinstance(c[0][cirq.q(0, 0)].gate, cg.SycamoreGate)


def test_custom_op_serializer():
    c = cirq.Circuit(BingBongGate(param=2.5)(cirq.q(0, 0)))
    serializer = cg.CircuitSerializer(
        op_serializer=BingBongSerializer(), op_deserializer=BingBongDeserializer()
    )
    msg = serializer.serialize(c)
    deserialized_circuit = serializer.deserialize(msg)
    moment = deserialized_circuit[0]
    assert len(moment) == 1
    op = moment[cirq.q(0, 0)]
    assert isinstance(op.gate, BingBongGate)
    assert op.gate.param == 2.5
    assert op.qubits == (cirq.q(0, 0),)


@attrs.frozen
class DiscountTag:
    discount: float


class DiscountTagSerializer(TagSerializer):
    """Describes how to serialize DiscountTag."""

    def can_serialize_tag(self, tag):
        return isinstance(tag, DiscountTag)

    def to_proto(
        self,
        tag: Any,
        msg: Optional[v2.program_pb2.Tag] = None,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> v2.program_pb2.Tag:
        assert isinstance(tag, DiscountTag)
        if msg is None:
            msg = v2.program_pb2.Tag()  # pragma: no cover
        msg.internal_tag.tag_name = 'Discount'
        msg.internal_tag.tag_package = 'test'
        msg.internal_tag.tag_args['discount'].arg_value.float_value = tag.discount
        return msg


class DiscountTagDeserializer(TagDeserializer):
    """Describes how to serialize CircuitOperations."""

    def can_deserialize_proto(self, proto):
        return (
            proto.WhichOneof("tag") == "internal_tag"
            and proto.internal_tag.tag_name == 'Discount'
            and proto.internal_tag.tag_package == 'test'
        )

    def from_proto(
        self,
        proto: v2.program_pb2.Operation,
        *,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> DiscountTag:
        return DiscountTag(discount=proto.internal_tag.tag_args["discount"].arg_value.float_value)


def test_custom_tag_serializer():
    c = cirq.Circuit(cirq.X(cirq.q(0, 0)).with_tags(DiscountTag(0.25)))
    serializer = cg.CircuitSerializer(
        tag_serializer=DiscountTagSerializer(), tag_deserializer=DiscountTagDeserializer()
    )
    msg = serializer.serialize(c)
    deserialized_circuit = serializer.deserialize(msg)
    moment = deserialized_circuit[0]
    assert len(moment) == 1
    op = moment[cirq.q(0, 0)]
    assert len(op.tags) == 1
    assert isinstance(op.tags[0], DiscountTag)
    assert op.tags[0].discount == 0.25


def test_custom_tag_serializer_with_tags_outside_constants():
    op_tag = v2.program_pb2.Operation()
    op_tag.xpowgate.exponent.float_value = 1.0
    op_tag.qubit_constant_index.append(0)
    tag = v2.program_pb2.Tag()
    tag.internal_tag.tag_name = 'Discount'
    tag.internal_tag.tag_package = 'test'
    tag.internal_tag.tag_args['discount'].arg_value.float_value = 0.5
    op_tag.tags.append(tag)
    circuit_proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[v2.program_pb2.Moment(operations=[op_tag])],
        ),
        constants=[v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_1'))],
    )
    expected_circuit = cirq.Circuit(cirq.X(cirq.GridQubit(1, 1)).with_tags(DiscountTag(0.50)))
    serializer = cg.CircuitSerializer(
        tag_serializer=DiscountTagSerializer(), tag_deserializer=DiscountTagDeserializer()
    )
    assert serializer.deserialize(circuit_proto) == expected_circuit


def test_reset_gate_with_improper_argument():
    serializer = cg.CircuitSerializer()

    op = v2.program_pb2.Operation()
    op.resetgate.arguments['dimension'].arg_value.float_value = 2.5
    op.qubit_constant_index.append(0)
    circuit_proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[v2.program_pb2.Moment(operations=[op])],
        ),
        constants=[v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_2'))],
    )

    with pytest.raises(ValueError, match="must be an integer"):
        serializer.deserialize(circuit_proto)


def test_reset_gate_with_no_dimension():
    serializer = cg.CircuitSerializer()

    op = v2.program_pb2.Operation()
    op.resetgate.SetInParent()
    op.qubit_constant_index.append(0)
    circuit_proto = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='exp', gate_set=_SERIALIZER_NAME),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[v2.program_pb2.Moment(operations=[op])],
        ),
        constants=[v2.program_pb2.Constant(qubit=v2.program_pb2.Qubit(id='1_2'))],
    )
    reset_circuit = serializer.deserialize(circuit_proto)
    assert reset_circuit == cirq.Circuit(cirq.R(cirq.q(1, 2)))


def test_stimcirq_gates():
    stimcirq = pytest.importorskip("stimcirq")
    serializer = cg.CircuitSerializer()
    q = cirq.q(1, 2)
    q2 = cirq.q(2, 2)
    c = cirq.Circuit(
        cirq.Moment(
            stimcirq.CumulativeObservableAnnotation(parity_keys=["m"], observable_index=123)
        ),
        cirq.Moment(
            stimcirq.MeasureAndOrResetGate(
                measure=True,
                reset=False,
                basis='Z',
                invert_measure=True,
                key='mmm',
                measure_flip_probability=0.125,
            )(q2)
        ),
        cirq.Moment(stimcirq.ShiftCoordsAnnotation([1.0, 2.0])),
        cirq.Moment(
            stimcirq.SweepPauli(stim_sweep_bit_index=2, cirq_sweep_symbol='t', pauli=cirq.X)(q)
        ),
        cirq.Moment(
            stimcirq.SweepPauli(stim_sweep_bit_index=3, cirq_sweep_symbol='y', pauli=cirq.Y)(q)
        ),
        cirq.Moment(
            stimcirq.SweepPauli(stim_sweep_bit_index=4, cirq_sweep_symbol='t', pauli=cirq.Z)(q)
        ),
        cirq.Moment(stimcirq.TwoQubitAsymmetricDepolarizingChannel([0.05] * 15)(q, q2)),
        cirq.Moment(stimcirq.CZSwapGate()(q, q2)),
        cirq.Moment(stimcirq.CXSwapGate(inverted=True)(q, q2)),
        cirq.Moment(cirq.measure(q, key="m")),
        cirq.Moment(stimcirq.DetAnnotation(parity_keys=["m"])),
    )
    msg = serializer.serialize(c)
    deserialized_circuit = serializer.deserialize(msg)
    assert deserialized_circuit == c
