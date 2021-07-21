# Copyright 2019 The Cirq Developers
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

from typing import Dict
import numpy as np
import pytest
import sympy

from google.protobuf import json_format

import cirq
import cirq_google as cg
from cirq_google.api import v2


def op_proto(json_dict: Dict) -> v2.program_pb2.Operation:
    op = v2.program_pb2.Operation()
    json_format.ParseDict(json_dict, op)
    return op


@pytest.mark.parametrize(
    ('gate', 'axis_half_turns', 'half_turns'),
    [
        (cirq.X, 0.0, 1.0),
        (cirq.X ** 0.25, 0.0, 0.25),
        (cirq.Y, 0.5, 1.0),
        (cirq.Y ** 0.25, 0.5, 0.25),
        (cirq.PhasedXPowGate(exponent=0.125, phase_exponent=0.25), 0.25, 0.125),
        (cirq.rx(0.125 * np.pi), 0.0, 0.125),
        (cirq.ry(0.25 * np.pi), 0.5, 0.25),
    ],
)
def test_serialize_exp_w(gate, axis_half_turns, half_turns):
    q = cirq.GridQubit(1, 2)
    expected = op_proto(
        {
            'gate': {'id': 'xy'},
            'args': {
                'axis_half_turns': {'arg_value': {'float_value': axis_half_turns}},
                'half_turns': {'arg_value': {'float_value': half_turns}},
            },
            'qubits': [{'id': '1_2'}],
        }
    )

    assert cg.XMON.serialize_op(gate.on(q)) == expected


@pytest.mark.parametrize(
    ('gate', 'axis_half_turns', 'half_turns'),
    [
        (cirq.X ** sympy.Symbol('x'), 0.0, 'x'),
        (cirq.Y ** sympy.Symbol('y'), 0.5, 'y'),
        (cirq.PhasedXPowGate(exponent=sympy.Symbol('x'), phase_exponent=0.25), 0.25, 'x'),
    ],
)
def test_serialize_exp_w_parameterized_half_turns(gate, axis_half_turns, half_turns):
    q = cirq.GridQubit(1, 2)
    expected = op_proto(
        {
            'gate': {'id': 'xy'},
            'args': {
                'axis_half_turns': {'arg_value': {'float_value': axis_half_turns}},
                'half_turns': {'symbol': half_turns},
            },
            'qubits': [{'id': '1_2'}],
        }
    )

    assert cg.XMON.serialize_op(gate.on(q)) == expected


def test_serialize_exp_w_parameterized_axis_half_turns():
    gate = cirq.PhasedXPowGate(exponent=0.25, phase_exponent=sympy.Symbol('x'))
    q = cirq.GridQubit(1, 2)
    expected = op_proto(
        {
            'gate': {'id': 'xy'},
            'args': {
                'axis_half_turns': {'symbol': 'x'},
                'half_turns': {'arg_value': {'float_value': 0.25}},
            },
            'qubits': [{'id': '1_2'}],
        }
    )

    assert cg.XMON.serialize_op(gate.on(q)) == expected


@pytest.mark.parametrize(
    ('gate', 'half_turns'),
    [
        (cirq.Z, 1.0),
        (cirq.Z ** 0.125, 0.125),
        (cirq.rz(0.125 * np.pi), 0.125),
    ],
)
def test_serialize_exp_z(gate, half_turns):
    q = cirq.GridQubit(1, 2)
    assert cg.XMON.serialize_op(gate.on(q)) == op_proto(
        {
            'gate': {'id': 'z'},
            'args': {
                'half_turns': {'arg_value': {'float_value': half_turns}},
                'type': {'arg_value': {'string_value': 'virtual_propagates_forward'}},
            },
            'qubits': [{'id': '1_2'}],
        }
    )


def test_serialize_exp_z_parameterized():
    q = cirq.GridQubit(1, 2)
    gate = cirq.Z ** sympy.Symbol('x')
    assert cg.XMON.serialize_op(gate.on(q)) == op_proto(
        {
            'gate': {'id': 'z'},
            'args': {
                'half_turns': {'symbol': 'x'},
                'type': {'arg_value': {'string_value': 'virtual_propagates_forward'}},
            },
            'qubits': [{'id': '1_2'}],
        }
    )


@pytest.mark.parametrize(
    ('gate', 'half_turns'),
    [
        (cirq.CZ, 1.0),
        (cirq.CZ ** 0.125, 0.125),
    ],
)
def test_serialize_exp_11(gate, half_turns):
    c = cirq.GridQubit(1, 2)
    t = cirq.GridQubit(1, 3)
    assert cg.XMON.serialize_op(gate.on(c, t)) == op_proto(
        {
            'gate': {'id': 'cz'},
            'args': {
                'half_turns': {'arg_value': {'float_value': half_turns}},
            },
            'qubits': [{'id': '1_2'}, {'id': '1_3'}],
        }
    )


def test_serialize_exp_11_parameterized():
    c = cirq.GridQubit(1, 2)
    t = cirq.GridQubit(1, 3)
    gate = cirq.CZ ** sympy.Symbol('x')
    assert cg.XMON.serialize_op(gate.on(c, t)) == op_proto(
        {
            'gate': {'id': 'cz'},
            'args': {'half_turns': {'symbol': 'x'}},
            'qubits': [{'id': '1_2'}, {'id': '1_3'}],
        }
    )


@pytest.mark.parametrize(
    ('qubits', 'qubit_ids', 'key', 'invert_mask'),
    [
        ([cirq.GridQubit(1, 1)], ['1_1'], 'a', ()),
        ([cirq.GridQubit(1, 2)], ['1_2'], 'b', (True,)),
        ([cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)], ['1_1', '1_2'], 'a', (True, False)),
    ],
)
def test_serialize_meas(qubits, qubit_ids, key, invert_mask):
    op = cirq.measure(*qubits, key=key, invert_mask=invert_mask)
    expected = op_proto(
        {
            'gate': {'id': 'meas'},
            'qubits': [{'id': id} for id in qubit_ids],
            'args': {
                'key': {'arg_value': {'string_value': key}},
                'invert_mask': {'arg_value': {'bool_values': {'values': list(invert_mask)}}},
            },
        }
    )
    assert cg.XMON.serialize_op(op) == expected


def test_serialize_circuit():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    circuit = cirq.Circuit(cirq.CZ(q0, q1), cirq.X(q0), cirq.Z(q1), cirq.measure(q1, key='m'))
    expected = v2.program_pb2.Program(
        language=v2.program_pb2.Language(arg_function_language='', gate_set='xmon'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[
                v2.program_pb2.Moment(operations=[cg.XMON.serialize_op(cirq.CZ(q0, q1))]),
                v2.program_pb2.Moment(
                    operations=[cg.XMON.serialize_op(cirq.X(q0)), cg.XMON.serialize_op(cirq.Z(q1))]
                ),
                v2.program_pb2.Moment(operations=[cg.XMON.serialize_op(cirq.measure(q1, key='m'))]),
            ],
        ),
    )
    assert cg.XMON.serialize(circuit) == expected


@pytest.mark.parametrize(
    ('axis_half_turns', 'half_turns'),
    [
        (0.25, 0.25),
        (0, 0.25),
        (0.5, 0.25),
    ],
)
def test_deserialize_exp_w(axis_half_turns, half_turns):
    serialized_op = op_proto(
        {
            'gate': {'id': 'xy'},
            'args': {
                'axis_half_turns': {'arg_value': {'float_value': axis_half_turns}},
                'half_turns': {'arg_value': {'float_value': half_turns}},
            },
            'qubits': [{'id': '1_2'}],
        }
    )
    q = cirq.GridQubit(1, 2)
    expected = cirq.PhasedXPowGate(exponent=half_turns, phase_exponent=axis_half_turns)(q)
    assert cg.XMON.deserialize_op(serialized_op) == expected


def test_deserialize_exp_w_parameterized():
    serialized_op = op_proto(
        {
            'gate': {'id': 'xy'},
            'args': {'axis_half_turns': {'symbol': 'x'}, 'half_turns': {'symbol': 'y'}},
            'qubits': [{'id': '1_2'}],
        }
    )
    q = cirq.GridQubit(1, 2)
    expected = cirq.PhasedXPowGate(exponent=sympy.Symbol('y'), phase_exponent=sympy.Symbol('x'))(q)
    assert cg.XMON.deserialize_op(serialized_op) == expected


@pytest.mark.parametrize('half_turns', [0, 0.25, 1.0])
def test_deserialize_exp_z(half_turns):
    serialized_op = op_proto(
        {
            'gate': {'id': 'z'},
            'args': {
                'half_turns': {'arg_value': {'float_value': half_turns}},
                'type': {'arg_value': {'string_value': 'virtual_propagates_forward'}},
            },
            'qubits': [{'id': '1_2'}],
        }
    )
    q = cirq.GridQubit(1, 2)
    expected = cirq.ZPowGate(exponent=half_turns)(q)
    assert cg.XMON.deserialize_op(serialized_op) == expected


def test_deserialize_exp_z_parameterized():
    serialized_op = op_proto(
        {
            'gate': {'id': 'z'},
            'args': {
                'half_turns': {'symbol': 'x'},
                'type': {'arg_value': {'string_value': 'virtual_propagates_forward'}},
            },
            'qubits': [{'id': '1_2'}],
        }
    )
    q = cirq.GridQubit(1, 2)
    expected = cirq.ZPowGate(exponent=sympy.Symbol('x'))(q)
    assert cg.XMON.deserialize_op(serialized_op) == expected


@pytest.mark.parametrize('half_turns', [0, 0.25, 1.0])
def test_deserialize_exp_11(half_turns):
    serialized_op = op_proto(
        {
            'gate': {'id': 'cz'},
            'args': {'half_turns': {'arg_value': {'float_value': half_turns}}},
            'qubits': [{'id': '1_2'}, {'id': '2_2'}],
        }
    )
    c = cirq.GridQubit(1, 2)
    t = cirq.GridQubit(2, 2)
    expected = cirq.CZPowGate(exponent=half_turns)(c, t)
    assert cg.XMON.deserialize_op(serialized_op) == expected


def test_deserialize_exp_11_parameterized():
    serialized_op = op_proto(
        {
            'gate': {'id': 'cz'},
            'args': {'half_turns': {'symbol': 'x'}},
            'qubits': [{'id': '1_2'}, {'id': '2_2'}],
        }
    )
    c = cirq.GridQubit(1, 2)
    t = cirq.GridQubit(2, 2)
    expected = cirq.CZPowGate(exponent=sympy.Symbol('x'))(c, t)
    assert cg.XMON.deserialize_op(serialized_op) == expected


@pytest.mark.parametrize(
    ('qubits', 'qubit_ids', 'key', 'invert_mask'),
    [
        ([cirq.GridQubit(1, 1)], ['1_1'], 'a', ()),
        ([cirq.GridQubit(1, 2)], ['1_2'], 'b', (True,)),
        ([cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)], ['1_1', '1_2'], 'a', (True, False)),
    ],
)
def test_deserialize_meas(qubits, qubit_ids, key, invert_mask):
    serialized_op = op_proto(
        {
            'gate': {'id': 'meas'},
            'args': {
                'invert_mask': {'arg_value': {'bool_values': {'values': list(invert_mask)}}},
                'key': {'arg_value': {'string_value': key}},
            },
            'qubits': [{'id': id} for id in qubit_ids],
        }
    )
    expected = cirq.measure(*qubits, key=key, invert_mask=invert_mask)
    assert cg.XMON.deserialize_op(serialized_op) == expected


def test_deserialize_circuit():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    circuit = cirq.Circuit(cirq.CZ(q0, q1), cirq.X(q0), cirq.Z(q1), cirq.measure(q1, key='m'))
    serialized = v2.program_pb2.Program(
        language=v2.program_pb2.Language(gate_set='xmon'),
        circuit=v2.program_pb2.Circuit(
            scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[
                v2.program_pb2.Moment(operations=[cg.XMON.serialize_op(cirq.CZ(q0, q1))]),
                v2.program_pb2.Moment(
                    operations=[cg.XMON.serialize_op(cirq.X(q0)), cg.XMON.serialize_op(cirq.Z(q1))]
                ),
                v2.program_pb2.Moment(operations=[cg.XMON.serialize_op(cirq.measure(q1, key='m'))]),
            ],
        ),
    )
    assert cg.XMON.deserialize(serialized) == circuit


def test_deserialize_schedule():
    q0 = cirq.GridQubit(4, 4)
    q1 = cirq.GridQubit(4, 5)
    circuit = cirq.Circuit(
        cirq.CZ(q0, q1), cirq.X(q0), cirq.Z(q1), cirq.measure(q0, key='a'), device=cg.Bristlecone
    )
    serialized = v2.program_pb2.Program(
        language=v2.program_pb2.Language(gate_set='xmon'),
        schedule=v2.program_pb2.Schedule(
            scheduled_operations=[
                v2.program_pb2.ScheduledOperation(
                    operation=cg.XMON.serialize_op(cirq.CZ(q0, q1)), start_time_picos=0
                ),
                v2.program_pb2.ScheduledOperation(
                    operation=cg.XMON.serialize_op(cirq.X(q0)), start_time_picos=200000
                ),
                v2.program_pb2.ScheduledOperation(
                    operation=cg.XMON.serialize_op(cirq.Z(q1)), start_time_picos=200000
                ),
                v2.program_pb2.ScheduledOperation(
                    operation=cg.XMON.serialize_op(cirq.measure(q0, key='a')),
                    start_time_picos=400000,
                ),
            ]
        ),
    )
    assert cg.XMON.deserialize(serialized, cg.Bristlecone) == circuit


def test_serialize_deserialize_syc():
    proto = op_proto({'gate': {'id': 'syc'}, 'args': {}, 'qubits': [{'id': '1_2'}, {'id': '1_3'}]})

    q0 = cirq.GridQubit(1, 2)
    q1 = cirq.GridQubit(1, 3)
    op = cg.SYC(q0, q1)
    assert cg.SYC_GATESET.serialize_op(op) == proto
    assert cg.SYC_GATESET.deserialize_op(proto) == op


def test_serialize_fails_on_other_fsim_gates():
    a = cirq.GridQubit(1, 2)
    b = cirq.GridQubit(2, 2)
    op = cirq.FSimGate(phi=0.5, theta=-0.2)(a, b)
    with pytest.raises(ValueError, match='Cannot serialize'):
        _ = cg.SYC_GATESET.serialize_op(op)
    with pytest.raises(ValueError, match='Cannot serialize'):
        _ = cg.SQRT_ISWAP_GATESET.serialize_op(op)


def test_serialize_fails_on_symbols():
    a = cirq.GridQubit(1, 2)
    b = cirq.GridQubit(2, 2)
    op = cirq.FSimGate(phi=np.pi / 2, theta=sympy.Symbol('t'))(a, b)
    with pytest.raises(ValueError, match='Cannot serialize'):
        _ = cg.SYC_GATESET.serialize_op(op)
    with pytest.raises(ValueError, match='Cannot serialize'):
        _ = cg.SQRT_ISWAP_GATESET.serialize_op(op)


def test_serialize_deserialize_sqrt_iswap():
    proto = op_proto(
        {'gate': {'id': 'fsim_pi_4'}, 'args': {}, 'qubits': [{'id': '1_2'}, {'id': '1_3'}]}
    )

    q0 = cirq.GridQubit(1, 2)
    q1 = cirq.GridQubit(1, 3)
    op = cirq.FSimGate(theta=np.pi / 4, phi=0)(q0, q1)
    op2 = cirq.ISWAP(q0, q1) ** -0.5
    assert cg.SQRT_ISWAP_GATESET.serialize_op(op) == proto
    assert cg.SQRT_ISWAP_GATESET.deserialize_op(proto) == op
    assert cg.SQRT_ISWAP_GATESET.serialize_op(op2) == proto
    # Note that ISWAP deserializes back to a FSimGate
    assert cg.SQRT_ISWAP_GATESET.deserialize_op(proto) == op


def test_serialize_deserialize_inv_sqrt_iswap():
    proto = op_proto(
        {'gate': {'id': 'inv_fsim_pi_4'}, 'args': {}, 'qubits': [{'id': '1_2'}, {'id': '1_3'}]}
    )

    q0 = cirq.GridQubit(1, 2)
    q1 = cirq.GridQubit(1, 3)
    op = cirq.FSimGate(theta=-np.pi / 4, phi=0)(q0, q1)
    op2 = cirq.ISWAP(q0, q1) ** +0.5
    assert cg.SQRT_ISWAP_GATESET.serialize_op(op) == proto
    assert cg.SQRT_ISWAP_GATESET.deserialize_op(proto) == op
    assert cg.SQRT_ISWAP_GATESET.serialize_op(op2) == proto
    # Note that ISWAP deserializes back to a FSimGate
    assert cg.SQRT_ISWAP_GATESET.deserialize_op(proto) == op


@pytest.mark.parametrize(
    ('gate', 'axis_half_turns', 'half_turns'),
    [
        (cirq.X ** 0.25, 0.0, 0.25),
        (cirq.Y ** 0.25, 0.5, 0.25),
        (cirq.XPowGate(exponent=0.125), 0.0, 0.125),
        (cirq.YPowGate(exponent=0.125), 0.5, 0.125),
        (cirq.PhasedXPowGate(exponent=0.125, phase_exponent=0.25), 0.25, 0.125),
        (cirq.rx(0.125 * np.pi), 0.0, 0.125),
        (cirq.ry(0.25 * np.pi), 0.5, 0.25),
    ],
)
def test_serialize_deserialize_arbitrary_xy(gate, axis_half_turns, half_turns):
    op = gate.on(cirq.GridQubit(1, 2))
    expected = op_proto(
        {
            'gate': {'id': 'xy'},
            'args': {
                'axis_half_turns': {'arg_value': {'float_value': axis_half_turns}},
                'half_turns': {'arg_value': {'float_value': half_turns}},
            },
            'qubits': [{'id': '1_2'}],
        }
    )
    assert cg.SYC_GATESET.serialize_op(op) == expected
    deserialized_op = cg.SYC_GATESET.deserialize_op(expected)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(deserialized_op),
        cirq.unitary(op),
        atol=1e-7,
    )
    assert cg.SQRT_ISWAP_GATESET.serialize_op(op) == expected
    deserialized_op = cg.SQRT_ISWAP_GATESET.deserialize_op(expected)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(deserialized_op),
        cirq.unitary(op),
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ('qubits', 'qubit_ids', 'key', 'invert_mask'),
    [
        ([cirq.GridQubit(1, 1)], ['1_1'], 'a', ()),
        ([cirq.GridQubit(1, 2)], ['1_2'], 'b', (True,)),
        ([cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)], ['1_1', '1_2'], 'a', (True, False)),
    ],
)
def test_serialize_deserialize_meas(qubits, qubit_ids, key, invert_mask):
    op = cirq.measure(*qubits, key=key, invert_mask=invert_mask)
    proto = op_proto(
        {
            'gate': {'id': 'meas'},
            'qubits': [{'id': id} for id in qubit_ids],
            'args': {
                'key': {'arg_value': {'string_value': key}},
                'invert_mask': {'arg_value': {'bool_values': {'values': list(invert_mask)}}},
            },
        }
    )
    assert cg.SYC_GATESET.serialize_op(op) == proto
    assert cg.SYC_GATESET.deserialize_op(proto) == op
    assert cg.SQRT_ISWAP_GATESET.serialize_op(op) == proto
    assert cg.SQRT_ISWAP_GATESET.deserialize_op(proto) == op


def test_serialize_deserialize_wait_gate():
    op = cirq.wait(cirq.GridQubit(1, 2), nanos=50.0)
    proto = op_proto(
        {
            'gate': {'id': 'wait'},
            'qubits': [{'id': '1_2'}],
            'args': {
                'nanos': {'arg_value': {'float_value': 50.0}},
            },
        }
    )
    assert cg.SYC_GATESET.serialize_op(op) == proto
    assert cg.SYC_GATESET.deserialize_op(proto) == op
    assert cg.SQRT_ISWAP_GATESET.serialize_op(op) == proto
    assert cg.SQRT_ISWAP_GATESET.deserialize_op(proto) == op


def default_circuit_proto():
    op1 = v2.program_pb2.Operation()
    op1.gate.id = 'x_pow'
    op1.args['half_turns'].arg_value.string_value = 'k'
    op1.qubits.add().id = '1_1'

    return v2.program_pb2.Circuit(
        scheduling_strategy=v2.program_pb2.Circuit.MOMENT_BY_MOMENT,
        moments=[
            v2.program_pb2.Moment(
                operations=[op1],
            ),
        ],
    )


def default_circuit():
    return cirq.FrozenCircuit(
        cirq.X(cirq.GridQubit(1, 1)) ** sympy.Symbol('k'),
        cirq.measure(cirq.GridQubit(1, 1), key='m'),
    )


@pytest.mark.parametrize('gateset', [cg.XMON, cg.SYC_GATESET, cg.SQRT_ISWAP_GATESET])
def test_serialize_deserialize_circuit_op(gateset):
    circuit_op = cirq.CircuitOperation(default_circuit())
    proto = v2.program_pb2.CircuitOperation()
    proto.circuit_constant_index = 0
    proto.repetition_specification.repetition_count = 1

    constants = [default_circuit_proto()]
    raw_constants = {default_circuit(): 0}
    assert (
        gateset.serialize_op(circuit_op, constants=constants, raw_constants=raw_constants) == proto
    )

    constants = [default_circuit_proto()]
    deserialized_constants = [default_circuit()]

    assert (
        gateset.deserialize_op(
            proto, constants=constants, deserialized_constants=deserialized_constants
        )
        == circuit_op
    )
