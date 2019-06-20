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

import numpy as np
import pytest
import sympy

import cirq
import cirq.google as cg


@pytest.mark.parametrize(('gate', 'axis_half_turns', 'half_turns'), [
    (cirq.X, 0.0, 1.0),
    (cirq.X**0.25, 0.0, 0.25),
    (cirq.Y, 0.5, 1.0),
    (cirq.Y**0.25, 0.5, 0.25),
    (cirq.PhasedXPowGate(exponent=0.125, phase_exponent=0.25), 0.25, 0.125),
    (cirq.Rx(0.125 * np.pi), 0.0, 0.125),
    (cirq.Ry(0.25 * np.pi), 0.5, 0.25),
])
def test_serialize_exp_w(gate, axis_half_turns, half_turns):
    q = cirq.GridQubit(1, 2)
    expected = {
        'gate': {
            'id': 'exp_w'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': axis_half_turns
                }
            },
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }

    assert cg.XMON.serialize_op_dict(gate.on(q)) == expected


@pytest.mark.parametrize(('gate', 'axis_half_turns', 'half_turns'), [
    (cirq.X**sympy.Symbol('x'), 0.0, 'x'),
    (cirq.Y**sympy.Symbol('y'), 0.5, 'y'),
    (cirq.PhasedXPowGate(exponent=sympy.Symbol('x'),
                         phase_exponent=0.25), 0.25, 'x'),
])
def test_serialize_exp_w_parameterized_half_turns(gate, axis_half_turns,
                                                  half_turns):
    q = cirq.GridQubit(1, 2)
    expected = {
        'gate': {
            'id': 'exp_w'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': axis_half_turns
                }
            },
            'half_turns': {
                'symbol': half_turns
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }

    assert cg.XMON.serialize_op_dict(gate.on(q)) == expected


def test_serialize_exp_w_parameterized_axis_half_turns():
    gate = cirq.PhasedXPowGate(exponent=0.25, phase_exponent=sympy.Symbol('x'))
    q = cirq.GridQubit(1, 2)
    expected = {
        'gate': {
            'id': 'exp_w'
        },
        'args': {
            'axis_half_turns': {
                'symbol': 'x'
            },
            'half_turns': {
                'arg_value': {
                    'float_value': 0.25
                }
            },
        },
        'qubits': [{
            'id': '1_2'
        }]
    }

    assert cg.XMON.serialize_op_dict(gate.on(q)) == expected


@pytest.mark.parametrize(('gate', 'half_turns'), [
    (cirq.Z, 1.0),
    (cirq.Z**0.125, 0.125),
    (cirq.Rz(0.125 * np.pi), 0.125),
])
def test_serialize_exp_z(gate, half_turns):
    q = cirq.GridQubit(1, 2)
    assert cg.XMON.serialize_op_dict(gate.on(q)) == {
        'gate': {
            'id': 'exp_z'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }


def test_serialize_exp_z_parameterized():
    q = cirq.GridQubit(1, 2)
    gate = cirq.Z**sympy.Symbol('x')
    assert cg.XMON.serialize_op_dict(gate.on(q)) == {
        'gate': {
            'id': 'exp_z'
        },
        'args': {
            'half_turns': {
                'symbol': 'x'
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }


@pytest.mark.parametrize(('gate', 'half_turns'), [
    (cirq.CZ, 1.0),
    (cirq.CZ**0.125, 0.125),
])
def test_serialize_exp_11(gate, half_turns):
    c = cirq.GridQubit(1, 2)
    t = cirq.GridQubit(1, 3)
    assert cg.XMON.serialize_op_dict(gate.on(c, t)) == {
        'gate': {
            'id': 'exp_11'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }, {
            'id': '1_3'
        }]
    }


def test_serialize_exp_11_parameterized():
    c = cirq.GridQubit(1, 2)
    t = cirq.GridQubit(1, 3)
    gate = cirq.CZ**sympy.Symbol('x')
    assert cg.XMON.serialize_op_dict(gate.on(c, t)) == {
        'gate': {
            'id': 'exp_11'
        },
        'args': {
            'half_turns': {
                'symbol': 'x'
            }
        },
        'qubits': [{
            'id': '1_2'
        }, {
            'id': '1_3'
        }]
    }


@pytest.mark.parametrize(('qubits', 'qubit_ids', 'key', 'invert_mask'), [
    ([cirq.GridQubit(1, 1)], ['1_1'], 'a', ()),
    ([cirq.GridQubit(1, 2)], ['1_2'], 'b', (True,)),
    ([cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)], ['1_1', '1_2'], 'a',
     (True, False)),
])
def test_serialize_meas(qubits, qubit_ids, key, invert_mask):
    op = cirq.measure(*qubits, key=key, invert_mask=invert_mask)
    expected = {
        'gate': {
            'id': 'meas'
        },
        'qubits': [],
        'args': {
            'key': {
                'arg_value': {
                    'string_value': key
                }
            },
            'invert_mask': {
                'arg_value': {
                    'bool_values': {
                        'values': list(invert_mask)
                    }
                }
            }
        },
    }
    for qubit_id in qubit_ids:
        expected['qubits'].append({'id': qubit_id})
    assert cg.XMON.serialize_op_dict(op) == expected


def test_serialize_circuit():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    circuit = cirq.Circuit.from_ops(cirq.CZ(q0, q1), cirq.X(q0), cirq.Z(q1),
                                    cirq.measure(q1, key='m'))
    expected = {
        'language': {
            'arg_function_language': '',
            'gate_set': 'xmon'
        },
        'circuit': {
            'scheduling_strategy':
            1,
            'moments': [{
                'operations': [cg.XMON.serialize_op_dict(cirq.CZ(q0, q1))]
            }, {
                'operations': [
                    cg.XMON.serialize_op_dict(cirq.X(q0)),
                    cg.XMON.serialize_op_dict(cirq.Z(q1))
                ]
            }, {
                'operations':
                [cg.XMON.serialize_op_dict(cirq.measure(q1, key='m'))]
            }]
        },
    }
    assert cg.XMON.serialize_dict(circuit) == expected


def test_serialize_schedule():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    scheduled_ops = [
        cirq.ScheduledOperation.op_at_on(cirq.CZ(q0, q1),
                                         cirq.Timestamp(nanos=0),
                                         device=cg.Bristlecone),
        cirq.ScheduledOperation.op_at_on(cirq.X(q0),
                                         cirq.Timestamp(nanos=200),
                                         device=cg.Bristlecone),
        cirq.ScheduledOperation.op_at_on(cirq.Z(q1),
                                         cirq.Timestamp(nanos=200),
                                         device=cg.Bristlecone),
        cirq.ScheduledOperation.op_at_on(cirq.measure(q0, key='a'),
                                         cirq.Timestamp(nanos=400),
                                         device=cg.Bristlecone)
    ]
    schedule = cirq.Schedule(device=cirq.google.Bristlecone,
                             scheduled_operations=scheduled_ops)
    expected = {
        'language': {
            'arg_function_language': '',
            'gate_set': 'xmon'
        },
        'schedule': {
            'scheduled_operations': [{
                'operation':
                cg.XMON.serialize_op_dict(cirq.CZ(q0, q1)),
                'start_time_picos':
                '0'
            }, {
                'operation':
                cg.XMON.serialize_op_dict(cirq.X(q0)),
                'start_time_picos':
                '200000',
            }, {
                'operation':
                cg.XMON.serialize_op_dict(cirq.Z(q1)),
                'start_time_picos':
                '200000',
            }, {
                'operation':
                cg.XMON.serialize_op_dict(cirq.measure(q0, key='a')),
                'start_time_picos':
                '400000',
            }]
        },
    }
    assert cg.XMON.serialize_dict(schedule) == expected


@pytest.mark.parametrize(('axis_half_turns', 'half_turns'), [
    (0.25, 0.25),
    (0, 0.25),
    (0.5, 0.25),
])
def test_deserialize_exp_w(axis_half_turns, half_turns):
    serialized_op = {
        'gate': {
            'id': 'exp_w'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': axis_half_turns
                }
            },
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    expected = cirq.PhasedXPowGate(exponent=half_turns,
                                   phase_exponent=axis_half_turns)(q)
    assert cg.XMON.deserialize_op_dict(serialized_op) == expected


def test_deserialize_exp_w_parameterized():
    serialized_op = {
        'gate': {
            'id': 'exp_w'
        },
        'args': {
            'axis_half_turns': {
                'symbol': 'x'
            },
            'half_turns': {
                'symbol': 'y'
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    expected = cirq.PhasedXPowGate(exponent=sympy.Symbol('y'),
                                   phase_exponent=sympy.Symbol('x'))(q)
    assert cg.XMON.deserialize_op_dict(serialized_op) == expected


@pytest.mark.parametrize('half_turns', [0, 0.25, 1.0])
def test_deserialize_exp_z(half_turns):
    serialized_op = {
        'gate': {
            'id': 'exp_z'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    expected = cirq.ZPowGate(exponent=half_turns)(q)
    assert cg.XMON.deserialize_op_dict(serialized_op) == expected


def test_deserialize_exp_z_parameterized():
    serialized_op = {
        'gate': {
            'id': 'exp_z'
        },
        'args': {
            'half_turns': {
                'symbol': 'x'
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    expected = cirq.ZPowGate(exponent=sympy.Symbol('x'))(q)
    assert cg.XMON.deserialize_op_dict(serialized_op) == expected


@pytest.mark.parametrize('half_turns', [0, 0.25, 1.0])
def test_deserialize_exp_11(half_turns):
    serialized_op = {
        'gate': {
            'id': 'exp_11'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }, {
            'id': '2_2'
        }]
    }
    c = cirq.GridQubit(1, 2)
    t = cirq.GridQubit(2, 2)
    expected = cirq.CZPowGate(exponent=half_turns)(c, t)
    assert cg.XMON.deserialize_op_dict(serialized_op) == expected


def test_deserialize_exp_11_parameterized():
    serialized_op = {
        'gate': {
            'id': 'exp_11'
        },
        'args': {
            'half_turns': {
                'symbol': 'x'
            }
        },
        'qubits': [{
            'id': '1_2'
        }, {
            'id': '2_2'
        }]
    }
    c = cirq.GridQubit(1, 2)
    t = cirq.GridQubit(2, 2)
    expected = cirq.CZPowGate(exponent=sympy.Symbol('x'))(c, t)
    assert cg.XMON.deserialize_op_dict(serialized_op) == expected


@pytest.mark.parametrize(('qubits', 'qubit_ids', 'key', 'invert_mask'), [
    ([cirq.GridQubit(1, 1)], ['1_1'], 'a', ()),
    ([cirq.GridQubit(1, 2)], ['1_2'], 'b', (True,)),
    ([cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)], ['1_1', '1_2'], 'a',
     (True, False)),
])
def test_deserialize_meas(qubits, qubit_ids, key, invert_mask):
    serialized_op = {
        'gate': {
            'id': 'meas'
        },
        'args': {
            'invert_mask': {
                'arg_value': {
                    'bool_values': {
                        'values': list(invert_mask)
                    }
                }
            },
            'key': {
                'arg_value': {
                    'string_value': key
                }
            }
        },
        'qubits': [{
            'id': id
        } for id in qubit_ids]
    }
    expected = cirq.measure(*qubits, key=key, invert_mask=invert_mask)
    assert cg.XMON.deserialize_op_dict(serialized_op) == expected


def test_deserialize_circuit():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    circuit = cirq.Circuit.from_ops(cirq.CZ(q0, q1), cirq.X(q0), cirq.Z(q1),
                                    cirq.measure(q1, key='m'))
    serialized = {
        'language': {
            'gate_set': 'xmon'
        },
        'circuit': {
            'scheduling_strategy':
            1,
            'moments': [{
                'operations': [cg.XMON.serialize_op_dict(cirq.CZ(q0, q1))]
            }, {
                'operations': [
                    cg.XMON.serialize_op_dict(cirq.X(q0)),
                    cg.XMON.serialize_op_dict(cirq.Z(q1))
                ]
            }, {
                'operations':
                [cg.XMON.serialize_op_dict(cirq.measure(q1, key='m'))]
            }]
        },
    }
    assert cg.XMON.deserialize_dict(serialized) == circuit


def test_deserialize_schedule():
    q0 = cirq.GridQubit(1, 1)
    q1 = cirq.GridQubit(1, 2)
    scheduled_ops = [
        cirq.ScheduledOperation.op_at_on(cirq.CZ(q0, q1),
                                         cirq.Timestamp(nanos=0),
                                         device=cg.Bristlecone),
        cirq.ScheduledOperation.op_at_on(cirq.X(q0),
                                         cirq.Timestamp(nanos=200),
                                         device=cg.Bristlecone),
        cirq.ScheduledOperation.op_at_on(cirq.Z(q1),
                                         cirq.Timestamp(nanos=200),
                                         device=cg.Bristlecone),
        cirq.ScheduledOperation.op_at_on(cirq.measure(q0, key='a'),
                                         cirq.Timestamp(nanos=400),
                                         device=cg.Bristlecone)
    ]
    schedule = cirq.Schedule(device=cirq.google.Bristlecone,
                             scheduled_operations=scheduled_ops)
    serialized = {
        'language': {
            'gate_set': 'xmon'
        },
        'schedule': {
            'scheduled_operations': [{
                'operation':
                cg.XMON.serialize_op_dict(cirq.CZ(q0, q1)),
                'start_time_picos':
                0
            }, {
                'operation':
                cg.XMON.serialize_op_dict(cirq.X(q0)),
                'start_time_picos':
                200000,
            }, {
                'operation':
                cg.XMON.serialize_op_dict(cirq.Z(q1)),
                'start_time_picos':
                200000,
            }, {
                'operation':
                cg.XMON.serialize_op_dict(cirq.measure(q0, key='a')),
                'start_time_picos':
                400000,
            }]
        },
    }
    assert cg.XMON.deserialize_dict(serialized, cg.Bristlecone) == schedule
