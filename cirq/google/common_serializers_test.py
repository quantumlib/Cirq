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
import pytest
import numpy as np
import sympy

import cirq
import cirq.google as cg
import cirq.google.common_serializers as cgc

SINGLE_QUBIT_GATE_SET = cg.serializable_gate_set.SerializableGateSet(
    gate_set_name='test_half_pi',
    serializers=([cgc.MEASUREMENT_SERIALIZER] + cgc.SINGLE_QUBIT_SERIALIZERS),
    deserializers=([cgc.MEASUREMENT_DESERIALIZER] +
                   cgc.SINGLE_QUBIT_DESERIALIZERS))

HALF_PI_GATE_SET = cg.serializable_gate_set.SerializableGateSet(
    gate_set_name='test_half_pi',
    serializers=([cgc.MEASUREMENT_SERIALIZER] +
                 cgc.SINGLE_QUBIT_HALF_PI_SERIALIZERS),
    deserializers=([cgc.MEASUREMENT_DESERIALIZER] +
                   cgc.SINGLE_QUBIT_HALF_PI_DESERIALIZERS))


@pytest.mark.parametrize('phase_exponent', (0, 0.25, 0.75))
def test_serialize_deserialize_phased_x_pi_gate(phase_exponent):
    proto_dict = {
        'gate': {
            'id': 'xy_pi'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': phase_exponent
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    op = cirq.PhasedXPowGate(phase_exponent=phase_exponent)(q)
    assert HALF_PI_GATE_SET.serialize_op_dict(op) == proto_dict
    assert HALF_PI_GATE_SET.deserialize_op_dict(proto_dict) == op


@pytest.mark.parametrize('phase_exponent', (0, 0.25, 0.75))
def test_serialize_deserialize_phased_x_half_pi_gate(phase_exponent):
    proto_dict = {
        'gate': {
            'id': 'xy_half_pi'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': phase_exponent
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    op = cirq.PhasedXPowGate(exponent=0.5, phase_exponent=phase_exponent)(q)
    assert HALF_PI_GATE_SET.serialize_op_dict(op) == proto_dict
    assert HALF_PI_GATE_SET.deserialize_op_dict(proto_dict) == op


def test_serialize_x_pow_gate():
    proto_dict = {
        'gate': {
            'id': 'xy_pi'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': 0
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    op = cirq.XPowGate(exponent=1)(q)
    assert HALF_PI_GATE_SET.serialize_op_dict(op) == proto_dict
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(
        HALF_PI_GATE_SET.deserialize_op_dict(proto_dict)),
                                                    cirq.unitary(op),
                                                    atol=1e-6)


def test_serialize_y_pow_gate():
    proto_dict = {
        'gate': {
            'id': 'xy_pi'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': 0.5
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    op = cirq.YPowGate(exponent=1)(q)
    assert HALF_PI_GATE_SET.serialize_op_dict(op) == proto_dict
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(
        HALF_PI_GATE_SET.deserialize_op_dict(proto_dict)),
                                                    cirq.unitary(op),
                                                    atol=1e-6)


def test_serialize_sqrt_x_pow_gate():
    proto_dict = {
        'gate': {
            'id': 'xy_half_pi'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': 0
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    op = cirq.XPowGate(exponent=0.5)(q)
    assert HALF_PI_GATE_SET.serialize_op_dict(op) == proto_dict
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(
        HALF_PI_GATE_SET.deserialize_op_dict(proto_dict)),
                                                    cirq.unitary(op),
                                                    atol=1e-6)


def test_serialize_sqrt_y_pow_gate():
    proto_dict = {
        'gate': {
            'id': 'xy_half_pi'
        },
        'args': {
            'axis_half_turns': {
                'arg_value': {
                    'float_value': 0.5
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    op = cirq.YPowGate(exponent=0.5)(q)
    assert HALF_PI_GATE_SET.serialize_op_dict(op) == proto_dict
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(
        HALF_PI_GATE_SET.deserialize_op_dict(proto_dict)),
                                                    cirq.unitary(op),
                                                    atol=1e-6)


@pytest.mark.parametrize(('gate', 'axis_half_turns', 'half_turns'), [
    (cirq.X**0.25, 0.0, 0.25),
    (cirq.Y**0.25, 0.5, 0.25),
    (cirq.XPowGate(exponent=0.125), 0.0, 0.125),
    (cirq.YPowGate(exponent=0.125), 0.5, 0.125),
    (cirq.PhasedXPowGate(exponent=0.125, phase_exponent=0.25), 0.25, 0.125),
    (cirq.rx(0.125 * np.pi), 0.0, 0.125),
    (cirq.ry(0.25 * np.pi), 0.5, 0.25),
])
def test_serialize_deserialize_arbitrary_xy(gate, axis_half_turns, half_turns):
    op = gate.on(cirq.GridQubit(1, 2))
    expected = {
        'gate': {
            'id': 'xy'
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
    assert SINGLE_QUBIT_GATE_SET.serialize_op_dict(op) == expected
    deserialized_op = SINGLE_QUBIT_GATE_SET.deserialize_op_dict(expected)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(deserialized_op),
        cirq.unitary(op),
        atol=1e-7,
    )


def test_half_pi_does_not_serialize_arbitrary_xy():
    q = cirq.GridQubit(1, 2)
    gate = cirq.PhasedXPowGate(exponent=0.125, phase_exponent=0.25)
    with pytest.raises(ValueError):
        HALF_PI_GATE_SET.serialize_op_dict(gate(q))

    gate = cirq.PhasedXPowGate(exponent=sympy.Symbol('a'),
                               phase_exponent=sympy.Symbol('b'))
    with pytest.raises(ValueError):
        HALF_PI_GATE_SET.serialize_op_dict(gate(q))


@pytest.mark.parametrize(('x_exponent', 'z_exponent', 'axis_phase_exponent'), [
    (0, 0, 0),
    (1, 0, 0),
    (0, 1, 0),
    (0.5, 0, 0.5),
    (0.5, 0.5, 0.5),
    (0.25, 0.375, 0.125),
])
def test_serialize_deserialize_arbitrary_xyz(
        x_exponent,
        z_exponent,
        axis_phase_exponent,
):
    gate = cirq.PhasedXZGate(
        x_exponent=x_exponent,
        z_exponent=z_exponent,
        axis_phase_exponent=axis_phase_exponent,
    )
    op = gate.on(cirq.GridQubit(1, 2))
    expected = {
        'gate': {
            'id': 'xyz'
        },
        'args': {
            'x_exponent': {
                'arg_value': {
                    'float_value': x_exponent
                }
            },
            'z_exponent': {
                'arg_value': {
                    'float_value': z_exponent
                }
            },
            'axis_phase_exponent': {
                'arg_value': {
                    'float_value': axis_phase_exponent
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    assert SINGLE_QUBIT_GATE_SET.serialize_op_dict(op) == expected
    deserialized_op = SINGLE_QUBIT_GATE_SET.deserialize_op_dict(expected)
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(deserialized_op),
        cirq.unitary(op),
        atol=1e-7,
    )


@pytest.mark.parametrize(('qubits', 'qubit_ids', 'key', 'invert_mask'), [
    ([cirq.GridQubit(1, 1)], ['1_1'], 'a', ()),
    ([cirq.GridQubit(1, 2)], ['1_2'], 'b', (True,)),
    ([cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)], ['1_1', '1_2'], 'a',
     (True, False)),
])
def test_serialize_deserialize_meas(qubits, qubit_ids, key, invert_mask):
    op = cirq.measure(*qubits, key=key, invert_mask=invert_mask)
    proto_dict = {
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
        proto_dict['qubits'].append({'id': qubit_id})
    assert SINGLE_QUBIT_GATE_SET.serialize_op_dict(op) == proto_dict
    assert SINGLE_QUBIT_GATE_SET.deserialize_op_dict(proto_dict) == op
    assert HALF_PI_GATE_SET.serialize_op_dict(op) == proto_dict
    assert HALF_PI_GATE_SET.deserialize_op_dict(proto_dict) == op


@pytest.mark.parametrize(('gate', 'axis_half_turns', 'half_turns'), [
    (cirq.X, 0.0, 1.0),
    (cirq.X**0.25, 0.0, 0.25),
    (cirq.Y, 0.5, 1.0),
    (cirq.Y**0.25, 0.5, 0.25),
    (cirq.PhasedXPowGate(exponent=0.125, phase_exponent=0.25), 0.25, 0.125),
    (cirq.rx(0.125 * np.pi), 0.0, 0.125),
    (cirq.ry(0.25 * np.pi), 0.5, 0.25),
])
def test_serialize_xy(gate, axis_half_turns, half_turns):
    q = cirq.GridQubit(1, 2)
    expected = {
        'gate': {
            'id': 'xy'
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

    assert SINGLE_QUBIT_GATE_SET.serialize_op_dict(gate.on(q)) == expected


@pytest.mark.parametrize(('gate', 'axis_half_turns', 'half_turns'), [
    (cirq.X**sympy.Symbol('a'), 0.0, 'a'),
    (cirq.Y**sympy.Symbol('b'), 0.5, 'b'),
    (cirq.PhasedXPowGate(exponent=sympy.Symbol('a'),
                         phase_exponent=0.25), 0.25, 'a'),
])
def test_serialize_xy_parameterized_half_turns(gate, axis_half_turns,
                                               half_turns):
    q = cirq.GridQubit(1, 2)
    expected = {
        'gate': {
            'id': 'xy'
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

    assert SINGLE_QUBIT_GATE_SET.serialize_op_dict(gate.on(q)) == expected


def test_serialize_xy_parameterized_axis_half_turns():
    gate = cirq.PhasedXPowGate(exponent=0.25, phase_exponent=sympy.Symbol('a'))
    q = cirq.GridQubit(1, 2)
    expected = {
        'gate': {
            'id': 'xy'
        },
        'args': {
            'axis_half_turns': {
                'symbol': 'a'
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

    assert SINGLE_QUBIT_GATE_SET.serialize_op_dict(gate.on(q)) == expected


@pytest.mark.parametrize(('gate', 'half_turns'), [
    (cirq.Z, 1.0),
    (cirq.Z**0.125, 0.125),
    (cirq.rz(0.125 * np.pi), 0.125),
])
def test_serialize_z(gate, half_turns):
    q = cirq.GridQubit(1, 2)
    assert SINGLE_QUBIT_GATE_SET.serialize_op_dict(gate.on(q)) == {
        'gate': {
            'id': 'z'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            },
            'type': {
                'arg_value': {
                    'string_value': 'virtual_propagates_forward'
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }


@pytest.mark.parametrize(('axis_half_turns', 'half_turns'), [
    (0.25, 0.25),
    (0, 0.25),
    (0.5, 0.25),
])
def test_deserialize_xy(axis_half_turns, half_turns):
    serialized_op = {
        'gate': {
            'id': 'xy'
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
    actual = SINGLE_QUBIT_GATE_SET.deserialize_op_dict(serialized_op)
    assert actual == expected


def test_deserialize_xy_parameterized():
    serialized_op = {
        'gate': {
            'id': 'xy'
        },
        'args': {
            'axis_half_turns': {
                'symbol': 'a'
            },
            'half_turns': {
                'symbol': 'b'
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    expected = cirq.PhasedXPowGate(exponent=sympy.Symbol('b'),
                                   phase_exponent=sympy.Symbol('a'))(q)
    assert (
        SINGLE_QUBIT_GATE_SET.deserialize_op_dict(serialized_op) == expected)


@pytest.mark.parametrize('half_turns', [0, 0.25, 1.0])
def test_deserialize_z(half_turns):
    serialized_op = {
        'gate': {
            'id': 'z'
        },
        'args': {
            'half_turns': {
                'arg_value': {
                    'float_value': half_turns
                }
            },
            'type': {
                'arg_value': {
                    'string_value': 'virtual_propagates_forward'
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    expected = cirq.ZPowGate(exponent=half_turns)(q)
    assert SINGLE_QUBIT_GATE_SET.deserialize_op_dict(serialized_op) == expected


def test_deserialize_z_parameterized():
    serialized_op = {
        'gate': {
            'id': 'z'
        },
        'args': {
            'half_turns': {
                'symbol': 'a'
            },
            'type': {
                'arg_value': {
                    'string_value': 'virtual_propagates_forward'
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    expected = cirq.ZPowGate(exponent=sympy.Symbol('a'))(q)
    assert SINGLE_QUBIT_GATE_SET.deserialize_op_dict(serialized_op) == expected


def test_wait_gate():
    gate_set = cg.SerializableGateSet('test', [cgc.WAIT_GATE_SERIALIZER],
                                      [cgc.WAIT_GATE_DESERIALIZER])
    proto_dict = {
        'gate': {
            'id': 'wait'
        },
        'args': {
            'nanos': {
                'arg_value': {
                    'float_value': 20.0
                }
            }
        },
        'qubits': [{
            'id': '1_2'
        }]
    }
    q = cirq.GridQubit(1, 2)
    op = cirq.WaitGate(cirq.Duration(nanos=20)).on(q)
    assert gate_set.serialize_op_dict(op) == proto_dict
    assert gate_set.deserialize_op_dict(proto_dict) == op
