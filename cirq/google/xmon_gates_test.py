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

import pytest
import numpy as np

import cirq
import cirq.google as cg


def assert_proto_dict_convert(gate_cls, gate, proto_dict, *qubits):
    assert gate.to_proto_dict(*qubits) == proto_dict
    assert gate_cls.from_proto_dict(proto_dict) == gate(*qubits)


def test_parameterized_value_from_proto():
    from_proto = cg.XmonGate.parameterized_value_from_proto_dict

    m1 = {'raw': 5}
    assert from_proto(m1) == 5

    with pytest.raises(ValueError):
        from_proto({})

    m3 = {'parameter_key': 'rr'}
    assert from_proto(m3) == cirq.Symbol('rr')


def test_measurement_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cg.XmonMeasurementGate(key=''))
    eq.make_equality_group(lambda: cg.XmonMeasurementGate('a'))
    eq.make_equality_group(lambda: cg.XmonMeasurementGate('b'))
    eq.make_equality_group(lambda: cg.XmonMeasurementGate(key='',
                                                          invert_mask=(True,)))
    eq.make_equality_group(lambda: cg.XmonMeasurementGate(key='',
                                                          invert_mask=(False,)))


def test_single_qubit_measurement_proto_dict_convert():
    gate = cg.XmonMeasurementGate('test')
    proto_dict = {
        'measurement': {
            'targets': [
                {
                    'row': 2,
                    'col': 3
                }
            ],
            'key': 'test'
        }
    }
    assert_proto_dict_convert(cg.XmonMeasurementGate, gate, proto_dict,
                              cirq.GridQubit(2, 3))


def test_single_qubit_measurement_invalid_dict():
    proto_dict = {
        'measurement': {
            'targets': [
                {
                    'row': 2,
                    'col': 3
                }
            ],
        }
    }
    with pytest.raises(ValueError):
        cg.XmonMeasurementGate.from_proto_dict(proto_dict)

    proto_dict = {
        'measurement': {
            'targets': [
                {
                    'row': 2,
                    'col': 3
                }
            ],
        }
    }
    with pytest.raises(ValueError):
        cg.XmonMeasurementGate.from_proto_dict(proto_dict)


def test_single_qubit_measurement_to_proto_dict_convert_invert_mask():
    gate = cg.XmonMeasurementGate('test', invert_mask=(True,))
    proto_dict = {
        'measurement': {
            'targets': [
                {
                    'row': 2,
                    'col': 3
                }
            ],
            'key': 'test',
            'invert_mask': ['true']
        }
    }
    assert_proto_dict_convert(cg.XmonMeasurementGate, gate, proto_dict,
                              cirq.GridQubit(2, 3))


def test_multi_qubit_measurement_to_proto_dict():
    gate = cg.XmonMeasurementGate('test')
    proto_dict = {
        'measurement': {
            'targets': [
                {
                    'row': 2,
                    'col': 3
                },
                {
                    'row': 3,
                    'col': 4
                }
            ],
            'key': 'test'
        }
    }
    assert_proto_dict_convert(cg.XmonMeasurementGate, gate, proto_dict,
                              cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))


@cirq.testing.only_test_in_python3
def test_measurement_repr():
    gate = cg.XmonMeasurementGate('test', invert_mask=(True,))
    assert repr(gate) == 'XmonMeasurementGate(\'test\', (True,))'


def test_invalid_measurement_gate():
    with pytest.raises(ValueError, match='length'):
        cg.XmonMeasurementGate('test', invert_mask=(True,)).to_proto_dict(
            cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))
    with pytest.raises(ValueError, match='no qubits'):
        cg.XmonMeasurementGate('test').to_proto_dict()


def test_z_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cg.ExpZGate(half_turns=0))
    eq.add_equality_group(cg.ExpZGate(),
                          cg.ExpZGate(half_turns=1),
                          cg.ExpZGate(degs=180),
                          cg.ExpZGate(rads=np.pi))
    eq.make_equality_group(
        lambda: cg.ExpZGate(half_turns=cirq.Symbol('a')))
    eq.make_equality_group(
        lambda: cg.ExpZGate(half_turns=cirq.Symbol('b')))
    eq.add_equality_group(
        cg.ExpZGate(half_turns=-1.5),
        cg.ExpZGate(half_turns=10.5))


def test_z_proto_dict_convert():
    gate = cg.ExpZGate(half_turns=cirq.Symbol('k'))
    proto_dict = {
        'exp_z': {
            'target': {
                'row': 2,
                'col': 3
            },
            'half_turns': {
                'parameter_key': 'k'
            }
        }
    }
    assert_proto_dict_convert(cg.ExpZGate, gate, proto_dict,
                              cirq.GridQubit(2, 3))

    gate = cg.ExpZGate(half_turns=0.5)
    proto_dict = {
        'exp_z': {
            'target': {
                'row': 2,
                'col': 3
            },
            'half_turns': {
                'raw': 0.5
            }
        }
    }
    assert_proto_dict_convert(cg.ExpZGate, gate, proto_dict,
                              cirq.GridQubit(2, 3))


def test_z_invalid_dict():
    proto_dict = {
        'exp_z': {
            'target': {
                'row': 2,
                'col': 3
            },
        }
    }
    with pytest.raises(ValueError):
        cg.ExpZGate.from_proto_dict(proto_dict)

    proto_dict = {
        'exp_z': {
            'half_turns': {
                'parameter_key': 'k'
            }
        }
    }
    with pytest.raises(ValueError):
        cg.ExpZGate.from_proto_dict(proto_dict)


def test_z_matrix():
    assert np.allclose(cirq.unitary(cg.ExpZGate(half_turns=1)),
                       np.array([[-1j, 0], [0, 1j]]))
    assert np.allclose(cirq.unitary(cg.ExpZGate(half_turns=0.5)),
                       np.array([[1 - 1j, 0], [0, 1 + 1j]]) / np.sqrt(2))
    assert np.allclose(cirq.unitary(cg.ExpZGate(half_turns=0)),
                       np.array([[1, 0], [0, 1]]))
    assert np.allclose(cirq.unitary(cg.ExpZGate(half_turns=-0.5)),
                       np.array([[1 + 1j, 0], [0, 1 - 1j]]) / np.sqrt(2))


def test_z_parameterize():
    parameterized_gate = cg.ExpZGate(half_turns=cirq.Symbol('a'))
    assert parameterized_gate.is_parameterized()
    assert cirq.unitary(parameterized_gate, None) is None
    resolver = cirq.ParamResolver({'a': 0.1})
    resolved_gate = parameterized_gate.with_parameters_resolved_by(resolver)
    assert resolved_gate == cg.ExpZGate(half_turns=0.1)


def test_z_repr():
    gate = cg.ExpZGate(half_turns=0.1)
    assert repr(gate) == 'ExpZGate(half_turns=0.1)'


def test_cz_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cg.Exp11Gate(half_turns=0))
    eq.add_equality_group(cg.Exp11Gate(),
                          cg.Exp11Gate(half_turns=1),
                          cg.Exp11Gate(degs=180),
                          cg.Exp11Gate(rads=np.pi))
    eq.make_equality_group(lambda: cg.Exp11Gate(half_turns=cirq.Symbol('a')))
    eq.make_equality_group(lambda: cg.Exp11Gate(half_turns=cirq.Symbol('b')))
    eq.add_equality_group(
        cg.Exp11Gate(half_turns=-1.5),
        cg.Exp11Gate(half_turns=6.5))


def test_cz_proto_dict_convert():
    gate = cg.Exp11Gate(half_turns=cirq.Symbol('k'))
    proto_dict = {
        'exp_11': {
            'target1': {
                'row': 2,
                'col': 3
            },
            'target2': {
                'row': 3,
                'col': 4
            },
            'half_turns': {
                'parameter_key': 'k'
            }
        }
    }
    assert_proto_dict_convert(cg.Exp11Gate, gate, proto_dict,
                              cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))

    gate = cg.Exp11Gate(half_turns=0.5)
    proto_dict = {
        'exp_11': {
            'target1': {
                'row': 2,
                'col': 3
            },
            'target2': {
                'row': 3,
                'col': 4
            },
            'half_turns': {
                'raw': 0.5
            }
        }
    }
    assert_proto_dict_convert(cg.Exp11Gate, gate, proto_dict,
                              cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))


def test_cz_invalid_dict():
    proto_dict = {
        'exp_11': {
            'target2': {
                'row': 3,
                'col': 4
            },
            'half_turns': {
                'parameter_key': 'k'
            }
        }
    }
    with pytest.raises(ValueError):
        cg.Exp11Gate.from_proto_dict(proto_dict)

    proto_dict = {
        'exp_11': {
            'target1': {
                'row': 2,
                'col': 3
            },
            'half_turns': {
                'parameter_key': 'k'
            }
        }
    }
    with pytest.raises(ValueError):
        cg.Exp11Gate.from_proto_dict(proto_dict)

    proto_dict = {
        'exp_11': {
            'target1': {
                'row': 2,
                'col': 3
            },
            'target2': {
                'row': 3,
                'col': 4
            },
        }
    }
    with pytest.raises(ValueError):
        cg.Exp11Gate.from_proto_dict(proto_dict)


def test_cz_potential_implementation():
    assert cirq.unitary(cg.Exp11Gate(half_turns=cirq.Symbol('a')), None) is None
    assert cirq.unitary(cg.Exp11Gate()) is not None


def test_cz_parameterize():
    parameterized_gate = cg.Exp11Gate(half_turns=cirq.Symbol('a'))
    assert parameterized_gate.is_parameterized()
    assert cirq.unitary(parameterized_gate, None) is None
    resolver = cirq.ParamResolver({'a': 0.1})
    resolved_gate = parameterized_gate.with_parameters_resolved_by(resolver)
    assert resolved_gate == cg.Exp11Gate(half_turns=0.1)


def test_cz_repr():
    gate = cg.Exp11Gate(half_turns=0.1)
    assert repr(gate) == 'Exp11Gate(half_turns=0.1)'


def test_w_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cg.ExpWGate(),
                          cg.ExpWGate(half_turns=1, axis_half_turns=0),
                          cg.ExpWGate(degs=180, axis_degs=0),
                          cg.ExpWGate(rads=np.pi, axis_rads=0))
    eq.make_equality_group(
        lambda: cg.ExpWGate(half_turns=cirq.Symbol('a')))
    eq.make_equality_group(lambda: cg.ExpWGate(half_turns=0))
    eq.make_equality_group(
        lambda: cg.ExpWGate(half_turns=0, axis_half_turns=cirq.Symbol('a')))
    eq.add_equality_group(
        cg.ExpWGate(half_turns=0, axis_half_turns=0.5),
        cg.ExpWGate(half_turns=0, axis_rads=np.pi / 2))
    eq.make_equality_group(
        lambda: cg.ExpWGate(
            half_turns=cirq.Symbol('ab'),
            axis_half_turns=cirq.Symbol('xy')))

    # Flipping the axis and negating the angle gives the same rotation.
    eq.add_equality_group(
        cg.ExpWGate(half_turns=0.25, axis_half_turns=1.5),
        cg.ExpWGate(half_turns=1.75, axis_half_turns=0.5))
    # ...but not when there are parameters.
    eq.add_equality_group(cg.ExpWGate(
        half_turns=cirq.Symbol('a'),
        axis_half_turns=1.5))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=cirq.Symbol('a'),
        axis_half_turns=0.5))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=0.25,
        axis_half_turns=cirq.Symbol('a')))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=1.75,
        axis_half_turns=cirq.Symbol('a')))

    # Adding or subtracting whole turns/phases gives the same rotation.
    eq.add_equality_group(
        cg.ExpWGate(half_turns=-2.25, axis_half_turns=1.25),
        cg.ExpWGate(half_turns=7.75, axis_half_turns=11.25))


def test_w_str():
    assert str(cg.ExpWGate()) == 'X'
    assert str(cg.ExpWGate(axis_half_turns=0.99999, half_turns=0.5)) == 'X^-0.5'
    assert str(cg.ExpWGate(axis_half_turns=0.5, half_turns=0.25)) == 'Y^0.25'
    assert str(cg.ExpWGate(axis_half_turns=0.25,
                           half_turns=0.5)) == 'W(0.25)^0.5'


def test_w_to_proto_dict():
    gate = cg.ExpWGate(half_turns=cirq.Symbol('k'), axis_half_turns=1)
    proto_dict = {
        'exp_w': {
            'target': {
                'row': 2,
                'col': 3
            },
            'axis_half_turns': {
                'raw': 1
            },
            'half_turns': {
                'parameter_key': 'k'
            }
        }
    }
    assert_proto_dict_convert(cg.ExpWGate, gate, proto_dict,
                              cirq.GridQubit(2, 3))

    gate = cg.ExpWGate(half_turns=0.5, axis_half_turns=cirq.Symbol('j'))
    proto_dict = {
        'exp_w': {
            'target': {
                'row': 2,
                'col': 3
            },
            'axis_half_turns': {
                'parameter_key': 'j'
            },
            'half_turns': {
                'raw': 0.5
            }
        }
    }
    assert_proto_dict_convert(cg.ExpWGate, gate, proto_dict,
                              cirq.GridQubit(2, 3))


def test_w_invalid_dict():
    proto_dict = {
        'exp_w': {
            'axis_half_turns': {
                'raw': 1
            },
            'half_turns': {
                'parameter_key': 'k'
            }
        }
    }
    with pytest.raises(ValueError):
        cg.ExpWGate.from_proto_dict(proto_dict)

    proto_dict = {
        'exp_w': {
            'target': {
                'row': 2,
                'col': 3
            },
            'half_turns': {
                'parameter_key': 'k'
            }
        }
    }
    with pytest.raises(ValueError):
        cg.ExpWGate.from_proto_dict(proto_dict)

    proto_dict = {
        'exp_w': {
            'target': {
                'row': 2,
                'col': 3
            },
            'axis_half_turns': {
                'raw': 1
            },
        }
    }
    with pytest.raises(ValueError):
        cg.ExpWGate.from_proto_dict(proto_dict)


def test_w_decomposition():
    q = cirq.NamedQubit('q')
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(
            cg.ExpWGate(half_turns=0.25, axis_half_turns=0.3).on(q)
        ).to_unitary_matrix(),
        cirq.Circuit.from_ops(
            cirq.Z(q) ** -0.3,
            cirq.X(q) ** 0.25,
            cirq.Z(q) ** 0.3
        ).to_unitary_matrix(),
        atol=1e-8)


def test_w_potential_implementation():
    assert not cirq.can_cast(cirq.ReversibleEffect,
                             cg.ExpWGate(half_turns=cirq.Symbol('a')))
    assert cirq.can_cast(cirq.ReversibleEffect, cg.ExpWGate())


def test_w_parameterize():
    parameterized_gate = cg.ExpWGate(half_turns=cirq.Symbol('a'),
                                     axis_half_turns=cirq.Symbol('b'))
    assert parameterized_gate.is_parameterized()
    assert cirq.unitary(parameterized_gate, None) is None
    resolver = cirq.ParamResolver({'a': 0.1, 'b': 0.2})
    resolved_gate = parameterized_gate.with_parameters_resolved_by(resolver)
    assert resolved_gate == cg.ExpWGate(half_turns=0.1, axis_half_turns=0.2)


def test_w_repr():
    gate = cg.ExpWGate(half_turns=0.1, axis_half_turns=0.2)
    assert repr(gate) == 'ExpWGate(half_turns=0.1, axis_half_turns=0.2)'


def test_trace_bound():
    assert cg.ExpZGate(half_turns=.001).trace_distance_bound() < 0.01
    assert cg.ExpWGate(half_turns=.001).trace_distance_bound() < 0.01
    assert cg.ExpZGate(
        half_turns=cirq.Symbol('a')).trace_distance_bound() >= 1
    assert cg.ExpWGate(
        half_turns=cirq.Symbol('a')).trace_distance_bound() >= 1


def test_has_inverse():
    assert cg.ExpZGate(half_turns=.1).has_inverse()
    assert cg.ExpWGate(half_turns=.1).has_inverse()
    assert not cg.ExpZGate(half_turns=cirq.Symbol('a')).has_inverse()
    assert not cg.ExpWGate(half_turns=cirq.Symbol('a')).has_inverse()


def test_measure_key_on():
    q = cirq.GridQubit(0, 0)

    assert cg.XmonMeasurementGate(key='').on(q) == cirq.GateOperation(
        gate=cg.XmonMeasurementGate(key=''),
        qubits=(q,))
    assert cg.XmonMeasurementGate(key='a').on(q) == cirq.GateOperation(
        gate=cg.XmonMeasurementGate(key='a'),
        qubits=(q,))


def test_unsupported_op():
    proto_dict = {
        'not_a_gate': {
            'target': {
                'row': 2,
                'col': 3
            },
        }
    }
    with pytest.raises(ValueError):
        cg.XmonGate.from_proto_dict(proto_dict)


def test_invalid_to_proto_dict_qubit_number():
    with pytest.raises(ValueError):
        cg.Exp11Gate(half_turns=0.5).to_proto_dict(cirq.GridQubit(2, 3))
    with pytest.raises(ValueError):
        cg.ExpZGate(half_turns=0.5).to_proto_dict(cirq.GridQubit(2, 3),
                                                  cirq.GridQubit(3, 4))
    with pytest.raises(ValueError):
        cg.ExpWGate(half_turns=0.5, axis_half_turns=0).to_proto_dict(
            cirq.GridQubit(2, 3), cirq.GridQubit(3, 4))


def test_cirq_symbol_diagrams():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    c = cirq.Circuit.from_ops(
        cg.ExpWGate(axis_half_turns=cirq.Symbol('a'),
                    half_turns=cirq.Symbol('b')).on(q00),
        cg.ExpZGate(half_turns=cirq.Symbol('c')).on(q01),
        cg.Exp11Gate(half_turns=cirq.Symbol('d')).on(q00, q01),
    )
    cirq.testing.assert_has_diagram(c, """
(0, 0): ───W(a)^b───@─────
                    │
(0, 1): ───Z^c──────@^d───
""")


def test_z_diagram_chars():
    q = cirq.GridQubit(0, 1)
    c = cirq.Circuit.from_ops(
        cg.ExpZGate().on(q),
        cg.ExpZGate(half_turns=0.5).on(q),
        cg.ExpZGate(half_turns=0.25).on(q),
        cg.ExpZGate(half_turns=0.125).on(q),
        cg.ExpZGate(half_turns=-0.5).on(q),
        cg.ExpZGate(half_turns=-0.25).on(q),
    )
    cirq.testing.assert_has_diagram(c, """
(0, 1): ───Z───S───T───Z^0.125───S^-1───T^-1───
""")


def test_w_diagram_chars():
    q = cirq.GridQubit(0, 1)
    c = cirq.Circuit.from_ops(
        cg.ExpWGate(axis_half_turns=0).on(q),
        cg.ExpWGate(axis_half_turns=0.25).on(q),
        cg.ExpWGate(axis_half_turns=0.5).on(q),
        cg.ExpWGate(axis_half_turns=cirq.Symbol('a')).on(q),
    )
    cirq.testing.assert_has_diagram(c, """
(0, 1): ───X───W(0.25)───Y───W(a)───
""")
