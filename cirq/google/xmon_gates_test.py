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
from google.protobuf import message, text_format

import cirq
import cirq.google as cg
from cirq.api.google.v1 import operations_pb2
from cirq.devices import GridQubit
from cirq.study import ParamResolver
from cirq.value import Symbol


def proto_matches_text(proto: message, expected_as_text: str):
    expected = text_format.Merge(expected_as_text, type(proto)())
    return str(proto) == str(expected)


def test_parameterized_value_from_proto():
    from_proto = cg.XmonGate.parameterized_value_from_proto

    m1 = operations_pb2.ParameterizedFloat(raw=5)
    assert from_proto(m1) == 5

    with pytest.raises(ValueError):
        from_proto(operations_pb2.ParameterizedFloat())

    m3 = operations_pb2.ParameterizedFloat(parameter_key='rr')
    assert from_proto(m3) == Symbol('rr')


def test_measurement_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cg.XmonMeasurementGate(key=''))
    eq.make_equality_group(lambda: cg.XmonMeasurementGate('a'))
    eq.make_equality_group(lambda: cg.XmonMeasurementGate('b'))
    eq.make_equality_group(lambda: cg.XmonMeasurementGate(key='',
                                                         invert_mask=(True,)))
    eq.make_equality_group(lambda: cg.XmonMeasurementGate(key='',
                                                         invert_mask=(False,)))


def test_single_qubit_measurement_to_proto():
    assert proto_matches_text(
        cg.XmonMeasurementGate('test').to_proto(GridQubit(2, 3)),
        """
        measurement {
            targets {
                row: 2
                col: 3
            }
            key: "test"
        }
        """)
    assert proto_matches_text(
        cg.XmonMeasurementGate('test', invert_mask=[True])
            .to_proto(GridQubit(2, 3)),
        """
        measurement {
            targets {
                row: 2
                col: 3
            }
            key: "test"
            invert_mask: true
        }
        """)


def test_multi_qubit_measurement_to_proto():
    assert proto_matches_text(
        cg.XmonMeasurementGate('test').to_proto(
            GridQubit(2, 3), GridQubit(3, 4)),
        """
        measurement {
            targets {
                row: 2
                col: 3
            }
            targets {
                row: 3
                col: 4
            }
            key: "test"
        }
        """)


def test_invalid_measurement_gate():
    with pytest.raises(ValueError, match='length'):
        cg.XmonMeasurementGate('test', invert_mask=[True]).to_proto(
            GridQubit(2, 3), GridQubit(3, 4))
    with pytest.raises(ValueError, match='no qubits'):
        cg.XmonMeasurementGate('test').to_proto()


def test_z_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cg.ExpZGate(half_turns=0))
    eq.add_equality_group(cg.ExpZGate(),
                          cg.ExpZGate(half_turns=1),
                          cg.ExpZGate(degs=180),
                          cg.ExpZGate(rads=np.pi))
    eq.make_equality_group(
        lambda: cg.ExpZGate(half_turns=Symbol('a')))
    eq.make_equality_group(
        lambda: cg.ExpZGate(half_turns=Symbol('b')))
    eq.add_equality_group(
        cg.ExpZGate(half_turns=-1.5),
        cg.ExpZGate(half_turns=10.5))


def test_z_to_proto():
    assert proto_matches_text(
        cg.ExpZGate(half_turns=Symbol('k')).to_proto(GridQubit(2, 3)),
        """
        exp_z {
            target {
                row: 2
                col: 3
            }
            half_turns {
                parameter_key: "k"
            }
        }
        """)

    assert proto_matches_text(
        cg.ExpZGate(half_turns=0.5).to_proto(GridQubit(2, 3)),
        """
        exp_z {
            target {
                row: 2
                col: 3
            }
            half_turns {
                raw: 0.5
            }
        }
        """)


def test_z_matrix():
    assert np.allclose(cg.ExpZGate(half_turns=1).matrix(),
                       np.array([[-1j, 0], [0, 1j]]))
    assert np.allclose(cg.ExpZGate(half_turns=0.5).matrix(),
                       np.array([[1 - 1j, 0], [0, 1 + 1j]]) / np.sqrt(2))
    assert np.allclose(cg.ExpZGate(half_turns=0).matrix(),
                       np.array([[1, 0], [0, 1]]))
    assert np.allclose(cg.ExpZGate(half_turns=-0.5).matrix(),
                       np.array([[1 + 1j, 0], [0, 1 - 1j]]) / np.sqrt(2))


def test_z_parameterize():
    parameterized_gate = cg.ExpZGate(half_turns=Symbol('a'))
    assert parameterized_gate.is_parameterized()
    with pytest.raises(ValueError):
        _ = parameterized_gate.matrix()
    resolver = ParamResolver({'a': 0.1})
    resolved_gate = parameterized_gate.with_parameters_resolved_by(resolver)
    assert resolved_gate == cg.ExpZGate(half_turns=0.1)


def test_cz_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cg.Exp11Gate(half_turns=0))
    eq.add_equality_group(cg.Exp11Gate(),
                          cg.Exp11Gate(half_turns=1),
                          cg.Exp11Gate(degs=180),
                          cg.Exp11Gate(rads=np.pi))
    eq.make_equality_group(lambda: cg.Exp11Gate(half_turns=Symbol('a')))
    eq.make_equality_group(lambda: cg.Exp11Gate(half_turns=Symbol('b')))
    eq.add_equality_group(
        cg.Exp11Gate(half_turns=-1.5),
        cg.Exp11Gate(half_turns=6.5))


def test_cz_to_proto():
    assert proto_matches_text(
        cg.Exp11Gate(half_turns=Symbol('k')).to_proto(
            GridQubit(2, 3), GridQubit(4, 5)),
        """
        exp_11 {
            target1 {
                row: 2
                col: 3
            }
            target2 {
                row: 4
                col: 5
            }
            half_turns {
                parameter_key: "k"
            }
        }
        """)

    assert proto_matches_text(
        cg.Exp11Gate(half_turns=0.5).to_proto(
            GridQubit(2, 3), GridQubit(4, 5)),
        """
        exp_11 {
            target1 {
                row: 2
                col: 3
            }
            target2 {
                row: 4
                col: 5
            }
            half_turns {
                raw: 0.5
            }
        }
        """)


def test_cz_potential_implementation():
    assert not cirq.can_cast(cirq.KnownMatrix,
                             cg.Exp11Gate(half_turns=Symbol('a')))
    assert cirq.can_cast(cirq.KnownMatrix, cg.Exp11Gate())


def test_cz_parameterize():
    parameterized_gate = cg.Exp11Gate(half_turns=Symbol('a'))
    assert parameterized_gate.is_parameterized()
    with pytest.raises(ValueError):
        _ = parameterized_gate.matrix()
    resolver = ParamResolver({'a': 0.1})
    resolved_gate = parameterized_gate.with_parameters_resolved_by(resolver)
    assert resolved_gate == cg.Exp11Gate(half_turns=0.1)


def test_w_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cg.ExpWGate(),
                          cg.ExpWGate(half_turns=1, axis_half_turns=0),
                          cg.ExpWGate(degs=180, axis_degs=0),
                          cg.ExpWGate(rads=np.pi, axis_rads=0))
    eq.make_equality_group(
        lambda: cg.ExpWGate(half_turns=Symbol('a')))
    eq.make_equality_group(lambda: cg.ExpWGate(half_turns=0))
    eq.make_equality_group(
        lambda: cg.ExpWGate(half_turns=0, axis_half_turns=Symbol('a')))
    eq.add_equality_group(
        cg.ExpWGate(half_turns=0, axis_half_turns=0.5),
        cg.ExpWGate(half_turns=0, axis_rads=np.pi / 2))
    eq.make_equality_group(
        lambda: cg.ExpWGate(
            half_turns=Symbol('ab'),
            axis_half_turns=Symbol('xy')))

    # Flipping the axis and negating the angle gives the same rotation.
    eq.add_equality_group(
        cg.ExpWGate(half_turns=0.25, axis_half_turns=1.5),
        cg.ExpWGate(half_turns=1.75, axis_half_turns=0.5))
    # ...but not when there are parameters.
    eq.add_equality_group(cg.ExpWGate(
        half_turns=Symbol('a'),
        axis_half_turns=1.5))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=Symbol('a'),
        axis_half_turns=0.5))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=0.25,
        axis_half_turns=Symbol('a')))
    eq.add_equality_group(cg.ExpWGate(
        half_turns=1.75,
        axis_half_turns=Symbol('a')))

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



def test_w_to_proto():
    assert proto_matches_text(
        cg.ExpWGate(half_turns=Symbol('k'),
                    axis_half_turns=1).to_proto(GridQubit(2, 3)),
        """
        exp_w {
            target {
                row: 2
                col: 3
            }
            axis_half_turns {
                raw: 1
            }
            half_turns {
                parameter_key: "k"
            }
        }
        """)

    assert proto_matches_text(
        cg.ExpWGate(half_turns=0.5,
                    axis_half_turns=Symbol('j')).to_proto(GridQubit(2, 3)),
        """
        exp_w {
            target {
                row: 2
                col: 3
            }
            axis_half_turns {
                parameter_key: "j"
            }
            half_turns {
                raw: 0.5
            }
        }
        """)


def test_w_decomposition():
    q = cirq.NamedQubit('q')
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.Circuit.from_ops(
            cg.ExpWGate(half_turns=0.25, axis_half_turns=0.3).on(q)
        ).to_unitary_matrix(),
        cirq.Circuit.from_ops(
            cirq.Z(q)**-0.3,
            cirq.X(q)**0.25,
            cirq.Z(q)**0.3
        ).to_unitary_matrix(),
        atol=1e-8)


def test_w_potential_implementation():
    assert not cirq.can_cast(cirq.KnownMatrix,
                             cg.ExpWGate(half_turns=Symbol('a')))
    assert not cirq.can_cast(cirq.ReversibleEffect,
                             cg.ExpWGate(half_turns=Symbol('a')))
    assert cirq.can_cast(cirq.KnownMatrix, cg.ExpWGate())
    assert cirq.can_cast(cirq.ReversibleEffect, cg.ExpWGate())


def test_w_parameterize():
    parameterized_gate = cg.ExpWGate(half_turns=Symbol('a'),
                                     axis_half_turns=Symbol('b'))
    assert parameterized_gate.is_parameterized()
    with pytest.raises(ValueError):
        _ = parameterized_gate.matrix()
    resolver = ParamResolver({'a': 0.1, 'b': 0.2})
    resolved_gate = parameterized_gate.with_parameters_resolved_by(resolver)
    assert resolved_gate == cg.ExpWGate(half_turns=0.1, axis_half_turns=0.2)


def test_trace_bound():
    assert cg.ExpZGate(half_turns=.001).trace_distance_bound() < 0.01
    assert cg.ExpWGate(half_turns=.001).trace_distance_bound() < 0.01
    assert cg.ExpZGate(half_turns=cirq.Symbol('a')).trace_distance_bound() >= 1
    assert cg.ExpWGate(half_turns=cirq.Symbol('a')).trace_distance_bound() >= 1


def test_has_inverse():
    assert cg.ExpZGate(half_turns=.1).has_inverse()
    assert cg.ExpWGate(half_turns=.1).has_inverse()
    assert not cg.ExpZGate(half_turns=cirq.Symbol('a')).has_inverse()
    assert not cg.ExpWGate(half_turns=cirq.Symbol('a')).has_inverse()


def test_measure_key_on():
    q = GridQubit(0, 0)

    assert cg.XmonMeasurementGate(key='').on(q) == cirq.GateOperation(
        gate=cg.XmonMeasurementGate(key=''),
        qubits=(q,))
    assert cg.XmonMeasurementGate(key='a').on(q) == cirq.GateOperation(
        gate=cg.XmonMeasurementGate(key='a'),
        qubits=(q,))


def test_symbol_diagrams():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    c = cirq.Circuit.from_ops(
        cg.ExpWGate(axis_half_turns=cirq.Symbol('a'),
                             half_turns=cirq.Symbol('b')).on(q00),
        cg.ExpZGate(half_turns=cirq.Symbol('c')).on(q01),
        cg.Exp11Gate(half_turns=cirq.Symbol('d')).on(q00, q01),
    )
    assert c.to_text_diagram() == """
(0, 0): ───W(a)^b───@─────
                    │
(0, 1): ───Z^c──────@^d───
    """.strip()


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
    assert c.to_text_diagram() == """
(0, 1): ───Z───S───T───Z^0.125───S^-1───T^-1───
    """.strip()


def test_w_diagram_chars():
    q = cirq.GridQubit(0, 1)
    c = cirq.Circuit.from_ops(
        cg.ExpWGate(axis_half_turns=0).on(q),
        cg.ExpWGate(axis_half_turns=0.25).on(q),
        cg.ExpWGate(axis_half_turns=0.5).on(q),
        cg.ExpWGate(axis_half_turns=cirq.Symbol('a')).on(q),
    )
    assert c.to_text_diagram() == """
(0, 1): ───X───W(0.25)───Y───W(a)───
    """.strip()
