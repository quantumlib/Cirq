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

from cirq.ops import gate_features, raw_types


def test_reversible_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.ReversibleGate()


def test_reversible_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class Missing(gate_features.ReversibleGate):
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_reversible_gate_is_abstract_can_implement():
    class Included(gate_features.ReversibleGate):
        def inverse(self):
            pass

    assert isinstance(Included(), gate_features.ReversibleGate)


def test_known_matrix_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.KnownMatrixGate()


def test_known_matrix_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class Missing(gate_features.KnownMatrixGate):
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_known_matrix_gate_is_abstract_can_implement():
    class Included(gate_features.KnownMatrixGate):
        def matrix(self):
            pass

    assert isinstance(Included(), gate_features.KnownMatrixGate)


def test_extrapolatable_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.ExtrapolatableGate()


def test_extrapolatable_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class Missing(gate_features.ExtrapolatableGate):
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_extrapolatable_gate_is_abstract_can_implement():
    class Included(gate_features.ExtrapolatableGate):
        def extrapolate_effect(self, factor):
            pass

    assert isinstance(Included(), gate_features.ExtrapolatableGate)


def test_composite_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.CompositeGate()


def test_composite_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class Missing(gate_features.CompositeGate):
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_composite_gate_is_abstract_can_implement():
    class Included(gate_features.CompositeGate):
        def default_decompose(self, qubits):
            pass

    assert isinstance(Included(), gate_features.CompositeGate)


def test_composite_gate_from_gates():
    class G1(raw_types.Gate):
        pass
    class G2(raw_types.Gate):
        pass

    gates = [G1(), G2()]
    composite = gate_features.CompositeGate.from_gates(gates)

    q1 = raw_types.QubitId()
    assert [gates[0](q1), gates[1](q1)] == composite.default_decompose([q1])


def test_composite_gate_from_gate_tuples():
    class G1(raw_types.Gate):
        pass
    class G2(raw_types.Gate):
        pass

    gates = [(G1(), (0,)), (G2(), (0, 1))]
    composite = gate_features.CompositeGate.from_gates(gates)

    q1 = raw_types.QubitId()
    q2 = raw_types.QubitId()
    assert ([gates[0][0](q1), gates[1][0](q1, q2)]
            == composite.default_decompose([q1, q2]))


def test_self_inverse_is_not_abstract():
    assert isinstance(gate_features.SelfInverseGate(),
                      gate_features.ReversibleGate)


def test_self_inverse_reverse():
    r = gate_features.SelfInverseGate()
    assert r.inverse() is r


def test_single_qubit_gate_validate_args():
    class Dummy(gate_features.SingleQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    q1 = raw_types.QubitId()
    q2 = raw_types.QubitId()

    g.validate_args([q1])
    g.validate_args([q2])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([q1, q2])


def test_two_qubit_gate_is_abstract_can_implement():
    class Included(gate_features.TwoQubitGate):
        def matrix(self):
            pass

    assert isinstance(Included(),
                      gate_features.TwoQubitGate)


def test_two_qubit_gate_validate_pass():
    class Dummy(gate_features.TwoQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    q1 = raw_types.QubitId()
    q2 = raw_types.QubitId()
    q3 = raw_types.QubitId()

    g.validate_args([q1, q2])
    g.validate_args([q2, q3])
    g.validate_args([q3, q2])


def test_two_qubit_gate_validate_wrong_number():
    class Dummy(gate_features.TwoQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    q1 = raw_types.QubitId()
    q2 = raw_types.QubitId()
    q3 = raw_types.QubitId()

    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([q1])
    with pytest.raises(ValueError):
        g.validate_args([q1, q2, q3])


def test_parameterizable_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.ParameterizableGate()


def test_parameterizable_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class Missing(gate_features.ParameterizableGate):
        pass

    with pytest.raises(TypeError):
        _ = Missing()

    # noinspection PyAbstractClass
    class Missing(gate_features.ParameterizableGate):
        def is_parameterized(self):
            pass
        pass

    with pytest.raises(TypeError):
        _ = Missing()

    # noinspection PyAbstractClass
    class Missing(gate_features.ParameterizableGate):
        def resolve_parameters(self):
            pass
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_parameterizable_gate_is_abstract_can_implement():
    class Included(gate_features.ParameterizableGate):
        def is_parameterized(self):
            pass
        def resolve_parameters(self):
            pass

    assert isinstance(Included(), gate_features.ParameterizableGate)
