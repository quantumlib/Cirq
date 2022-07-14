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
import collections.abc
import pathlib

import numpy as np
import pytest
import sympy

import cirq
import cirq.testing


def test_gate_operation_init():
    q = cirq.NamedQubit('q')
    g = cirq.testing.SingleQubitGate()
    v = cirq.GateOperation(g, (q,))
    assert v.gate == g
    assert v.qubits == (q,)


def test_invalid_gate_operation():
    three_qubit_gate = cirq.testing.ThreeQubitGate()
    single_qubit = [cirq.GridQubit(0, 0)]
    with pytest.raises(ValueError, match="number of qubits"):
        cirq.GateOperation(three_qubit_gate, single_qubit)


def test_immutable():
    a, b = cirq.LineQubit.range(2)
    op = cirq.X(a)

    with pytest.raises(AttributeError, match="can't set attribute"):
        op.gate = cirq.Y

    with pytest.raises(AttributeError, match="can't set attribute"):
        op.qubits = [b]


def test_gate_operation_eq():
    g1 = cirq.testing.SingleQubitGate()
    g2 = cirq.testing.SingleQubitGate()
    g3 = cirq.testing.TwoQubitGate()
    r1 = [cirq.NamedQubit('r1')]
    r2 = [cirq.NamedQubit('r2')]
    r12 = r1 + r2
    r21 = r2 + r1

    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r1))
    eq.make_equality_group(lambda: cirq.GateOperation(g2, r1))
    eq.make_equality_group(lambda: cirq.GateOperation(g1, r2))
    eq.make_equality_group(lambda: cirq.GateOperation(g3, r12))
    eq.make_equality_group(lambda: cirq.GateOperation(g3, r21))
    eq.add_equality_group(cirq.GateOperation(cirq.CZ, r21), cirq.GateOperation(cirq.CZ, r12))

    @cirq.value_equality
    class PairGate(cirq.Gate, cirq.InterchangeableQubitsGate):
        """Interchangeable subsets."""

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def qubit_index_to_equivalence_group_key(self, index: int):
            return index // 2

        def _value_equality_values_(self):
            return (self.num_qubits(),)

    def p(*q):
        return PairGate(len(q)).on(*q)

    a0, a1, b0, b1, c0 = cirq.LineQubit.range(5)
    eq.add_equality_group(p(a0, a1, b0, b1), p(a1, a0, b1, b0))
    eq.add_equality_group(p(b0, b1, a0, a1))
    eq.add_equality_group(p(a0, a1, b0, b1, c0), p(a1, a0, b1, b0, c0))
    eq.add_equality_group(p(a0, b0, a1, b1, c0))
    eq.add_equality_group(p(a0, c0, b0, b1, a1))
    eq.add_equality_group(p(b0, a1, a0, b1, c0))


def test_gate_operation_approx_eq():
    a = [cirq.NamedQubit('r1')]
    b = [cirq.NamedQubit('r2')]

    assert cirq.approx_eq(
        cirq.GateOperation(cirq.XPowGate(), a), cirq.GateOperation(cirq.XPowGate(), a)
    )
    assert not cirq.approx_eq(
        cirq.GateOperation(cirq.XPowGate(), a), cirq.GateOperation(cirq.XPowGate(), b)
    )

    assert cirq.approx_eq(
        cirq.GateOperation(cirq.XPowGate(exponent=0), a),
        cirq.GateOperation(cirq.XPowGate(exponent=1e-9), a),
    )
    assert not cirq.approx_eq(
        cirq.GateOperation(cirq.XPowGate(exponent=0), a),
        cirq.GateOperation(cirq.XPowGate(exponent=1e-7), a),
    )
    assert cirq.approx_eq(
        cirq.GateOperation(cirq.XPowGate(exponent=0), a),
        cirq.GateOperation(cirq.XPowGate(exponent=1e-7), a),
        atol=1e-6,
    )


def test_gate_operation_qid_shape():
    class ShapeGate(cirq.Gate):
        def _qid_shape_(self):
            return (1, 2, 3, 4)

    op = ShapeGate().on(*cirq.LineQid.for_qid_shape((1, 2, 3, 4)))
    assert cirq.qid_shape(op) == (1, 2, 3, 4)
    assert cirq.num_qubits(op) == 4


def test_gate_operation_num_qubits():
    class NumQubitsGate(cirq.Gate):
        def _num_qubits_(self):
            return 4

    op = NumQubitsGate().on(*cirq.LineQubit.range(4))
    assert cirq.qid_shape(op) == (2, 2, 2, 2)
    assert cirq.num_qubits(op) == 4


def test_gate_operation_pow():
    Y = cirq.Y
    q = cirq.NamedQubit('q')
    assert (Y**0.5)(q) == Y(q) ** 0.5


def test_with_qubits_and_transform_qubits():
    g = cirq.testing.ThreeQubitGate()
    g = cirq.testing.ThreeQubitGate()
    op = cirq.GateOperation(g, cirq.LineQubit.range(3))
    assert op.with_qubits(*cirq.LineQubit.range(3, 0, -1)) == cirq.GateOperation(
        g, cirq.LineQubit.range(3, 0, -1)
    )
    assert op.transform_qubits(lambda e: cirq.LineQubit(-e.x)) == cirq.GateOperation(
        g, [cirq.LineQubit(0), cirq.LineQubit(-1), cirq.LineQubit(-2)]
    )


def test_extrapolate():
    q = cirq.NamedQubit('q')

    # If the gate isn't extrapolatable, you get a type error.
    op0 = cirq.GateOperation(cirq.testing.SingleQubitGate(), [q])
    with pytest.raises(TypeError):
        _ = op0**0.5

    op1 = cirq.GateOperation(cirq.Y, [q])
    assert op1**0.5 == cirq.GateOperation(cirq.Y**0.5, [q])
    assert (cirq.Y**0.5).on(q) == cirq.Y(q) ** 0.5


def test_inverse():
    q = cirq.NamedQubit('q')

    # If the gate isn't reversible, you get a type error.
    op0 = cirq.GateOperation(cirq.testing.SingleQubitGate(), [q])
    assert cirq.inverse(op0, None) is None

    op1 = cirq.GateOperation(cirq.S, [q])
    assert cirq.inverse(op1) == op1**-1 == cirq.GateOperation(cirq.S**-1, [q])
    assert cirq.inverse(cirq.S).on(q) == cirq.inverse(cirq.S.on(q))


def test_text_diagrammable():
    q = cirq.NamedQubit('q')

    # If the gate isn't diagrammable, you get a type error.
    op0 = cirq.GateOperation(cirq.testing.SingleQubitGate(), [q])
    with pytest.raises(TypeError):
        _ = cirq.circuit_diagram_info(op0)

    op1 = cirq.GateOperation(cirq.S, [q])
    actual = cirq.circuit_diagram_info(op1)
    expected = cirq.circuit_diagram_info(cirq.S)
    assert actual == expected


def test_bounded_effect():
    q = cirq.NamedQubit('q')

    # If the gate isn't bounded, you get a type error.
    op0 = cirq.GateOperation(cirq.testing.SingleQubitGate(), [q])
    assert cirq.trace_distance_bound(op0) >= 1
    op1 = cirq.GateOperation(cirq.Z**0.000001, [q])
    op1_bound = cirq.trace_distance_bound(op1)
    assert op1_bound == cirq.trace_distance_bound(cirq.Z**0.000001)


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable_effect(resolve_fn):
    q = cirq.NamedQubit('q')
    r = cirq.ParamResolver({'a': 0.5})

    op1 = cirq.GateOperation(cirq.Z ** sympy.Symbol('a'), [q])
    assert cirq.is_parameterized(op1)
    op2 = resolve_fn(op1, r)
    assert not cirq.is_parameterized(op2)
    assert op2 == cirq.S.on(q)


def test_pauli_expansion():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.pauli_expansion(cirq.X(a)) == cirq.LinearDict({'X': 1})
    assert cirq.pauli_expansion(cirq.CNOT(a, b)) == cirq.pauli_expansion(cirq.CNOT)

    class No(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

    class Yes(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _pauli_expansion_(self):
            return cirq.LinearDict({'X': 0.5})

    assert cirq.pauli_expansion(No().on(a), default=None) is None
    assert cirq.pauli_expansion(Yes().on(a)) == cirq.LinearDict({'X': 0.5})


def test_unitary():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert not cirq.has_unitary(cirq.measure(a))
    assert cirq.unitary(cirq.measure(a), None) is None
    np.testing.assert_allclose(cirq.unitary(cirq.X(a)), np.array([[0, 1], [1, 0]]), atol=1e-8)
    np.testing.assert_allclose(cirq.unitary(cirq.CNOT(a, b)), cirq.unitary(cirq.CNOT), atol=1e-8)


def test_channel():
    a = cirq.NamedQubit('a')
    op = cirq.bit_flip(0.5).on(a)
    np.testing.assert_allclose(cirq.kraus(op), cirq.kraus(op.gate))
    assert cirq.has_kraus(op)

    assert cirq.kraus(cirq.testing.SingleQubitGate()(a), None) is None
    assert not cirq.has_kraus(cirq.testing.SingleQubitGate()(a))


def test_measurement_key():
    a = cirq.NamedQubit('a')
    assert cirq.measurement_key_name(cirq.measure(a, key='lock')) == 'lock'


def assert_mixtures_equal(actual, expected):
    """Assert equal for tuple of mixed scalar and array types."""
    for a, e in zip(actual, expected):
        np.testing.assert_almost_equal(a[0], e[0])
        np.testing.assert_almost_equal(a[1], e[1])


def test_mixture():
    a = cirq.NamedQubit('a')
    op = cirq.bit_flip(0.5).on(a)
    assert_mixtures_equal(cirq.mixture(op), cirq.mixture(op.gate))
    assert cirq.has_mixture(op)

    assert cirq.has_mixture(cirq.X(a))
    m = cirq.mixture(cirq.X(a))
    assert len(m) == 1
    assert m[0][0] == 1
    np.testing.assert_allclose(m[0][1], cirq.unitary(cirq.X))


def test_repr():
    a, b = cirq.LineQubit.range(2)
    assert (
        repr(cirq.GateOperation(cirq.CZ, (a, b))) == 'cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1))'
    )

    class Inconsistent(cirq.testing.SingleQubitGate):
        def __repr__(self):
            return 'Inconsistent'

        def on(self, *qubits):
            return cirq.GateOperation(Inconsistent(), qubits)

    assert (
        repr(cirq.GateOperation(Inconsistent(), [a]))
        == 'cirq.GateOperation(gate=Inconsistent, qubits=[cirq.LineQubit(0)])'
    )


@pytest.mark.parametrize(
    'gate1,gate2,eq_up_to_global_phase',
    [
        (cirq.rz(0.3 * np.pi), cirq.Z**0.3, True),
        (cirq.rz(0.3), cirq.Z**0.3, False),
        (cirq.ZZPowGate(global_shift=0.5), cirq.ZZ, True),
        (cirq.ZPowGate(global_shift=0.5) ** sympy.Symbol('e'), cirq.Z, False),
        (cirq.Z ** sympy.Symbol('e'), cirq.Z ** sympy.Symbol('f'), False),
    ],
)
def test_equal_up_to_global_phase_on_gates(gate1, gate2, eq_up_to_global_phase):
    num_qubits1, num_qubits2 = (cirq.num_qubits(g) for g in (gate1, gate2))
    qubits = cirq.LineQubit.range(max(num_qubits1, num_qubits2) + 1)
    op1, op2 = gate1(*qubits[:num_qubits1]), gate2(*qubits[:num_qubits2])
    assert cirq.equal_up_to_global_phase(op1, op2) == eq_up_to_global_phase
    op2_on_diff_qubits = gate2(*qubits[1 : num_qubits2 + 1])
    assert not cirq.equal_up_to_global_phase(op1, op2_on_diff_qubits)


def test_equal_up_to_global_phase_on_diff_types():
    op = cirq.X(cirq.LineQubit(0))
    assert not cirq.equal_up_to_global_phase(op, 3)


def test_gate_on_operation_besides_gate_operation():
    a, b = cirq.LineQubit.range(2)

    op = -1j * cirq.X(a) * cirq.Y(b)
    assert isinstance(op.gate, cirq.DensePauliString)
    assert op.gate == -1j * cirq.DensePauliString('XY')
    assert not isinstance(op.gate, cirq.XPowGate)


def test_mul():
    class GateRMul(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _rmul_with_qubits(self, qubits, other):
            if other == 2:
                return 3
            if isinstance(other, cirq.Operation) and isinstance(other.gate, GateRMul):
                return 4
            raise NotImplementedError()

    class GateMul(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _mul_with_qubits(self, qubits, other):
            if other == 2:
                return 5
            if isinstance(other, cirq.Operation) and isinstance(other.gate, GateMul):
                return 6
            raise NotImplementedError()

    # Delegates right multiplication.
    q = cirq.LineQubit(0)
    r = GateRMul().on(q)
    assert 2 * r == 3
    with pytest.raises(TypeError):
        _ = r * 2

    # Delegates left multiplication.
    m = GateMul().on(q)
    assert m * 2 == 5
    with pytest.raises(TypeError):
        _ = 2 * m

    # Handles the symmetric type case correctly.
    assert m * m == 6
    assert r * r == 4


def test_with_gate():
    g1 = cirq.GateOperation(cirq.X, cirq.LineQubit.range(1))
    g2 = cirq.GateOperation(cirq.Y, cirq.LineQubit.range(1))
    assert g1.with_gate(cirq.X) is g1
    assert g1.with_gate(cirq.Y) == g2


def test_with_measurement_key_mapping():
    a = cirq.LineQubit(0)
    op = cirq.measure(a, key='m')

    remap_op = cirq.with_measurement_key_mapping(op, {'m': 'k'})
    assert cirq.measurement_key_names(remap_op) == {'k'}
    assert cirq.with_measurement_key_mapping(op, {'x': 'k'}) is op


def test_with_key_path():
    a = cirq.LineQubit(0)
    op = cirq.measure(a, key='m')

    remap_op = cirq.with_key_path(op, ('a', 'b'))
    assert cirq.measurement_key_names(remap_op) == {'a:b:m'}
    assert cirq.with_key_path(remap_op, ('a', 'b')) is remap_op

    assert cirq.with_key_path(op, tuple()) is op

    assert cirq.with_key_path(cirq.X(a), ('a', 'b')) is NotImplemented


def test_with_key_path_prefix():
    a = cirq.LineQubit(0)
    op = cirq.measure(a, key='m')
    remap_op = cirq.with_key_path_prefix(op, ('a', 'b'))
    assert cirq.measurement_key_names(remap_op) == {'a:b:m'}
    assert cirq.with_key_path_prefix(remap_op, tuple()) is remap_op
    assert cirq.with_key_path_prefix(op, tuple()) is op
    assert cirq.with_key_path_prefix(cirq.X(a), ('a', 'b')) is NotImplemented


def test_cannot_remap_non_measurement_gate():
    a = cirq.LineQubit(0)
    op = cirq.X(a)

    assert cirq.with_measurement_key_mapping(op, {'m': 'k'}) is NotImplemented


def test_is_parameterized():
    class No1(cirq.testing.SingleQubitGate):
        def num_qubits(self) -> int:
            return 1

    class No2(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _is_parameterized_(self):
            return False

    class Yes(cirq.Gate):
        def num_qubits(self) -> int:
            return 1

        def _is_parameterized_(self):
            return True

    q = cirq.LineQubit(0)
    assert not cirq.is_parameterized(No1().on(q))
    assert not cirq.is_parameterized(No2().on(q))
    assert cirq.is_parameterized(Yes().on(q))


def test_group_interchangeable_qubits_creates_tuples_with_unique_keys():
    class MyGate(cirq.Gate, cirq.InterchangeableQubitsGate):
        def __init__(self, num_qubits) -> None:
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def qubit_index_to_equivalence_group_key(self, index: int) -> int:
            if index % 2 == 0:
                return index
            return 0

    qubits = cirq.LineQubit.range(4)
    gate = MyGate(len(qubits))

    assert gate(qubits[0], qubits[1], qubits[2], qubits[3]) == gate(
        qubits[3], qubits[1], qubits[2], qubits[0]
    )


def test_gate_to_operation_to_gate_round_trips():
    def all_subclasses(cls):
        return set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in all_subclasses(c)]
        )

    # Only test gate subclasses in cirq-core.
    gate_subclasses = {
        g
        for g in all_subclasses(cirq.Gate)
        if "cirq." in g.__module__ and "contrib" not in g.__module__ and "test" not in g.__module__
    }

    test_module_spec = cirq.testing.json.spec_for("cirq.protocols")

    skip_classes = {
        # Abstract or private parent classes.
        cirq.ArithmeticGate,
        cirq.BaseDensePauliString,
        cirq.EigenGate,
        cirq.Pauli,
        # Private gates.
        cirq.transformers.analytical_decompositions.two_qubit_to_fsim._BGate,
        cirq.ops.raw_types._InverseCompositeGate,
        cirq.circuits.qasm_output.QasmTwoQubitGate,
        cirq.ops.MSGate,
        # Interop gates
        cirq.interop.quirk.QuirkQubitPermutationGate,
        cirq.interop.quirk.QuirkArithmeticGate,
    }

    skipped = set()
    for gate_cls in gate_subclasses:
        filename = test_module_spec.test_data_path.joinpath(f"{gate_cls.__name__}.json")
        if pathlib.Path(filename).is_file():
            gates = cirq.read_json(filename)
        else:
            if gate_cls in skip_classes:
                skipped.add(gate_cls)
                continue
            # coverage:ignore
            raise AssertionError(
                f"{gate_cls} has no json file, please add a json file or add to the list of "
                "classes to be skipped if there is a reason this gate should not round trip "
                "to a gate via creating an operation."
            )

        if not isinstance(gates, collections.abc.Iterable):
            gates = [gates]
        for gate in gates:
            if gate.num_qubits():
                qudits = [cirq.LineQid(i, d) for i, d in enumerate(cirq.qid_shape(gate))]
                assert gate.on(*qudits).gate == gate

    assert (
        skipped == skip_classes
    ), "A gate that was supposed to be skipped was not, please update the list of skipped gates."
