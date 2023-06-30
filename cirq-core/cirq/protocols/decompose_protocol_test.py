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
import itertools
from typing import Optional
from unittest import mock
import pytest

import cirq


class NoMethod:
    pass


class DecomposeNotImplemented:
    def _decompose_(self, qubits=None):
        return NotImplemented


class DecomposeNone:
    def _decompose_(self, qubits=None):
        return None


class DecomposeGiven:
    def __init__(self, val):
        self.val = val

    def _decompose_(self):
        return self.val


class DecomposeWithQubitsGiven:
    def __init__(self, func):
        self.func = func

    def _decompose_(self, qubits):
        return self.func(*qubits)


class DecomposeGenerated:
    def _decompose_(self):
        yield cirq.X(cirq.LineQubit(0))
        yield cirq.Y(cirq.LineQubit(1))


class DecomposeQuditGate:
    def _decompose_(self, qids):
        yield cirq.identity_each(*qids)


def test_decompose_once():
    # No default value results in descriptive error.
    with pytest.raises(TypeError, match='no _decompose_with_context_ or _decompose_ method'):
        _ = cirq.decompose_once(NoMethod())
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.decompose_once(DecomposeNotImplemented())
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.decompose_once(DecomposeNone())

    # Default value works.
    assert cirq.decompose_once(NoMethod(), 5) == 5
    assert cirq.decompose_once(DecomposeNotImplemented(), None) is None
    assert cirq.decompose_once(NoMethod(), NotImplemented) is NotImplemented
    assert cirq.decompose_once(DecomposeNone(), 0) == 0

    # Flattens into a list.
    op = cirq.X(cirq.NamedQubit('q'))
    assert cirq.decompose_once(DecomposeGiven(op)) == [op]
    assert cirq.decompose_once(DecomposeGiven([[[op]], []])) == [op]
    assert cirq.decompose_once(DecomposeGiven(op for _ in range(2))) == [op, op]
    assert type(cirq.decompose_once(DecomposeGiven(op for _ in range(2)))) == list
    assert cirq.decompose_once(DecomposeGenerated()) == [
        cirq.X(cirq.LineQubit(0)),
        cirq.Y(cirq.LineQubit(1)),
    ]


def test_decompose_once_with_qubits():
    qs = cirq.LineQubit.range(3)

    # No default value results in descriptive error.
    with pytest.raises(TypeError, match='no _decompose_with_context_ or _decompose_ method'):
        _ = cirq.decompose_once_with_qubits(NoMethod(), qs)
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.decompose_once_with_qubits(DecomposeNotImplemented(), qs)
    with pytest.raises(TypeError, match='returned NotImplemented or None'):
        _ = cirq.decompose_once_with_qubits(DecomposeNone(), qs)

    # Default value works.
    assert cirq.decompose_once_with_qubits(NoMethod(), qs, 5) == 5
    assert cirq.decompose_once_with_qubits(DecomposeNotImplemented(), qs, None) is None
    assert cirq.decompose_once_with_qubits(NoMethod(), qs, NotImplemented) is NotImplemented

    # Flattens into a list.
    assert cirq.decompose_once_with_qubits(DecomposeWithQubitsGiven(cirq.X.on_each), qs) == [
        cirq.X(cirq.LineQubit(0)),
        cirq.X(cirq.LineQubit(1)),
        cirq.X(cirq.LineQubit(2)),
    ]
    assert cirq.decompose_once_with_qubits(
        DecomposeWithQubitsGiven(lambda *qubits: cirq.Y(qubits[0])), qs
    ) == [cirq.Y(cirq.LineQubit(0))]
    assert cirq.decompose_once_with_qubits(
        DecomposeWithQubitsGiven(lambda *qubits: (cirq.Y(q) for q in qubits)), qs
    ) == [cirq.Y(cirq.LineQubit(0)), cirq.Y(cirq.LineQubit(1)), cirq.Y(cirq.LineQubit(2))]

    # Qudits, _decompose_ argument name is not 'qubits'.
    assert cirq.decompose_once_with_qubits(
        DecomposeQuditGate(), cirq.LineQid.for_qid_shape((1, 2, 3))
    ) == [cirq.identity_each(*cirq.LineQid.for_qid_shape((1, 2, 3)))]

    # Works when qubits are generated.
    def use_qubits_twice(*qubits):
        a = list(qubits)
        b = list(qubits)
        yield cirq.X.on_each(*a)
        yield cirq.Y.on_each(*b)

    assert cirq.decompose_once_with_qubits(
        DecomposeWithQubitsGiven(use_qubits_twice), (q for q in qs)
    ) == list(cirq.X.on_each(*qs)) + list(cirq.Y.on_each(*qs))


def test_decompose_general():
    a, b, c = cirq.LineQubit.range(3)
    no_method = NoMethod()
    assert cirq.decompose(no_method) == [no_method]

    # Flattens iterables.
    assert cirq.decompose([cirq.SWAP(a, b), cirq.SWAP(a, b)]) == 2 * cirq.decompose(cirq.SWAP(a, b))

    # Decomposed circuit should be equivalent. The ordering should be correct.
    ops = cirq.TOFFOLI(a, b, c), cirq.H(a), cirq.CZ(a, c)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        cirq.Circuit(ops), cirq.Circuit(cirq.decompose(ops)), atol=1e-8
    )


def test_decompose_keep():
    a, b = cirq.LineQubit.range(2)

    # Recursion can be stopped.
    assert cirq.decompose(cirq.SWAP(a, b), keep=lambda e: isinstance(e.gate, cirq.CNotPowGate)) == [
        cirq.CNOT(a, b),
        cirq.CNOT(b, a),
        cirq.CNOT(a, b),
    ]

    # Recursion continues down to CZ + single-qubit gates.
    cirq.testing.assert_has_diagram(
        cirq.Circuit(cirq.decompose(cirq.SWAP(a, b))),
        """
0: ────────────@───Y^-0.5───@───Y^0.5────@───────────
               │            │            │
1: ───Y^-0.5───@───Y^0.5────@───Y^-0.5───@───Y^0.5───
""",
    )

    # If you're happy with everything, no decomposition happens.
    assert cirq.decompose(cirq.SWAP(a, b), keep=lambda _: True) == [cirq.SWAP(a, b)]
    # Unless it's not an operation.
    assert cirq.decompose(DecomposeGiven(cirq.SWAP(b, a)), keep=lambda _: True) == [cirq.SWAP(b, a)]
    # E.g. lists still get flattened.
    assert cirq.decompose([[[cirq.SWAP(a, b)]]], keep=lambda _: True) == [cirq.SWAP(a, b)]


def test_decompose_on_stuck_raise():
    a, b = cirq.LineQubit.range(2)
    no_method = NoMethod()

    # If you're not happy with anything, you're going to get an error.
    with pytest.raises(ValueError, match="but can't be decomposed"):
        _ = cirq.decompose(NoMethod(), keep=lambda _: False)
    # Unless there's no operations to be unhappy about.
    assert cirq.decompose([], keep=lambda _: False) == []
    assert cirq.decompose([], on_stuck_raise=None) == []
    # Or you say you're fine.
    assert cirq.decompose(no_method, keep=lambda _: False, on_stuck_raise=None) == [no_method]
    assert cirq.decompose(no_method, keep=lambda _: False, on_stuck_raise=lambda _: None) == [
        no_method
    ]
    # You can customize the error.
    with pytest.raises(TypeError, match='test'):
        _ = cirq.decompose(no_method, keep=lambda _: False, on_stuck_raise=TypeError('test'))
    with pytest.raises(NotImplementedError, match='op cirq.CZ'):
        _ = cirq.decompose(
            cirq.CZ(a, b),
            keep=lambda _: False,
            on_stuck_raise=lambda op: NotImplementedError(f'op {op!r}'),
        )

    # There's a nice warning if you specify `on_stuck_raise` but not `keep`.
    with pytest.raises(ValueError, match='on_stuck_raise'):
        assert cirq.decompose([], on_stuck_raise=TypeError('x'))


def test_decompose_intercept():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Runs instead of normal decomposition.
    actual = cirq.decompose(
        cirq.SWAP(a, b),
        intercepting_decomposer=lambda op: (cirq.X(a) if op == cirq.SWAP(a, b) else NotImplemented),
    )
    assert actual == [cirq.X(a)]

    # Falls back to normal decomposition when NotImplemented.
    actual = cirq.decompose(
        cirq.SWAP(a, b),
        keep=lambda op: isinstance(op.gate, cirq.CNotPowGate),
        intercepting_decomposer=lambda _: NotImplemented,
    )
    assert actual == [cirq.CNOT(a, b), cirq.CNOT(b, a), cirq.CNOT(a, b)]

    # Accepts a context, when provided.
    def _intercept_with_context(
        op: cirq.Operation, context: Optional[cirq.DecompositionContext] = None
    ):
        assert context is not None
        if op.gate == cirq.SWAP:
            q = context.qubit_manager.qalloc(1)
            a, b = op.qubits
            return [cirq.X(a), cirq.X(*q), cirq.X(b)]
        return NotImplemented

    context = cirq.DecompositionContext(cirq.ops.SimpleQubitManager())
    actual = cirq.decompose(
        cirq.SWAP(a, b), intercepting_decomposer=_intercept_with_context, context=context
    )
    assert actual == [cirq.X(a), cirq.X(cirq.ops.CleanQubit(0)), cirq.X(b)]


def test_decompose_preserving_structure():
    a, b = cirq.LineQubit.range(2)
    fc1 = cirq.FrozenCircuit(cirq.SWAP(a, b), cirq.FSimGate(0.1, 0.2).on(a, b))
    cop1_1 = cirq.CircuitOperation(fc1).with_tags('test_tag')
    cop1_2 = cirq.CircuitOperation(fc1).with_qubit_mapping({a: b, b: a})
    fc2 = cirq.FrozenCircuit(cirq.X(a), cop1_1, cop1_2)
    cop2 = cirq.CircuitOperation(fc2)

    circuit = cirq.Circuit(cop2, cirq.measure(a, b, key='m'))
    actual = cirq.Circuit(cirq.decompose(circuit, preserve_structure=True))

    # This should keep the CircuitOperations but decompose their SWAPs.
    fc1_decomp = cirq.FrozenCircuit(cirq.decompose(fc1))
    expected = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.X(a),
                cirq.CircuitOperation(fc1_decomp).with_tags('test_tag'),
                cirq.CircuitOperation(fc1_decomp).with_qubit_mapping({a: b, b: a}),
            )
        ),
        cirq.measure(a, b, key='m'),
    )
    assert actual == expected


# Test both intercepting and fallback decomposers.
@pytest.mark.parametrize('decompose_mode', ['intercept', 'fallback'])
def test_decompose_preserving_structure_forwards_args(decompose_mode):
    a, b = cirq.LineQubit.range(2)
    fc1 = cirq.FrozenCircuit(cirq.SWAP(a, b), cirq.FSimGate(0.1, 0.2).on(a, b))
    cop1_1 = cirq.CircuitOperation(fc1).with_tags('test_tag')
    cop1_2 = cirq.CircuitOperation(fc1).with_qubit_mapping({a: b, b: a})
    fc2 = cirq.FrozenCircuit(cirq.X(a), cop1_1, cop1_2)
    cop2 = cirq.CircuitOperation(fc2)

    circuit = cirq.Circuit(cop2, cirq.measure(a, b, key='m'))

    def keep_func(op: 'cirq.Operation'):
        # Only decompose SWAP and X.
        return not isinstance(op.gate, (cirq.SwapPowGate, cirq.XPowGate))

    def x_to_hzh(op: 'cirq.Operation'):
        if isinstance(op.gate, cirq.XPowGate) and op.gate.exponent == 1:
            return [cirq.H(*op.qubits), cirq.Z(*op.qubits), cirq.H(*op.qubits)]

    actual = cirq.Circuit(
        cirq.decompose(
            circuit,
            keep=keep_func,
            intercepting_decomposer=x_to_hzh if decompose_mode == 'intercept' else None,
            fallback_decomposer=x_to_hzh if decompose_mode == 'fallback' else None,
            preserve_structure=True,
        )
    )

    # This should keep the CircuitOperations but decompose their SWAPs.
    fc1_decomp = cirq.FrozenCircuit(
        cirq.decompose(fc1, keep=keep_func, fallback_decomposer=x_to_hzh)
    )
    expected = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.H(a),
                cirq.Z(a),
                cirq.H(a),
                cirq.CircuitOperation(fc1_decomp).with_tags('test_tag'),
                cirq.CircuitOperation(fc1_decomp).with_qubit_mapping({a: b, b: a}),
            )
        ),
        cirq.measure(a, b, key='m'),
    )
    assert actual == expected


def test_decompose_tagged_operation():
    op = cirq.TaggedOperation(
        cirq.CircuitOperation(
            circuit=cirq.FrozenCircuit(
                [cirq.Moment(cirq.SWAP(cirq.LineQubit(0), cirq.LineQubit(1)))]
            )
        ),
        'tag',
    )
    assert cirq.decompose_once(op) == cirq.decompose_once(op.untagged)


class RecursiveDecompose(cirq.Gate):
    def __init__(
        self,
        recurse: bool = True,
        mock_qm=mock.Mock(spec=cirq.QubitManager),
        with_context: bool = False,
    ):
        self.recurse = recurse
        self.mock_qm = mock_qm
        self.with_context = with_context

    def _num_qubits_(self) -> int:
        return 2

    def _decompose_impl(self, qubits, mock_qm: mock.Mock):
        mock_qm.qalloc(self.recurse)
        yield RecursiveDecompose(
            recurse=False, mock_qm=self.mock_qm, with_context=self.with_context
        ).on(*qubits) if self.recurse else cirq.Z.on_each(*qubits)
        mock_qm.qfree(self.recurse)

    def _decompose_(self, qubits):
        if self.with_context:
            assert False
        else:
            return self._decompose_impl(qubits, self.mock_qm)

    def _decompose_with_context_(self, qubits, context):
        if self.with_context:
            qm = self.mock_qm if context is None else context.qubit_manager
            return self._decompose_impl(qubits, qm)
        else:
            return NotImplemented

    def _has_unitary_(self):
        return True


@pytest.mark.parametrize('with_context', [True, False])
def test_decompose_recursive_dfs(with_context: bool):
    expected_calls = [
        mock.call.qalloc(True),
        mock.call.qalloc(False),
        mock.call.qfree(False),
        mock.call.qfree(True),
    ]
    mock_qm = mock.Mock(spec=cirq.QubitManager)
    context_qm = mock.Mock(spec=cirq.QubitManager)
    gate = RecursiveDecompose(mock_qm=mock_qm, with_context=with_context)
    q = cirq.LineQubit.range(3)
    gate_op = gate.on(*q[:2])
    tagged_op = gate_op.with_tags("custom tag")
    controlled_op = gate_op.controlled_by(q[2])
    classically_controlled_op = gate_op.with_classical_controls('key')
    moment = cirq.Moment(gate_op)
    circuit = cirq.Circuit(moment)
    for val in [gate_op, tagged_op, controlled_op, classically_controlled_op, moment, circuit]:
        mock_qm.reset_mock()
        _ = cirq.decompose(val, context=cirq.DecompositionContext(qubit_manager=mock_qm))
        assert mock_qm.method_calls == expected_calls

        mock_qm.reset_mock()
        context_qm.reset_mock()
        _ = cirq.decompose(val, context=cirq.DecompositionContext(context_qm))
        assert (
            context_qm.method_calls == expected_calls
            if with_context
            else mock_qm.method_calls == expected_calls
        )


class G1(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 1

    def _decompose_with_context_(self, qubits, context):
        yield cirq.CNOT(qubits[0], context.qubit_manager.qalloc(1)[0])


class G2(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 1

    def _decompose_with_context_(self, qubits, context):
        yield G1()(*context.qubit_manager.qalloc(1))


@mock.patch('cirq.protocols.decompose_protocol._CONTEXT_COUNTER', itertools.count())
def test_successive_decompose_once_succeed():
    op = G2()(cirq.NamedQubit('q'))
    d1 = cirq.decompose_once(op)
    d2 = cirq.decompose_once(d1[0])
    assert d2 == [
        cirq.CNOT(
            cirq.ops.CleanQubit(0, prefix='_decompose_protocol_0'),
            cirq.ops.CleanQubit(0, prefix='_decompose_protocol_1'),
        )
    ]


def test_decompose_without_context_succeed():
    op = G2()(cirq.NamedQubit('q'))
    assert cirq.decompose(op, keep=lambda op: op.gate is cirq.CNOT) == [
        cirq.CNOT(
            cirq.ops.CleanQubit(0, prefix='_decompose_protocol'),
            cirq.ops.CleanQubit(1, prefix='_decompose_protocol'),
        )
    ]
