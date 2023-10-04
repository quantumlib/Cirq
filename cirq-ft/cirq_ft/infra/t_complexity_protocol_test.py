# Copyright 2023 The Cirq Developers
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

import cirq
import cirq_ft
import pytest
from cirq_ft import infra
from cirq_ft.infra.jupyter_tools import execute_notebook


class SupportTComplexity:
    def _t_complexity_(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(t=1)


class DoesNotSupportTComplexity:
    ...


class SupportsTComplexityGateWithRegisters(cirq_ft.GateWithRegisters):
    @property
    def signature(self) -> cirq_ft.Signature:
        return cirq_ft.Signature.build(s=1, t=2)

    def _t_complexity_(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(t=1, clifford=2)


class SupportTComplexityGate(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 1

    def _t_complexity_(self) -> cirq_ft.TComplexity:
        return cirq_ft.TComplexity(t=1)


class DoesNotSupportTComplexityGate(cirq.Gate):
    def _num_qubits_(self):
        return 1


def test_t_complexity():
    with pytest.raises(TypeError):
        _ = cirq_ft.t_complexity(DoesNotSupportTComplexity())

    with pytest.raises(TypeError):
        _ = cirq_ft.t_complexity(DoesNotSupportTComplexityGate())

    assert cirq_ft.t_complexity(DoesNotSupportTComplexity(), fail_quietly=True) is None
    assert cirq_ft.t_complexity([DoesNotSupportTComplexity()], fail_quietly=True) is None
    assert cirq_ft.t_complexity(DoesNotSupportTComplexityGate(), fail_quietly=True) is None

    assert cirq_ft.t_complexity(SupportTComplexity()) == cirq_ft.TComplexity(t=1)
    assert cirq_ft.t_complexity(SupportTComplexityGate().on(cirq.q('t'))) == cirq_ft.TComplexity(
        t=1
    )

    g = cirq_ft.testing.GateHelper(SupportsTComplexityGateWithRegisters())
    assert g.gate._decompose_with_context_(g.operation.qubits) is NotImplemented
    assert cirq_ft.t_complexity(g.gate) == cirq_ft.TComplexity(t=1, clifford=2)
    assert cirq_ft.t_complexity(g.operation) == cirq_ft.TComplexity(t=1, clifford=2)

    assert cirq_ft.t_complexity([cirq.T, cirq.X]) == cirq_ft.TComplexity(t=1, clifford=1)

    q = cirq.NamedQubit('q')
    assert cirq_ft.t_complexity([cirq.T(q), cirq.X(q)]) == cirq_ft.TComplexity(t=1, clifford=1)


def test_gates():
    # T gate and its adjoint
    assert cirq_ft.t_complexity(cirq.T) == cirq_ft.TComplexity(t=1)
    assert cirq_ft.t_complexity(cirq.T**-1) == cirq_ft.TComplexity(t=1)

    assert cirq_ft.t_complexity(cirq.H) == cirq_ft.TComplexity(clifford=1)  # Hadamard
    assert cirq_ft.t_complexity(cirq.CNOT) == cirq_ft.TComplexity(clifford=1)  # CNOT
    assert cirq_ft.t_complexity(cirq.S) == cirq_ft.TComplexity(clifford=1)  # S
    assert cirq_ft.t_complexity(cirq.S**-1) == cirq_ft.TComplexity(clifford=1)  # S†

    # Pauli operators are clifford
    assert cirq_ft.t_complexity(cirq.X) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.Y) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.Z) == cirq_ft.TComplexity(clifford=1)

    # Rotation about X, Y, and Z axes
    assert cirq_ft.t_complexity(cirq.Rx(rads=2)) == cirq_ft.TComplexity(rotations=1)
    assert cirq_ft.t_complexity(cirq.Ry(rads=2)) == cirq_ft.TComplexity(rotations=1)
    assert cirq_ft.t_complexity(cirq.Rz(rads=2)) == cirq_ft.TComplexity(rotations=1)

    # clifford+T
    assert cirq_ft.t_complexity(cirq_ft.And()) == cirq_ft.TComplexity(t=4, clifford=9)
    assert cirq_ft.t_complexity(cirq_ft.And() ** -1) == cirq_ft.TComplexity(clifford=4)

    assert cirq_ft.t_complexity(cirq.FREDKIN) == cirq_ft.TComplexity(t=7, clifford=10)


def test_operations():
    q = cirq.NamedQubit('q')
    assert cirq_ft.t_complexity(cirq.T(q)) == cirq_ft.TComplexity(t=1)

    gate = cirq_ft.And()
    op = gate.on_registers(**infra.get_named_qubits(gate.signature))
    assert cirq_ft.t_complexity(op) == cirq_ft.TComplexity(t=4, clifford=9)

    gate = cirq_ft.And() ** -1
    op = gate.on_registers(**infra.get_named_qubits(gate.signature))
    assert cirq_ft.t_complexity(op) == cirq_ft.TComplexity(clifford=4)


def test_circuits():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(
        cirq.Rz(rads=0.6)(q),
        cirq.T(q),
        cirq.X(q) ** 0.5,
        cirq.Rx(rads=0.1)(q),
        cirq.Ry(rads=0.6)(q),
        cirq.measure(q, key='m'),
    )
    assert cirq_ft.t_complexity(circuit) == cirq_ft.TComplexity(clifford=2, rotations=3, t=1)

    circuit = cirq.FrozenCircuit(cirq.T(q) ** -1, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert cirq_ft.t_complexity(circuit) == cirq_ft.TComplexity(clifford=1, rotations=1, t=1)


def test_circuit_operations():
    q = cirq.NamedQubit('q')
    circuit = cirq.FrozenCircuit(
        cirq.T(q), cirq.X(q) ** 0.5, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m')
    )
    assert cirq_ft.t_complexity(cirq.CircuitOperation(circuit)) == cirq_ft.TComplexity(
        clifford=2, rotations=1, t=1
    )
    assert cirq_ft.t_complexity(
        cirq.CircuitOperation(circuit, repetitions=10)
    ) == cirq_ft.TComplexity(clifford=20, rotations=10, t=10)

    circuit = cirq.FrozenCircuit(cirq.T(q) ** -1, cirq.Rx(rads=0.1)(q), cirq.measure(q, key='m'))
    assert cirq_ft.t_complexity(cirq.CircuitOperation(circuit)) == cirq_ft.TComplexity(
        clifford=1, rotations=1, t=1
    )
    assert cirq_ft.t_complexity(
        cirq.CircuitOperation(circuit, repetitions=3)
    ) == cirq_ft.TComplexity(clifford=3, rotations=3, t=3)


def test_classically_controlled_operations():
    q = cirq.NamedQubit('q')
    assert cirq_ft.t_complexity(cirq.X(q).with_classical_controls('c')) == cirq_ft.TComplexity(
        clifford=1
    )
    assert cirq_ft.t_complexity(
        cirq.Rx(rads=0.1)(q).with_classical_controls('c')
    ) == cirq_ft.TComplexity(rotations=1)
    assert cirq_ft.t_complexity(cirq.T(q).with_classical_controls('c')) == cirq_ft.TComplexity(t=1)


def test_tagged_operations():
    q = cirq.NamedQubit('q')
    assert cirq_ft.t_complexity(cirq.X(q).with_tags('tag1')) == cirq_ft.TComplexity(clifford=1)
    assert cirq_ft.t_complexity(cirq.T(q).with_tags('tage1')) == cirq_ft.TComplexity(t=1)
    assert cirq_ft.t_complexity(
        cirq.Ry(rads=0.1)(q).with_tags('tag1', 'tag2')
    ) == cirq_ft.TComplexity(rotations=1)


def test_cache_clear():
    class IsCachable(cirq.Operation):
        def __init__(self) -> None:
            super().__init__()
            self.num_calls = 0
            self._gate = cirq.X

        def _t_complexity_(self) -> cirq_ft.TComplexity:
            self.num_calls += 1
            return cirq_ft.TComplexity()

        @property
        def qubits(self):
            return [cirq.LineQubit(3)]  # pragma: no cover

        def with_qubits(self, _):
            ...

        @property
        def gate(self):
            return self._gate

    assert cirq_ft.t_complexity(cirq.X) == cirq_ft.TComplexity(clifford=1)
    # Using a global cache will result in a failure of this test since `cirq.X` has
    # `T-complexity(clifford=1)` but we explicitly return `cirq_ft.TComplexity()` for IsCachable
    # operation; for which the hash would be equivalent to the hash of it's subgate i.e. `cirq.X`.
    cirq_ft.t_complexity.cache_clear()
    op = IsCachable()
    assert cirq_ft.t_complexity([op, op]) == cirq_ft.TComplexity()
    assert op.num_calls == 1
    cirq_ft.t_complexity.cache_clear()


def test_notebook():
    execute_notebook('t_complexity')
