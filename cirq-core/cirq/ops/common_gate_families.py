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

"""Common Gate Families used in cirq-core"""

from typing import Any, cast, Optional, Type, Union

from cirq.ops import gateset, raw_types, parallel_gate, eigen_gate
from cirq import protocols


class AnyUnitaryGateFamily(gateset.GateFamily):
    """GateFamily which accepts any N-Qubit unitary gate."""

    def __init__(self, num_qubits: Optional[int] = None) -> None:
        """Init AnyUnitaryGateFamily

        Args:
            num_qubits: The GateFamily will accept any unitary gate acting on `num_qubits`.
                        If left `None`, the GateFamily will accept every unitary gate.
        Raises:
            ValueError: If `num_qubits` <= 0.
        """
        if num_qubits is not None and num_qubits <= 0:
            raise ValueError(f'num_qubits: {num_qubits} must be a positive integer.')

        self._num_qubits = num_qubits
        name = f'{str(num_qubits) if num_qubits else "Any"}-Qubit UnitaryGateFamily'
        description = 'Accepts any {}unitary gate'.format(
            f'{num_qubits}-qubit ' if num_qubits else ''
        )
        super().__init__(raw_types.Gate, name=name, description=description)

    def _predicate(self, g: raw_types.Gate) -> bool:
        return (
            self._num_qubits is None or protocols.num_qubits(g) == self._num_qubits
        ) and protocols.has_unitary(g)

    def __repr__(self) -> str:
        return f'cirq.AnyUnitaryGateFamily(num_qubits = {self._num_qubits})'

    def _value_equality_values_(self) -> Any:
        return self._num_qubits


class AnyIntegerPowerGateFamily(gateset.GateFamily):
    """GateFamily which accepts instances of a given `cirq.EigenGate`, raised to integer power."""

    def __init__(self, gate: Type[eigen_gate.EigenGate]) -> None:
        """Init AnyIntegerPowerGateFamily

        Args:
            gate: A subclass of `cirq.EigenGate` s.t. an instance `g` of `gate` will be
                accepted if `g.exponent` is an integer.

        Raises:
            ValueError: If `gate` is not a subclass of `cirq.EigenGate`.
        """
        if not (isinstance(gate, type) and issubclass(gate, eigen_gate.EigenGate)):
            raise ValueError(f'{gate} must be a subclass of `cirq.EigenGate`.')
        super().__init__(
            gate,
            name=f'AnyIntegerPowerGateFamily: {gate}',
            description=f'Accepts any instance `g` of `{gate}` s.t. `g.exponent` is an integer.',
        )

    def _predicate(self, g: raw_types.Gate) -> bool:
        if protocols.is_parameterized(g) or not super()._predicate(g):
            return False
        exp = cast(eigen_gate.EigenGate, g).exponent  # for mypy
        return int(exp) == exp

    def __repr__(self) -> str:
        return f'cirq.AnyIntegerPowerGateFamily({self._gate_str()})'

    def _value_equality_values_(self) -> Any:
        return self.gate


class ParallelGateFamily(gateset.GateFamily):
    """GateFamily which accepts instances of `cirq.ParallelGate` and it's sub_gate.

    ParallelGateFamily is useful for description and validation of scenarios where multiple
    copies of a unitary gate can act in parallel. `cirq.ParallelGate` is used to express
    such a gate with a corresponding unitary `sub_gate` that acts in parallel.

    ParallelGateFamily supports initialization via
        a) Gate Instance that can be applied in parallel.
        b) Gate Type whose instances can be applied in parallel.

    In both the cases, the users can specify an additional parameter `max_parallel_allowed` which
    is used to verify the maximum number of qubits on which any given gate instance can act on.

    To verify containment of a given `cirq.Gate` instance `g`, the gate family verfies that
        a) `cirq.num_qubits(g)` <= `max_parallel_allowed` if `max_parallel_allowed` is not None.
        b) `g` or `g.sub_gate` (if `g` is an instance of `cirq.ParallelGate`) is an accepted gate
            based on type or instance checks depending on the initialization gate type.
    """

    def __init__(
        self,
        gate: Union[Type[raw_types.Gate], raw_types.Gate],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_parallel_allowed: Optional[int] = None,
    ) -> None:
        """Inits ParallelGateFamily

        Args:
            gate: The gate which can act in parallel. It can be a python `type` inheriting from
                `cirq.Gate` or a non-parameterized instance of a `cirq.Gate`. If an instance of
                `cirq.ParallelGate` is passed, then the corresponding `gate.sub_gate` is used.
            name: The name of the gate family.
            description: Human readable description of the gate family.
            max_parallel_allowed: The maximum number of qubits on which a given gate `g`
            can act on. If None, then any number of qubits are allowed.
        """
        if isinstance(gate, parallel_gate.ParallelGate):
            if not max_parallel_allowed:
                max_parallel_allowed = protocols.num_qubits(gate)
            gate = cast(parallel_gate.ParallelGate, gate).sub_gate
        self._max_parallel_allowed = max_parallel_allowed
        super().__init__(gate, name=name, description=description)

    def _max_parallel_str(self):
        return self._max_parallel_allowed if self._max_parallel_allowed is not None else 'INF'

    def _default_name(self) -> str:
        return f'{self._max_parallel_str()} Parallel ' + super()._default_name()

    def _default_description(self) -> str:
        check_type = r'g == {}' if isinstance(self.gate, raw_types.Gate) else r'isinstance(g, {})'
        return (
            f'Accepts\n'
            f'1. `cirq.Gate` instances `g` s.t. `{check_type.format(self._gate_str())}` OR\n'
            f'2. `cirq.ParallelGate` instance `g` s.t. `g.sub_gate` satisfies 1. and '
            f'`cirq.num_qubits(g) <= {self._max_parallel_str()}` OR\n'
            f'3. `cirq.Operation` instance `op` s.t. `op.gate` satisfies 1. or 2.'
        )

    def _predicate(self, gate: raw_types.Gate) -> bool:
        if (
            self._max_parallel_allowed is not None
            and protocols.num_qubits(gate) > self._max_parallel_allowed
        ):
            return False
        gate = gate.sub_gate if isinstance(gate, parallel_gate.ParallelGate) else gate
        return super()._predicate(gate)

    def __repr__(self) -> str:
        name_and_description = ''
        if self.name != self._default_name() or self.description != self._default_description():
            name_and_description = (
                f'name="{self.name}", description=r\'\'\'{self.description}\'\'\', '
            )
        return (
            f'cirq.ParallelGateFamily('
            f'gate={self._gate_str(repr)}, '
            f'{name_and_description}'
            f'max_parallel_allowed={self._max_parallel_allowed})'
        )

    def _value_equality_values_(self) -> Any:
        # `isinstance` is used to ensure the a gate type and gate instance is not compared.
        return super()._value_equality_values_() + (self._max_parallel_allowed,)
