# Copyright 2026 The Cirq Developers
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

from cirq import circuits, devices, protocols, ops
import cirq.transformers.apply_args_to_circuit_operation as aaco
import sympy
from typing import cast
from cirq.circuits.circuit import CIRCUIT_TYPE

class TestApplyLazyArgs:
    def setup_method(self) -> None:
        self.key0, self.key1 = "old", "new"
        self.var_name = "x"
        self.q0, self.q1 = devices.LineQubit.range(2)

        q0_new, q1_new = devices.GridQubit.rect(rows=1, cols=2)

        inner_circuit = circuits.FrozenCircuit(
            ops.H(self.q0) ** sympy.Symbol(self.var_name),
            ops.CNOT(self.q0, self.q1),
            ops.measure(self.q0, self.q1, key=self.key0),
        )
        mid_circuit = circuits.FrozenCircuit(circuits.CircuitOperation(inner_circuit))
        self.tags = ("asdf",)
        self.circuit = circuits.FrozenCircuit(
            circuits.CircuitOperation(mid_circuit.with_tags(*self.tags))
        )

        mapped_circuit = protocols.resolve_parameters(
            self.circuit,
            {self.var_name: 1}, recursive=True
        )
        mapped_circuit = protocols.with_measurement_key_mapping(
            mapped_circuit,
            {self.key0: self.key1}
        )
        self.mapped_circuit = mapped_circuit.unfreeze()

    @staticmethod
    def get_inner(circuit: CIRCUIT_TYPE, outer_moment: int = 0) -> circuits.FrozenCircuit:
        mid = cast(circuits.CircuitOperation, circuit.moments[outer_moment].operations[0]).circuit
        inner = cast(circuits.CircuitOperation, mid.moments[0].operations[0]).circuit
        return inner

    def test_apply_lazy_param_resolver(self) -> None:
        applied_param_circuit = aaco.apply_lazy_args_on_circuit_operation(self.mapped_circuit)

        assert protocols.parameter_names(self.get_inner(self.circuit)) == {self.var_name}
        assert protocols.parameter_names(self.get_inner(self.mapped_circuit)) == {self.var_name}
        assert protocols.parameter_names(self.get_inner(applied_param_circuit)) == set()

    def test_apply_lazy_param_resolver_preserves_inner_tags(self) -> None:
        applied_param_circuit = aaco.apply_lazy_args_on_circuit_operation(self.circuit)

        op = applied_param_circuit[0].operations[0]
        assert isinstance(op, circuits.CircuitOperation)
        assert op.circuit.tags == self.tags

    def test_apply_lazy_measurement_key_map(self) -> None:
        applied_param_circuit = aaco.apply_lazy_args_on_circuit_operation(self.mapped_circuit)

        assert protocols.measurement_key_names(self.get_inner(self.circuit)) == {self.key0}
        assert protocols.measurement_key_names(self.get_inner(self.mapped_circuit)) == {self.key0}
        assert protocols.measurement_key_names(self.get_inner(applied_param_circuit)) == {self.key1}

    def test_apply_only_param_resolver(self) -> None:
        out_circuit = aaco.apply_lazy_args_on_circuit_operation(
            self.mapped_circuit,
            apply_param_resolver=True,
            apply_measurement_key_map=False,
        )
        assert protocols.measurement_key_names(self.get_inner(out_circuit)) == {self.key0}
        assert protocols.parameter_names(self.get_inner(out_circuit)) == set()

    def test_apply_only_measurement_key_map(self) -> None:
        out_circuit = aaco.apply_lazy_args_on_circuit_operation(
            self.mapped_circuit,
            apply_param_resolver=False,
            apply_measurement_key_map=True,
        )
        assert protocols.measurement_key_names(self.get_inner(out_circuit)) == {self.key1}
        assert protocols.parameter_names(self.get_inner(out_circuit)) == {self.var_name}

    def test_circuit_identity_before_leads_to_circuit_identify_after(self) -> None:
        self.mapped_circuit.append(self.mapped_circuit.moments[0])

        assert self.get_inner(self.mapped_circuit, outer_moment=0) is self.get_inner(
            self.mapped_circuit, outer_moment=1
        )
        out_circuit = aaco.apply_lazy_args_on_circuit_operation(
            self.mapped_circuit,
            apply_param_resolver=True,
            apply_measurement_key_map=True,
        )

        assert self.get_inner(out_circuit, outer_moment=0) == self.get_inner(
            out_circuit, outer_moment=1
        )
        assert self.get_inner(out_circuit, outer_moment=0) is self.get_inner(
            out_circuit, outer_moment=1
        )
