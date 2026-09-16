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

from cirq import circuits, ops, protocols, transformers
from cirq.transformers import transformer_api

@transformer_api.transformer
def apply_lazy_args_on_circuit_operation(
    circuit: circuits.AbstractCircuit,
    *,
    context: transformer_api.TransformerContext | None = None,
    apply_param_resolver: bool = True,
    apply_measurement_key_map: bool = True,
) -> circuits.AbstractCircuit:
    """Eagerly apply "lazy" arguments on CircuitOperation.

    When resolving parameters in a circuit containing a CircuitOperation,
    e.g. using cirq.resolve_parameters(), you may notice that the parameters
    still appear inside of the circuit operation and the resolver is applied
    outside of it. This function applies the resolver within the CircuitOperation.
    It can also do the same for measurement key maps.

    Args:
        circuit: The circuit containing a CircuitOperation that has a parameter
            resolver, a measurement key map, or a qubit map applied
            to it.
        context: The transformer context.
        apply_param_resolver: Whether to apply the parameter resolver within the
            circuit operation.
        apply_measurement_key_map: Whether to apply the measurement key map within
            the circuit operation.

    Returns:
        A new circuit with the arguments applied to the circuit operation.
    """
    context = context or transformer_api.TransformerContext(deep=True)

    memo: dict[circuits.FrozenCircuit, set[circuits.FrozenCircuit]] = {}

    def func(op: ops.Operation, idx: int) -> ops.Operation:
        if not isinstance((circuit_op := op.untagged), circuits.CircuitOperation):
            return op

        replace_dict: dict[str, None] = {}
        raw_circuit: circuits.FrozenCircuit = circuit_op.circuit
        if apply_param_resolver and circuit_op.param_resolver:
            raw_circuit = protocols.resolve_parameters(
                raw_circuit, circuit_op.param_resolver, recursive=True
            )
            replace_dict["param_resolver"] = None

        if apply_measurement_key_map and circuit_op.measurement_key_map:
            raw_circuit = protocols.with_measurement_key_mapping(
                raw_circuit, circuit_op.measurement_key_map
            )
            replace_dict["measurement_key_map"] = None
            
        final_circuit = transformers.map_operations(raw_circuit, func, deep=False)

        # Conserve circuit identity (in particular if
        # `initial_circuit_1 is initial_circuit_2 and final_circuit_1 == final_circuit_2`
        # then `final_circuit_1 is final_circuit_2`)
        initial_circuit = circuit_op.circuit
        if initial_circuit in memo:
            for _circuit in memo[initial_circuit]:
                if _circuit == final_circuit:
                    final_circuit = _circuit
                    break
            else:
                memo[initial_circuit].add(final_circuit)
        else:
            memo[initial_circuit] = {final_circuit}

        return circuit_op.replace(circuit=final_circuit, **replace_dict).with_tags(*op.tags)

    return transformers.map_operations(circuit, func, deep=False)