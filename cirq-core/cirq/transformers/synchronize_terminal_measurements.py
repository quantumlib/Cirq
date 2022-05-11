# Copyright 2022 The Cirq Developers
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

"""Transformer pass to move terminal measurements to the end of circuit."""

from typing import List, Optional, Set, Tuple, TYPE_CHECKING
from cirq import protocols, circuits
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import cirq


def find_terminal_measurements(
    circuit: 'cirq.AbstractCircuit',
) -> List[Tuple[int, 'cirq.Operation']]:
    """Finds all terminal measurements in the given circuit.

    A measurement is terminal if there are no other operations acting on the measured qubits
    after the measurement operation occurs in the circuit.

    Args:
        circuit: The circuit to find terminal measurements in.

    Returns:
        List of terminal measurements, each specified as (moment_index, measurement_operation).
    """

    open_qubits: Set['cirq.Qid'] = set(circuit.all_qubits())
    seen_control_keys: Set['cirq.MeasurementKey'] = set()
    terminal_measurements: List[Tuple[int, 'cirq.Operation']] = []
    for i in range(len(circuit) - 1, -1, -1):
        moment = circuit[i]
        for q in open_qubits:
            op = moment.operation_at(q)
            if (
                op is not None
                and open_qubits.issuperset(op.qubits)
                and protocols.is_measurement(op)
                and not (seen_control_keys & protocols.measurement_key_objs(op))
            ):
                terminal_measurements.append((i, op))
        open_qubits -= moment.qubits
        seen_control_keys |= protocols.control_keys(moment)
        if not open_qubits:
            break
    return terminal_measurements


@transformer_api.transformer(add_deep_support=True)
def synchronize_terminal_measurements(
    circuit: 'cirq.AbstractCircuit',
    *,
    context: Optional['cirq.TransformerContext'] = None,
    after_other_operations: bool = True,
) -> 'cirq.Circuit':
    """Move measurements to the end of the circuit.

    Move all measurements in a circuit to the final moment, if it can accommodate them (without
    overlapping with other operations). If `after_other_operations` is true, then a new moment will
    be added to the end of the circuit containing all the measurements that should be brought
    forward.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.
          after_other_operations: Set by default. If the circuit's final moment contains
                non-measurement operations and this is set then a new empty moment is appended to
                the circuit before pushing measurements to the end.
    Returns:
          Copy of the transformed input circuit.
    """
    if context is None:
        context = transformer_api.TransformerContext()
    terminal_measurements = [
        (i, op)
        for i, op in find_terminal_measurements(circuit)
        if set(op.tags).isdisjoint(context.tags_to_ignore)
    ]
    ret = circuit.unfreeze(copy=True)
    if not terminal_measurements:
        return ret

    ret.batch_remove(terminal_measurements)
    if ret[-1] and after_other_operations:
        ret.append(circuits.Moment())
    ret[-1] = ret[-1].with_operations(op for _, op in terminal_measurements)
    return ret
