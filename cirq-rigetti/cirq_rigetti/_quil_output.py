##############################################################################
# Copyright 2021 The Cirq Developers
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

from typing import Callable, Dict, Set, Tuple, Optional, List
import cirq


class RigettiQCSQuilOutput(cirq.QuilOutput):
    """A sub-class of `cirq.circuits.quil_output.QuilOutput` that additionally accepts a
    `qubit_id_map` for explicitly mapping logical qubits to physical qubits.

    Attributes:
        qubit_id_map: A dictionary mapping `cirq.Qid` to strings that
            address physical qubits in the outputted QUIL.
        measurement_id_map: A dictionary mapping a Cirq measurement key to
            the corresponding QUIL memory region.
        formatter: A QUIL formatter that formats QUIL strings account for both
            the `qubit_id_map` and `measurement_id_map`.
    """

    def __init__(
        self,
        *,
        operations: cirq.OP_TREE,
        qubits: Tuple[cirq.Qid, ...],
        decompose_operation: Optional[Callable[[cirq.Operation], List[cirq.Operation]]] = None,
        qubit_id_map: Optional[Dict[cirq.Qid, str]] = None,
    ):
        """Initializes an instance of `RigettiQCSQuilOutput`.

        Args:
            operations: A list or tuple of `cirq.OP_TREE` arguments.
            qubits: The qubits used in the operations.
            decompose_operation: Optional; A callable that decomposes a circuit operation
                into a list of equivalent operations. If None provided, this class
                decomposes operations by invoking `QuilOutput._write_quil`.
            qubit_id_map: Optional; A dictionary mapping `cirq.Qid` to strings that
                address physical qubits in the outputted QUIL.
        """
        super().__init__(operations, qubits)
        self.qubit_id_map = qubit_id_map or self._generate_qubit_ids()
        self.measurement_id_map = self._generate_measurement_ids()

        self.formatter = cirq.QuilFormatter(
            qubit_id_map=self.qubit_id_map, measurement_id_map=self.measurement_id_map
        )
        self._decompose_operation = decompose_operation

    def _write_quil(self, output_func: Callable[[str], None]) -> None:
        """Calls `output_func` for successive lines of QUIL output.

        Args:
            output_func: A function that accepts a string of QUIL. This will likely
                write the QUIL to a file.

        Returns:
            None.
        """
        if self._decompose_operation is None:
            return super()._write_quil(output_func)

        output_func("# Created using Cirq.\n\n")

        if len(self.measurements) > 0:
            measurements_declared: Set[str] = set()
            for m in self.measurements:
                key = cirq.measurement_key(m)
                if key in measurements_declared:
                    continue
                measurements_declared.add(key)
                output_func(f"DECLARE {self.measurement_id_map[key]} BIT[{len(m.qubits)}]\n")
            output_func("\n")

        for main_op in self.operations:
            decomposed = self._decompose_operation(main_op)
            for decomposed_op in decomposed:
                output_func(cirq.quil(decomposed_op, formatter=self.formatter))
