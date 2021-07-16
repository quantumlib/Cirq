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

from cirq_web import widget
from cirq_web.circuits.gates import Gate3DSymbols, UnknownSingleQubitGate, UnknownTwoQubitGate, SingleQubitGate, ControlledGate

class Circuit3D(widget.Widget):
    def __init__(self, circuit):
        """Initializes a Circuit instance.
        
        Args:
            circuit: The cirq.Circuit to be represented in 3D.
        """
        super().__init__()
        self.circuit = circuit
        pass

    def get_client_code(self) -> str:
        # Remove hyphens from the id so that we can use
        # it as the variable name in TS.
        # It's important that we assign the circuit to a variable
        # for animation purposes. Alternatively, there may be ways
        # to select/manipulate elements on the screen from three.js
        stripped_id = self.id.replace('-', '')
        qubits = self._init_qubits()
        moments = len(self.circuit.moments)
        serialized_circuit = self._serialize_circuit()

        return f"""
            <div id="test">
            <script>
            let viz_{stripped_id} = createGridCircuit({qubits}, {moments}, "{self.id}");
            viz_{stripped_id}.displayGatesFromList({serialized_circuit})
            </script>
        """

    def get_widget_bundle_name(self) -> str:
        return 'circuit.bundle.js'

    def _init_qubits(self):
        init_tuples = []
        for qubit in self.circuit.all_qubits():
            init_tuples.append({'row': qubit.row, 'col': qubit.col})

        fn_argument = ','.join(str(item) for item in init_tuples)
        return f'[{fn_argument}]'

    def _serialize_circuit(self):
        args = []
        moments = self.circuit.moments
        for moment_id, moment in enumerate(moments):
            for item in moment:
                gate = Gate3DSymbols.get(str(item.gate), None)
                if isinstance(gate, SingleQubitGate):
                    gate.set_text(item.gate.circuit_diagram_info)
                    gate.set_moment(moment_id)
                    gate.set_location(item.qubits[0].row, item.qubits[0].col)
                if isinstance(gate, ControlledGate):
                    gate.set_moment(moment_id)
                    gate.set_location(item.qubits[0].row, item.qubits[0].col)

                    gate.target_gate.set_moment(moment_id)
                    gate.target_gate.set_location(item.qubits[1].row, item.qubits[1].col)
                else:
                    if len(item.qubits) == 1:
                        gate = UnknownSingleQubitGate()
                        gate.set_moment(moment_id)
                        gate.set_location(item.qubits[0].row, item.qubits[0].col)
                    else:
                        gate = UnknownTwoQubitGate()
                        gate.set_moment(moment_id)
                        gate.set_location(item.qubits[0].row, item.qubits[0].col)

                        gate.target_gate.set_moment(moment_id)
                        gate.target_gate.set_location(item.qubits[1].row, item.qubits[1].col)
                args.append(gate.to_typescript())

        argument_str = ','.join(str(item) for item in args)
        return f'[{argument_str}]'
