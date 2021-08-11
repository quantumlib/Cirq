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
from typing import Iterable
import cirq
from cirq_web import widget
from cirq_web.circuits.symbols import (
    Operation3DSymbol,
    SymbolResolver,
    resolve_operation,
    DEFAULT_SYMBOL_RESOLVERS,
)


class Circuit3D(widget.Widget):
    """Takes cirq.Circuit objects and displays them in 3D."""

    def __init__(
        self,
        circuit: cirq.Circuit,
        resolvers: Iterable[SymbolResolver] = DEFAULT_SYMBOL_RESOLVERS,
        padding_factor: float = 1,
    ):
        """Initializes a Circuit instance.

        Args:
            circuit: The cirq.Circuit to be represented in 3D.
        """
        super().__init__()
        self.circuit = circuit
        self._resolvers = resolvers
        self.padding_factor = padding_factor

    def get_client_code(self) -> str:
        # Remove hyphens from the id so that we can use
        # it as the variable name in TS.
        # It's important that we assign the circuit to a variable
        # for animation purposes. Alternatively, there may be ways
        # to select/manipulate elements on the screen from three.js
        stripped_id = self.id.replace('-', '')
        moments = len(self.circuit.moments)
        self.serialized_circuit = self._serialize_circuit()

        return f"""
            <button id="camera-reset">Reset Camera</button>
            <button id="camera-toggle">Toggle Camera Type</button>
            <script>
            let viz_{stripped_id} = createGridCircuit({self.serialized_circuit}, {moments}, "{self.id}", {self.padding_factor});

            document.getElementById("camera-reset").addEventListener('click', ()  => {{
            viz_{stripped_id}.scene.setCameraAndControls(viz_{stripped_id}.circuit);
            }});

            document.getElementById("camera-toggle").addEventListener('click', ()  => {{
            viz_{stripped_id}.scene.toggleCamera(viz_{stripped_id}.circuit);
            }});
            </script>
        """

    def get_widget_bundle_name(self) -> str:
        return 'circuit.bundle.js'

    def _serialize_circuit(self) -> str:
        args = []
        moments = self.circuit.moments
        for moment_id, moment in enumerate(moments):
            for item in moment:
                symbol = self._build_3D_symbol(item, moment_id)
                args.append(symbol.to_typescript())

        argument_str = ','.join(str(item) for item in args)
        return f'[{argument_str}]'

    def _build_3D_symbol(self, operation, moment) -> Operation3DSymbol:
        symbol_info = resolve_operation(operation, self._resolvers)
        location_info = []
        for qubit in operation.qubits:
            location_info.append({'row': qubit.row, 'col': qubit.col})
        return Operation3DSymbol(symbol_info.labels, location_info, symbol_info.colors, moment)
