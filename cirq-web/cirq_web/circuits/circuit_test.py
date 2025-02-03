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
import cirq
import cirq_web

import pytest


def strip_ws(string):
    return "".join(string.split())


def test_circuit_init_type():
    qubits = [cirq.GridQubit(x, y) for x in range(2) for y in range(2)]
    moment = cirq.Moment(cirq.H(qubits[0]))
    circuit = cirq.Circuit(moment)

    circuit3d = cirq_web.Circuit3D(circuit)
    assert isinstance(circuit3d, cirq_web.Circuit3D)


@pytest.mark.parametrize('qubit', [cirq.GridQubit(0, 0), cirq.LineQubit(0)])
def test_circuit_client_code(qubit):
    moment = cirq.Moment(cirq.H(qubit))
    circuit = cirq_web.Circuit3D(cirq.Circuit(moment))

    circuit_obj = [
        {
            'wire_symbols': ['H'],
            'location_info': [{'row': 0, 'col': 0}],
            'color_info': ['yellow'],
            'moment': 0,
        }
    ]

    moments = 1
    stripped_id = circuit.id.replace('-', '')

    expected_client_code = f"""
        <button id="camera-reset">Reset Camera</button>
        <button id="camera-toggle">Toggle Camera Type</button>
        <script>
        let viz_{stripped_id} = createGridCircuit(
            {str(circuit_obj)}, {str(moments)}, "{circuit.id}", {circuit.padding_factor}
        );

        document.getElementById("camera-reset").addEventListener('click', ()  => {{
        viz_{stripped_id}.scene.setCameraAndControls(viz_{stripped_id}.circuit);
        }});

        document.getElementById("camera-toggle").addEventListener('click', ()  => {{
        viz_{stripped_id}.scene.toggleCamera(viz_{stripped_id}.circuit);
        }});
        </script>
        """

    assert strip_ws(circuit.get_client_code()) == strip_ws(expected_client_code)


def test_circuit_client_code_unsupported_qubit_type():
    moment = cirq.Moment(cirq.H(cirq.NamedQubit('q0')))
    circuit = cirq_web.Circuit3D(cirq.Circuit(moment))

    with pytest.raises(ValueError, match='Unsupported qubit type'):
        circuit.get_client_code()


def test_circuit_default_bundle_name():
    qubits = [cirq.GridQubit(x, y) for x in range(2) for y in range(2)]
    moment = cirq.Moment(cirq.H(qubits[0]))
    circuit = cirq_web.Circuit3D(cirq.Circuit(moment))

    assert circuit.get_widget_bundle_name() == 'circuit.bundle.js'
