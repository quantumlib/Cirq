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

from __future__ import annotations

import json

import pytest

import cirq
import cirq_web


def strip_ws(string):
    return "".join(string.split())


def test_circuit_init_type() -> None:
    qubits = [cirq.GridQubit(x, y) for x in range(2) for y in range(2)]
    moment = cirq.Moment(cirq.H(qubits[0]))
    circuit = cirq.Circuit(moment)

    circuit3d = cirq_web.Circuit3D(circuit)
    assert isinstance(circuit3d, cirq_web.Circuit3D)


@pytest.mark.parametrize('qubit', [cirq.GridQubit(0, 0), cirq.LineQubit(0)])
def test_circuit_client_code(qubit) -> None:
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
            {json.dumps(circuit_obj)}, {str(moments)}, "{circuit.id}", {circuit.padding_factor}
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


def test_circuit_client_code_unsupported_qubit_type() -> None:
    moment = cirq.Moment(cirq.H(cirq.NamedQubit('q0')))
    circuit = cirq_web.Circuit3D(cirq.Circuit(moment))

    with pytest.raises(ValueError, match='Unsupported qubit type'):
        circuit.get_client_code()


def test_circuit_default_bundle_name() -> None:
    qubits = [cirq.GridQubit(x, y) for x in range(2) for y in range(2)]
    moment = cirq.Moment(cirq.H(qubits[0]))
    circuit = cirq_web.Circuit3D(cirq.Circuit(moment))

    assert circuit.get_widget_bundle_name() == 'circuit.bundle.js'


class _UntrustedSymbolGate(cirq.testing.SingleQubitGate):
    """A gate whose circuit-diagram symbol carries an HTML/JS injection payload."""

    def _circuit_diagram_info_(self, args) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=("</script><img src=x onerror=alert(document.domain)>",)
        )


def test_circuit_client_code_escapes_untrusted_symbols() -> None:
    # Diagram symbols are attacker-controllable (a custom gate's `_circuit_diagram_info_`),
    # and a circuit may originate from an untrusted source, e.g. `cirq.read_json`. Such a
    # symbol must not be able to break out of the inline <script> that renders the circuit
    # (CWE-79).
    qubit = cirq.GridQubit(0, 0)
    circuit = cirq_web.Circuit3D(cirq.Circuit(_UntrustedSymbolGate().on(qubit)))

    client_code = circuit.get_client_code()

    payload = "</script><img src=x onerror=alert(document.domain)>"
    # The injected markup must never reach the HTML tokenizer: the payload must not
    # appear, no "</script>" should come from the data (only the one legitimate closing
    # tag of the template remains), and the breakout sequence must be absent.
    assert payload not in client_code
    assert client_code.count("</script>") == 1
    assert "</script><img" not in client_code
    # The dangerous characters survive only in their escaped, inert form.
    assert "\\u003c/script\\u003e" in client_code
    # The escaped payload still round-trips back to the original data.
    decoded = json.loads(circuit._serialize_circuit())
    assert decoded[0]['wire_symbols'] == [payload]
