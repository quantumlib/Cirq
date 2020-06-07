# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List, Optional, Sequence

import numpy as np

import cirq
from cirq import quirk_url_to_circuit


def assert_url_to_circuit_returns(
        json_text: str,
        circuit: 'cirq.Circuit' = None,
        *,
        unitary: Optional[np.ndarray] = None,
        diagram: Optional[str] = None,
        output_amplitudes_from_quirk: Optional[List[Dict[str, float]]] = None,
        maps: Optional[Dict[int, int]] = None):
    """
    Args:
        json_text: The part of the quirk URL after "#circuit=".
        circuit: The optional expected circuit. If specified and not
            equal to the parsed circuit, an assertion fails.
        unitary: The optional expected unitary of the circuit. If specified
            and the parsed circuit has a different unitary, an assertion fails.
        diagram: The optional expected circuit diagram. If specified and the
            parsed circuit has a different diagram, an assertion fails.
        output_amplitudes_from_quirk: Optional data copied from Quirk's "export
            simulation data" function, for comparison to Cirq's simulator
            results. If specified and the output from the simulation differs
            from this data (after accounting for differences in endian-ness),
            an assertion fails.
        maps: Optional dictionary of test computational basis input states and
            the output computational basis state that they should be mapped to.
            If any state is mapped to the wrong thing, an assertion fails. Note
            that the states are specified using Quirk's little endian
            convention, meaning that the last bit of a binary literal will refer
            to the last qubit's value instead of vice versa.
    """
    parsed = quirk_url_to_circuit(
        f'https://algassert.com/quirk#circuit={json_text}')

    if diagram is not None:
        cirq.testing.assert_has_diagram(parsed, diagram)

    if circuit is not None:
        cirq.testing.assert_same_circuits(parsed, circuit)

    if unitary is not None:
        np.testing.assert_allclose(cirq.unitary(parsed), unitary, atol=1e-8)

    if output_amplitudes_from_quirk is not None:
        expected = np.array([
            float(e['r']) + 1j * float(e['i'])
            for e in output_amplitudes_from_quirk
        ])

        np.testing.assert_allclose(
            cirq.final_state_vector(
                parsed,
                # Match Quirk's endian-ness for comparison purposes.
                qubit_order=sorted(parsed.all_qubits(), reverse=True),
            ),
            expected,
            atol=1e-8)

    if maps:
        keys = sorted(maps.keys())
        actual_map = _sparse_computational_basis_map(keys, parsed)
        for k in keys:
            assert actual_map.get(k) == maps[k], (
                f'{_bin_dec(k)} was mapped to '
                f'{_bin_dec(actual_map.get(k))} '
                f'instead of {_bin_dec(maps[k])}.')


def _sparse_computational_basis_map(inputs: Sequence[int],
                                    circuit: cirq.Circuit) -> Dict[int, int]:
    # Pick a unique amplitude for each computational basis input state.
    amps = [
        np.exp(1j * i / len(inputs)) / len(inputs)**0.5
        for i in range(len(inputs))
    ]

    # Permute the amplitudes using the circuit.
    input_state = np.zeros(1 << len(circuit.all_qubits()), dtype=np.complex128)
    for k, amp in zip(inputs, amps):
        input_state[k] = amp
    output_state = cirq.final_state_vector(circuit, initial_state=input_state)

    # Find where each amplitude went.
    actual_map = {}
    for k, amp in zip(inputs, amps):
        for i, amp2 in enumerate(output_state):
            if abs(amp2 - amp) < 1e-5:
                actual_map[k] = i

    return actual_map


def _bin_dec(x: Optional[int]) -> str:
    if x is None:
        return 'None'
    return f'{bin(x)} ({x})'
