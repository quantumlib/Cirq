# Copyright 2018 The ops Developers
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

import numpy as np

from cirq import ops, circuits, linalg, google

from cirq.contrib.paulistring.pauli_string_phasor import PauliStringPhasor


def converted_gate_set(circuit: circuits.Circuit, atol: float = 1e-7
                       ) -> circuits.Circuit:
    """Returns a new, equivalent circuit using the gate set {CliffordGate,
    PauliInteractionGate, PauliStringPhasor}.

    The circuit structure may differ because it is optimized during conversion.
    """
    xmon_circuit = google.optimized_for_xmon(circuit, allow_partial_czs=False)

    qubits = circuit.all_qubits()
    tol = linalg.Tolerance(atol=atol)

    def is_clifford_rotation(half_turns):
        return tol.all_near_zero_mod(half_turns, 0.5)

    def to_quarter_turns(half_turns):
        return round(2 * half_turns) % 4

    def is_quarter_turn(half_turns):
        return (is_clifford_rotation(half_turns) and
                to_quarter_turns(half_turns) % 2 == 1)

    def is_half_turn(half_turns):
        return (is_clifford_rotation(half_turns) and
                to_quarter_turns(half_turns) == 2)

    def is_no_turn(half_turns):
        return (is_clifford_rotation(half_turns) and
                to_quarter_turns(half_turns) == 0)

    def rotation_to_clifford_op(pauli, qubit, half_turns):
        quarter_turns = to_quarter_turns(half_turns)
        if quarter_turns == 0:
            return ops.CliffordGate.I(qubit)
        elif quarter_turns == 2:
            return ops.CliffordGate.from_pauli(pauli)(qubit)
        else:
            gate = ops.CliffordGate.from_pauli(pauli, True)(qubit)
            if quarter_turns == 3:
                gate = gate.inverse()
            return gate

    def rotation_to_non_clifford_op(pauli, qubit, half_turns):
        return PauliStringPhasor(ops.PauliString.from_single(qubit, pauli),
                                 half_turns=half_turns)

    def single_qubit_matrix_to_ops(mat, qubit):
        # Decompose matrix
        z_rad_before, y_rad, z_rad_after = (
            linalg.deconstruct_single_qubit_matrix_into_angles(mat))
        z_ht_before = z_rad_before / np.pi - 0.5
        m_ht = y_rad / np.pi
        m_pauli = ops.Pauli.X
        z_ht_after = z_rad_after / np.pi + 0.5

        # Clean up angles
        if is_clifford_rotation(z_ht_before):
            if is_quarter_turn(z_ht_before) or is_quarter_turn(z_ht_after):
                z_ht_before += 0.5
                z_ht_after -= 0.5
                m_pauli = ops.Pauli.Y
            if is_half_turn(z_ht_before) or is_half_turn(z_ht_after):
                z_ht_before -= 1
                z_ht_after += 1
                m_ht = -m_ht
        if is_no_turn(m_ht):
            z_ht_before += z_ht_after
            z_ht_after = 0
        elif is_half_turn(m_ht):
            z_ht_after -= z_ht_before
            z_ht_before = 0

        # Generate operations
        rotation_list = [
            (ops.Pauli.Z, z_ht_before),
            (m_pauli, m_ht),
            (ops.Pauli.Z, z_ht_after)]
        is_clifford_list = [is_clifford_rotation(ht)
                            for pauli, ht in rotation_list]
        op_list = [rotation_to_clifford_op(pauli, qubit, ht) if is_clifford
                   else rotation_to_non_clifford_op(pauli, qubit, ht)
                   for is_clifford, (pauli, ht) in
                        zip(is_clifford_list, rotation_list)]

        # Merge adjacent Clifford gates
        for i in reversed(range(len(op_list) - 1)):
            if is_clifford_list[i] and is_clifford_list[i+1]:
                op_list[i] = op_list[i].gate.merged_with(op_list[i+1].gate
                                                         )(qubit)
                is_clifford_list.pop(i+1)
                op_list.pop(i+1)

        # Yield non-identity ops
        for is_clifford, op in zip(is_clifford_list, op_list):
            if is_clifford and op.gate == ops.CliffordGate.I:
                continue
            yield op

    def generate_ops():
        conv_cz = ops.PauliInteractionGate(ops.Pauli.Z, False,
                                           ops.Pauli.Z, False)
        matrices = {qubit: np.eye(2) for qubit in qubits}

        def clear_matrices(qubits):
            for qubit in qubits:
                yield single_qubit_matrix_to_ops(matrices[qubit], qubit)
                matrices[qubit] = np.eye(2)

        for op in xmon_circuit.all_operations():
            # Assumes all Xmon operations are GateOperation
            gate = op.gate
            if isinstance(gate, google.XmonMeasurementGate):
                yield clear_matrices(op.qubits)
                yield ops.MeasurementGate(gate.key, gate.invert_mask
                                          )(*op.qubits)
            elif len(op.qubits) == 1:
                # Collect all one qubit rotations
                # Assumes all Xmon gates implement KnownMatrix
                qubit, = op.qubits
                matrices[qubit] = op.matrix().dot(matrices[qubit])
            elif isinstance(gate, google.Exp11Gate):
                yield clear_matrices(op.qubits)
                if gate.half_turns != 1:
                    # coverage: ignore
                    raise ValueError('Unexpected partial CZ: {}'.format(op))
                yield conv_cz(*op.qubits)
            else:
                # coverage: ignore
                raise TypeError('Unknown Xmon operation: {}'.format(op))

        yield clear_matrices(qubits)

    return circuits.Circuit.from_ops(
                generate_ops(),
                strategy=circuits.InsertStrategy.EARLIEST)
