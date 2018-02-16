import random

from cirq.circuits import Circuit
from cirq.google import XmonDevice, Exp11Gate, XmonQubit, ExpWGate, ExpZGate


def generate_supremacy_circuit(
        xmon_device: XmonDevice,
        cz_depth: int) -> Circuit:

    circuit = Circuit()
    cz = Exp11Gate()

    def single_qubit_layer():
        vals = [random.randint(0, 7) / 4.0 for _ in xmon_device.qubits]
        phases = [random.randint(0, 7) / 4.0 for _ in xmon_device.qubits]
        return [
            ExpWGate(half_turns=v, axis_half_turns=w).on(q)
            for v, w, q in zip(vals, phases, xmon_device.qubits)
            if v
        ]

    i = 0
    while cz_depth:
        cz_offset = (i >> 1) % 4
        dx = i % 2
        dy = 1 - dx
        cz_layer = [
            cz.on(q, XmonQubit(q.x + dx, q.y + dy))
            for q in xmon_device.qubits
            if (q.x * (2 - dx) + q.y * (2 - dy)) % 4 == cz_offset
            if XmonQubit(q.x + dx, q.y + dy) in xmon_device.qubits
        ]
        if cz_layer:
            circuit.append(single_qubit_layer())
            circuit.append(cz_layer)
            cz_depth -= 1
        i += 1

    circuit.append(single_qubit_layer())

    return circuit
