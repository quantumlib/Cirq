"""Quantum Teleportation.
Quantum Teleportation is a process by which a quantum state can be transmitted
by sending only two classical bits of information. This is accomplished by
pre-sharing an entangled state between the sender (Alice) and the receiver
(Bob). This entangled state allows the receiver (Bob) of the two classical
bits of information to possess a qubit with the same state as the one held by
the sender (Alice).

In the following example output, qubit 0 (the Message) is set to a random state
by applying X and Y gates. By sending two classical bits of information after
qubit 0 (the Message) and qubit 1 (Alice's entangled qubit) are measured, the
final state of qubit 2 (Bob's entangled qubit) will be identical to the
original random state of qubit 0 (the Message). This is only possible given
that an entangled state is pre-shared between Alice and Bob.

=== REFERENCE ===
https://en.wikipedia.org/wiki/Quantum_teleportation
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.70.1895

=== EXAMPLE OUTPUT ===
Circuit:
0: -----------X^0.25---Y^0.125---@---H---M-------@---
                                 |       |       |
1: ---H---@----------------------X-------M---@---|---
          |                                  |   |
2: -------X----------------------------------X---@---

Bloch Sphere of Message After Random X and Y Gates:
x:  0.2706 y:  -0.7071 z:  0.6533

Bloch Sphere of Qubit 2 at Final State:
x:  0.2706 y:  -0.7071 z:  0.6533

"""
import time
import random
from typing import Tuple

import numpy as np
import cirq
from cirq.sim.simulator import (
    TStepResult,
    SimulatesIntermediateState,
    TSimulationTrialResult,
    TSimulatorState,
    TActOnArgs,
)


def _run(
    num_qubits: int,
    circuit_length: int,
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ]
) -> Tuple[TStepResult, TStepResult]:
    # Initialize our qubit state space.
    circuit = cirq.Circuit()
    for i in range(circuit_length):
        for j in range(num_qubits):
            if random.random() > 0.5:
                circuit.append(cirq.X(cirq.LineQubit(j)) ** random.random())
            elif random.random() > 0.5:
                circuit.append(cirq.Y(cirq.LineQubit(j)) ** random.random())
            else:
                #circuit.append(cirq.Y(cirq.LineQubit(j)) ** random.random())
                circuit.append(cirq.CX(cirq.LineQubit(j), cirq.LineQubit((j + 1) % num_qubits)))

    qubits = tuple(cirq.LineQubit.range(num_qubits))
    args = sim.create_act_on_args(0, qubits=qubits)
    t1 = time.perf_counter()
    *_, results = sim.simulate_moment_steps(circuit, None, qubits, args)
    print(time.perf_counter() - t1)

    t1 = time.perf_counter()
    qubit_map = {q: i for i, q in enumerate(qubits)}
    qubitses = [(cirq.LineQubit(j),) for j in range(num_qubits)]
    argses = [sim.create_act_on_args(0, qubits=qubitses[j]) for j in range(num_qubits)]
    for op in circuit.all_operations():
        full_args = None
        full_qubits = tuple()
        for q in op.qubits:
            if q not in full_qubits:
                j = qubit_map[q]
                full_args = argses[j] if full_args is None else full_args.join(argses[j])
                full_qubits = full_qubits + qubitses[j]
        for q in full_qubits:
            j = qubit_map[q]
            argses[j] = full_args
            qubitses[j] = full_qubits
        full_qubit_map = {q: i for i, q in enumerate(full_qubits)}
        full_args.axes = tuple(full_qubit_map[q] for q in op.qubits)
        cirq.act_on(op, full_args)
    print(time.perf_counter() - t1)

    args_join = None
    joined = set()
    for j in range(num_qubits):
        if argses[j] not in joined:
            args_join = argses[j] if args_join is None else args_join.join(argses[j])
            joined.add(argses[j])
    print(time.perf_counter() - t1)

    *_, results1 = sim.simulate_moment_steps(cirq.Circuit(), None, qubits, args_join)

    print(time.perf_counter() - t1)
    return results, results1


def main(seed=None):
    """Run a simple simulation of quantum teleportation.

    Args:
        seed: The seed to use for the simulation.
    """
    random.seed(seed)

    # Run with density matrix simulator
    print('***Run with density matrix simulator***')
    sim = cirq.DensityMatrixSimulator(seed=seed)
    results, results1 = _run(num_qubits=10, circuit_length=10, sim=sim)


if __name__ == '__main__':
    main()
