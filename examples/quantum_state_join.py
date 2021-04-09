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


def _print_bloch(vector):
    print(
        'x: ',
        np.around(vector[0], 4),
        'y: ',
        np.around(vector[1], 4),
        'z: ',
        np.around(vector[2], 4),
    )


def _run(
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ]
) -> Tuple[TStepResult, TStepResult]:
    # Initialize our qubit state space.
    circuit = cirq.Circuit()
    num_qubits = 20
    for i in range(20):
        for j in range(num_qubits):
            circuit.append(cirq.X(cirq.LineQubit(j))**random.random() if random.random() > 0.5 else cirq.Y(cirq.LineQubit(j))**random.random())

    sim = cirq.Simulator()
    qubits = tuple(cirq.LineQubit.range(num_qubits))
    args = sim.create_act_on_args(0, qubits=qubits)
    t1 = time.perf_counter()
    *_, results = sim.simulate_moment_steps(circuit, None, qubits, args)
    print(time.perf_counter() - t1)

    t1 = time.perf_counter()
    qubit_map = {q: i for i, q in enumerate(qubits)}
    argses = [sim.create_act_on_args(0, qubits=(cirq.LineQubit(j),)) for j in range(num_qubits)]
    for op in circuit.all_operations():
        j = qubit_map[op.qubits[0]]
        args = argses[j]
        args.axes = (0,)
        cirq.act_on(op, args)
    print(time.perf_counter() - t1)

    args_join = None
    for j in range(num_qubits):
        args_join = argses[j] if args_join is None else args_join.join(argses[j])
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
    sim = cirq.Simulator(seed=seed)
    results, results1 = _run(sim)
    assert np.allclose(results.state_vector(), results1.state_vector())


if __name__ == '__main__':
    main()
