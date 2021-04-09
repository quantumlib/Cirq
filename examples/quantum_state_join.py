import time
import random
from typing import Tuple, cast

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
) -> Tuple[TActOnArgs, TActOnArgs]:
    # Initialize our qubit state space.
    circuit = cirq.Circuit()
    for i in range(circuit_length):
        for j in range(num_qubits):
            if random.random() > 0.5:
                circuit.append(cirq.X(cirq.LineQubit(j)) ** random.random())
            elif random.random() > 0.5:
                circuit.append(cirq.Y(cirq.LineQubit(j)) ** random.random())
            else:
                circuit.append(cirq.CX(cirq.LineQubit(j), cirq.LineQubit((j + 1) % num_qubits)))

    qubits = tuple(cirq.LineQubit.range(num_qubits))
    args = sim.create_act_on_args(0, qubits=qubits)
    t1 = time.perf_counter()
    *_, results = sim.simulate_moment_steps(circuit, None, qubits, args)
    results = cast(TStepResult, results)
    print(time.perf_counter() - t1)
    sam = results.sample(list(qubits), 10000)

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
    sam1 = results.sample(list(qubits), 100000)
    sam = np.transpose(sam)
    sam1 = np.transpose(sam1)
    for i in range(num_qubits):
        print(sam1[i].mean())
        print(sam[i].mean())
    return args, args_join


def main(seed=None):
    """Run a random simulation with joining args.

    Args:
        seed: The seed to use for the simulation.
    """
    random.seed(seed)

    # Run with density matrix simulator
    print('***Run with sparse simulator***')
    sim = cirq.Simulator(seed=seed)
    args, args_join = _run(num_qubits=22, circuit_length=10, sim=sim)

    print('***Run with density matrix simulator***')
    sim = cirq.DensityMatrixSimulator(seed=seed)
    args, args_join = _run(num_qubits=11, circuit_length=10, sim=sim)


if __name__ == '__main__':
    main()
