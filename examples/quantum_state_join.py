import time
import random
from typing import Tuple, cast, List

import numpy as np
import cirq
from cirq.sim.simulator import (
    TStepResult,
    SimulatesIntermediateState,
    TSimulationTrialResult,
    TSimulatorState,
    TActOnArgs,
)


def _run_normal(sim, circuit: cirq.Circuit, qubits: List[cirq.Qid]) -> cirq.StepResult:
    qubits = list(circuit.all_qubits())
    args = sim.create_act_on_args(0, qubits=qubits)
    *_, results = sim.simulate_moment_steps(circuit, None, qubits, args)
    return results


def _run_fast(sim, circuit: cirq.Circuit, qubits: List[cirq.Qid]) -> cirq.StepResult:
    num_qubits = len(qubits)
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

    args_join = None
    joined = set()
    for j in range(num_qubits):
        if argses[j] not in joined:
            args_join = argses[j] if args_join is None else args_join.join(argses[j])
            joined.add(argses[j])

    *_, results = sim.simulate_moment_steps(cirq.Circuit(), None, qubits, args_join)
    return results


def _run(
    num_qubits: int,
    circuit_length: int,
    cnot_freq: float,
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ]
):
    qubits = cirq.LineQubit.range(num_qubits)
    circuit = cirq.Circuit()
    for i in range(circuit_length):
        for j in range(num_qubits):
            if random.random() < cnot_freq:
                circuit.append(cirq.CX(cirq.LineQubit(j), cirq.LineQubit((j + 1) % num_qubits)))
            else:
                circuit.append(cirq.H(cirq.LineQubit(j)) ** random.random())

    t1 = time.perf_counter()
    results = _run_normal(sim, circuit, qubits)
    print(time.perf_counter() - t1)

    t1 = time.perf_counter()
    results1 = _run_fast(sim, circuit, qubits)
    print(time.perf_counter() - t1)

    sam = results.sample(qubits, 10000)
    sam1 = results1.sample(qubits, 10000)
    sam = np.transpose(sam)
    sam1 = np.transpose(sam1)
    for i in range(num_qubits):
        print(sam1[i].mean())
        print(sam[i].mean())


def main(seed=None):
    """Run a random simulation with joining args.

    Args:
        seed: The seed to use for the simulation.
    """
    random.seed(seed)

    # Run with density matrix simulator
    print('***Run with sparse simulator***')
    sim = cirq.Simulator(seed=seed)
    _run(num_qubits=10, circuit_length=10, cnot_freq=.25, sim=sim)

    print('***Run with density matrix simulator***')
    sim = cirq.DensityMatrixSimulator(seed=seed)
    _run(num_qubits=11, circuit_length=10, cnot_freq=.25, sim=sim)


if __name__ == '__main__':
    main()
