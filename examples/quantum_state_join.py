import random
import time
from typing import cast, List, Optional

import numpy as np

import cirq
from cirq.sim.simulator import (
    TStepResult,
    SimulatesIntermediateState,
    TSimulationTrialResult,
    TSimulatorState,
    TActOnArgs,
)


def _run_normal_simulation(sim, circuit: cirq.Circuit, qubits: List[cirq.Qid]) -> cirq.StepResult:
    args = sim.create_act_on_args(0, qubits=qubits)
    *_, results = sim.simulate_moment_steps(circuit, None, qubits, args)
    return results


def _run_join_args_simulation(
    sim, circuit: cirq.Circuit, qubits: List[cirq.Qid]
) -> cirq.StepResult:
    argses = {q: sim.create_act_on_args(0, qubits=[q]) for q in qubits}
    for op in circuit.all_operations():
        full_args: Optional[TActOnArgs] = None
        for q in op.qubits:
            if full_args is None or q not in full_args.qubits:
                full_args = (
                    argses[q] if full_args is None else cast(TActOnArgs, full_args).join(argses[q])
                )
        for q in full_args.qubits:
            argses[q] = full_args
        full_args.axes = tuple(full_args.qubit_map[q] for q in op.qubits)
        cirq.act_on(op, full_args)

    args_join = None
    for args in set(argses.values()):
        args_join = args if args_join is None else args_join.join(args)

    *_, results = sim.simulate_moment_steps(cirq.Circuit(), None, qubits, args_join)
    return results


def _run_comparison(
    num_qubits: int,
    circuit_length: int,
    cnot_freq: float,
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ],
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
    results = _run_normal_simulation(sim, circuit, qubits)
    print(time.perf_counter() - t1)

    t1 = time.perf_counter()
    results1 = _run_join_args_simulation(sim, circuit, qubits)
    print(time.perf_counter() - t1)

    sam = results.sample(qubits, 10000)
    sam1 = results1.sample(qubits, 10000)
    sam = np.transpose(sam)
    sam1 = np.transpose(sam1)
    for i in range(num_qubits):
        print(sam1[i].mean())
        print(sam[i].mean())


def main():
    print('***Run with sparse simulator***')
    sim = cirq.Simulator()
    _run_comparison(num_qubits=22, circuit_length=10, cnot_freq=0.15, sim=sim)

    print('***Run with density matrix simulator***')
    sim = cirq.DensityMatrixSimulator()
    _run_comparison(num_qubits=11, circuit_length=10, cnot_freq=0.15, sim=sim)


if __name__ == '__main__':
    main()
