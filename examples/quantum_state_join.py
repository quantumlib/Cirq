import random
import time
from typing import cast, Optional, Sequence

import numpy as np

import cirq
from cirq.sim.simulator import (
    TStepResult,
    SimulatesIntermediateState,
    TSimulationTrialResult,
    TSimulatorState,
    TActOnArgs,
)


def _run_normal_simulation(
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ],
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.Qid],
) -> cirq.StepResult:
    args = sim.create_act_on_args(0, qubits=qubits)
    *_, results = sim.simulate_moment_steps(circuit, None, qubits, args)
    return results


def _run_join_args_simulation(
    sim: SimulatesIntermediateState[
        TStepResult, TSimulationTrialResult, TSimulatorState, TActOnArgs
    ],
    circuit: cirq.Circuit,
    qubits: Sequence[cirq.Qid],
) -> cirq.StepResult:
    # Initialize the ActOnArgs, one per qubit.
    args_map = {q: sim.create_act_on_args(0, qubits=[q]) for q in qubits}

    # Iterate
    for op in circuit.all_operations():
        # Go through the op's qubits and join any disparate ActOnArgs states
        # into a new combined state.
        op_args: Optional[TActOnArgs] = None
        for q in op.qubits:
            if op_args is None or q not in op_args.qubits:
                op_args = (
                    args_map[q] if op_args is None else cast(TActOnArgs, op_args).join(args_map[q])
                )
        assert op_args is not None

        # (Backfill the args map with the new value)
        for q in op_args.qubits:
            args_map[q] = op_args

        # Act on the args with the operation
        op_args.axes = tuple(op_args.qubit_map[q] for q in op.qubits)
        cirq.act_on(op, op_args)

    # Create the final ActOnArgs by joining everything together
    final_args = None
    for args in set(args_map.values()):
        final_args = args if final_args is None else cast(TActOnArgs, final_args).join(args)

    # Run an empty simulation to get the results
    *_, results = sim.simulate_moment_steps(cirq.Circuit(), None, qubits, final_args)
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
