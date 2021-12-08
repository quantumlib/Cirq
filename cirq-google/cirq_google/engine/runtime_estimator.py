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

"""Utility functions to estimate runtime using Engine to execute circuits.

Users can call estimate_run_time, estimate_run_sweep_time, or
estimate_run_batch_time to retrieve approximations of runtime on QCS
of various sizes and shapes of circuits.

Times were extrapolated from actual runs on Sycamore processors duing
November 2021.  These times should only be used as a rough guide.
Your experience may vary based on many factors not captured here.

Parameters were calculated using a variety of width/depth/sweeps from
the rep rate calculator, see:

https://github.com/quantumlib/ReCirq/blob/master/recirq/benchmarks/rep_rate/

Model was then fitted by hand, correcting for anomalies and outliers
when possible.

"""
from typing import List, Optional, Sequence
import cirq

# Estimated end-to-end latency of circuits through the system
_BASE_LATENCY = 1.5


def _initial_time(width, depth, sweeps):
    """Estimates the initiation time of a circuit.

    This estimate includes tasks like electronics setup, gate compiling,
    and throughput of constant-time data.

    This time depends on the size of the circuits being compiled
    (width and depth) and also includes a factor for the number of
    times the compilation is done (sweeps).  Since sweeps save some of
    the work, this factor scales less than one.

    Args:
        width: number of qubits
        depth: number of moments
        sweeps: total number of parameter sweeps
    """
    return ((width / 8) * (depth / 125) + (width / 12)) * max(1, sweeps / 5)


def _rep_time(width: int, depth: int, sweeps: int, reps: int) -> float:
    """Estimated time of executing repetitions.

    This includes all incremental costs of executing a repetition and of
    sending data back and forth from the electronics.

    This is based on an approximate rep rate for "fast" circuits at about
    24k reps per second.  More qubits measured (width) primarily slows
    this down, with an additional factor for very high depth circuits.

    For multiple sweeps, there is some additional cost, since not all
    sweeps can be batched together.  Sweeping in general is more efficient,
    but it is not perfectly parallel.  Sweeping also seems to be more
    sensitive to the number of qubits measured, for reasons not understood.

    Args:
        width: number of qubits
        depth: number of moments
        sweeps: total number of parameter sweeps
        reps: number of repetitions per parameter sweep
    """
    total_reps = sweeps * reps
    rep_rate = 24000 / (0.9 + width / 38) / (0.9 + depth / 5000)
    if sweeps > 1:
        rep_rate *= 0.72
        rep_rate *= 1 - (width - 25) / 40
    return total_reps / rep_rate


def _estimate_run_time_seconds(
    width: int, depth: int, sweeps: int, repetitions: int, latency: Optional[float] = _BASE_LATENCY
) -> float:
    """Returns an approximate number of seconds for execution of a single circuit.

    This includes the total cost of set up (initial time), cost per repetition (rep_time),
    and a base end-to-end latency cost of operation (configurable).


    Args:
        width: number of qubits
        depth: number of moments
        sweeps: total number of parameter sweeps
        repetitions: number of repetitions per parameter sweep
        latency: Optional latency to add (defaults to 1.5 seconds)
    """
    init_time = _initial_time(width, depth, sweeps)
    rep_time = _rep_time(width, depth, sweeps, repetitions)
    return rep_time + init_time + latency


def estimate_run_time(
    program: cirq.AbstractCircuit, repetitions: int, latency: Optional[float] = _BASE_LATENCY
) -> float:
    """Compute the estimated time for running a single circuit.

    This should approximate, in seconds, the time for the execution of a batch of circuits
    using Engine.run() on QCS at a time where there is no queue (such as a reserved slot).
    This estimation should be considered a rough approximation.  Many factors can contribute to
    the execution time of a circuit, and the run time can also vary as the service's code changes
    frequently.

    Args:
        program: circuit to be executed
        repetitions: number of repetitions to execute
        latency: Optional latency to add (defaults to 1.5 seconds)
    """
    width = len(program.all_qubits())
    depth = len(program)
    return _estimate_run_time_seconds(width, depth, 1, repetitions, latency)


def estimate_run_sweep_time(
    program: cirq.AbstractCircuit,
    params: cirq.Sweepable = None,
    repetitions: int = 1000,
    latency: Optional[float] = _BASE_LATENCY,
) -> float:
    """Compute the estimated time for running a parameter sweep across a single Circuit.

    This should approximate, in seconds, the time for the execution of a batch of circuits
    using Engine.run_sweep() on QCS at a time where there is no queue (such as a reserved slot).
    This estimation should be considered a rough approximation.  Many factors can contribute to
    the execution time of a circuit, and the run time can also vary as the service's code changes
    frequently.

    Args:
        program: circuit to be executed
        params: a parameter sweep of variable resolvers to use with the circuit
        repetitions: number of repetitions to execute per parameter sweep
        latency: Optional latency to add (defaults to 1.5 seconds)
    """
    width = len(program.all_qubits())
    depth = len(program)
    sweeps = len(list(cirq.to_resolvers(params)))
    return _estimate_run_time_seconds(width, depth, sweeps, repetitions, latency)


def estimate_run_batch_time(
    programs: Sequence[cirq.AbstractCircuit],
    params_list: List[cirq.Sweepable],
    repetitions: int = 1000,
    latency: float = _BASE_LATENCY,
) -> float:
    """Compute the estimated time for running a batch of programs.

    This should approximate, in seconds, the time for the execution of a batch of circuits
    using Engine.run_batch() on QCS at a time where there is no queue (such as a reserved slot).
    This estimation should be considered a rough approximation.  Many factors can contribute to
    the execution time of a circuit, and the run time can also vary as the service's code changes
    frequently.

    Args:
        programs: a sequence of circuits to be executed
        params_list: a parameter sweep for each circuit
        repetitions: number of repetitions to execute per parameter sweep
        latency: Optional latency to add (defaults to 1.5 seconds)
    """
    total_time = 0.0
    current_width = None
    total_depth = 0
    total_sweeps = 0
    num_circuits = 0
    for idx, program in enumerate(programs):
        width = len(program.all_qubits())
        if width != current_width:
            if num_circuits > 0:
                total_time += _estimate_run_time_seconds(
                    width, total_depth // num_circuits, total_sweeps, repetitions, 0.25
                )
            num_circuits = 0
            total_depth = 0
            total_sweeps = 0
            current_width = width
        total_depth += len(program)
        num_circuits += 1
        total_sweeps += len(list(cirq.to_resolvers(params_list[idx])))
    if num_circuits > 0:
        total_time += _estimate_run_time_seconds(
            width, total_depth // num_circuits, total_sweeps, repetitions, 0.0
        )

    return total_time + latency
