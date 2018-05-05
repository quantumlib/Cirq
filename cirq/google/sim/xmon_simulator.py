# Copyright 2018 The Cirq Developers
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

"""Simulator for the Google's Xmon class quantum computers.

The simulator can be used to run all of a Circuit or to step through the
simulation Moment by Moment. The simulator requires that all gates used in
the circuit are either an XmonGate or are CompositeGate which can be
decomposed into XmonGates. Measurement gates must all have unique string keys.

A simple example:
    circuit = Circuit([Moment([X(q1), X(q2)]), Moment([CZ(q1, q2)])])
    sim = Simulator()
    results = sim.run(circuit)
"""

import math
from collections import defaultdict, Iterable
from typing import Dict, Iterator, List, Set, Union, cast
from typing import Tuple  # pylint: disable=unused-import

import numpy as np

from cirq import ops
from cirq.circuits import Circuit
from cirq.circuits.drop_empty_moments import DropEmptyMoments
from cirq.extension import Extensions
from cirq.google import xmon_gates
from cirq.google import xmon_gate_ext
from cirq.google.convert_to_xmon_gates import ConvertToXmonGates
from cirq.google.sim import xmon_stepper
from cirq.schedules import Schedule
from cirq.study import ParamResolver, Sweep, Sweepable, TrialResult


class Options:
    """Options for the Simulator.

    Attributes:
        num_prefix_qubits: Sharding of the wave function is performed over 2
            raised to this value number of qubits.
        min_qubits_before_shard: Sharding will be done only for this number
            of qubits or more. The default is 18.
    """

    def __init__(self,
                 num_shards: int=None,
                 min_qubits_before_shard: int=18) -> None:
        """Simulator options constructor.

        Args:
            num_shards: sharding will be done for the greatest value of a
                power of two less than this value. If None, the default will
                be used which is the smallest power of two less than or equal
                to the number of CPUs.
            min_qubits_before_shard: Sharding will be done only for this number
                of qubits or more. The default is 18.
        """
        assert num_shards is None or num_shards > 0, (
            "Num_shards cannot be less than 1.")
        if num_shards is None:
            self.num_prefix_qubits = None
        else:
            self.num_prefix_qubits = int(math.log(num_shards, 2))

        assert min_qubits_before_shard >= 0, (
            'Min_qubit_before_shard must be positive.')
        self.min_qubits_before_shard = min_qubits_before_shard


class SimulatorTrialResult(TrialResult):
    """Results of a simulation run.

    Attributes:
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a list of lists (a numpy ndarray),
            the first list corresponding to the repetition, and the second is
            the actual boolean measurement results (ordered by the qubits acted
            the measurement gate.)
        final_states: The final states (wave function) of the system after
            the trial finishes.
    """

    def __init__(self,
                 params: ParamResolver,
                 repetitions: int,
                 measurements: Dict[str, np.ndarray],
                 final_states: List[np.ndarray] = None) -> None:
        self.params = params
        self.repetitions = repetitions
        self.measurements = measurements
        self.final_states = final_states

    def __repr__(self):
        return ('SimulatorTrialResult(params={!r}, '
                'repetitions={!r}, '
                'measurements={!r}, '
                'final_states={!r})').format(self.params,
                                             self.repetitions,
                                             self.measurements,
                                             self.final_states)

    def __str__(self):
        def bitstring(vals):
            return ''.join('1' if v else '0' for v in vals)

        results_by_rep = (sorted([(key, bitstring(val[i])) for key, val in
                                  self.measurements.items()]) for i in
                          range(self.repetitions))
        str_by_rep = (' '.join(
            '{}={}'.format(key, val) for key, val in result) for result in
            results_by_rep)

        return '\n'.join('repetition {} : {}'.format(i, result) for i, result in
                         enumerate(str_by_rep))

class Simulator:
    """Simulator for Xmon class quantum circuits."""

    def run(
        self,
        circuit: Circuit,
        param_resolver: ParamResolver = ParamResolver({}),
        repetitions: int = 1,
        options: Options = None,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
        extensions: Extensions = None,
    ) -> SimulatorTrialResult:
        """Simulates the entire supplied Circuit.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            options: Options configuring the simulation.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding corresponding to this state.
                Otherwise  if this is a np.ndarray it is the full initial
                state. In this case it must be the correct size, be normalized
                (an L2 norm of 1), and be safely castable to a np.complex64.
            extensions: Extensions that will be applied while trying to
                decompose the circuit's gates into XmonGates. If None, this
                uses the default of xmon_gate_ext.

        Returns:
            Results for this run.
        """
        return self.run_sweep(circuit, [param_resolver], repetitions, options,
                              qubit_order, initial_state,
                              extensions or xmon_gate_ext)[0]

    def run_sweep(
            self,
            program: Union[Circuit, Schedule],
            params: Sweepable = ParamResolver({}),
            repetitions: int = 1,
            options: Options = None,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            initial_state: Union[int, np.ndarray] = 0,
            extensions: Extensions = None
    ) -> List[SimulatorTrialResult]:
        """Simulates the entire supplied Circuit.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            options: Options configuring the simulation.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding corresponding to this state.
                Otherwise if this is a np.ndarray it is the full initial state.
                In this case it must be the correct size, be normalized (an L2
                norm of 1), and be safely castable to a np.complex64.
            extensions: Extensions that will be applied while trying to
                decompose the circuit's gates into XmonGates. If None, this
                uses the default of xmon_gate_ext.

        Returns:
            List of trial results for this run, one for each possible parameter
            resolver.
        """
        circuit = program if isinstance(program,
                                        Circuit) else program.to_circuit()
        param_resolvers = self._to_resolvers(params or ParamResolver({}))

        xmon_circuit, keys = self._to_xmon_circuit(circuit,
                                                   extensions or xmon_gate_ext)
        trial_results = []  # type: List[SimulatorTrialResult]
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        for param_resolver in param_resolvers:
            measurements = {
                k: [] for k in keys}  # type: Dict[str, List[np.ndarray]]
            final_states = []  # type: List[np.ndarray]
            for _ in range(repetitions):
                all_step_results = simulator_iterator(
                    xmon_circuit,
                    options or Options(),
                    qubit_order,
                    initial_state,
                    param_resolver)
                step_result = None
                for step_result in all_step_results:
                    for k, v in step_result.measurements.items():
                        measurements[k].append(np.array(v, dtype=bool))
                if step_result:
                    final_states.append(step_result.state())
                else:
                    # Empty circuit, so final state should be initial state.
                    num_qubits = len(qubit_order.order_for(circuit.qubits()))
                    final_states.append(
                        xmon_stepper.decode_initial_state(initial_state,
                                                          num_qubits))
            trial_results.append(SimulatorTrialResult(
                param_resolver,
                repetitions,
                measurements={k: np.array(v) for k, v in measurements.items()},
                final_states=final_states))
        return trial_results

    def _to_resolvers(self, sweepable: Sweepable) -> List[ParamResolver]:
        if isinstance(sweepable, ParamResolver):
            return [sweepable]
        elif isinstance(sweepable, Sweep):
            return list(sweepable)
        elif isinstance(sweepable, Iterable):
            iterable = cast(Iterable, sweepable)
            return list(iterable) if isinstance(next(iter(iterable)),
                                                ParamResolver) else sum(
                [list(s) for s in iterable], [])
        raise TypeError('Unexpected Sweepable type')

    def moment_steps(
            self,
            program: Circuit,
            options: 'Options' = None,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            initial_state: Union[int, np.ndarray]=0,
            param_resolver: ParamResolver = None,
            extensions: Extensions = None) -> Iterator['StepResult']:
        """Returns an iterator of XmonStepResults for each moment simulated.

        Args:
            program: The Circuit to simulate.
            options: Options configuring the simulation.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding corresponding to this state.
                Otherwise if this is a np.ndarray it is the full initial state.
                In this case it must be the correct size, be normalized (an L2
                norm of 1), and be safely castable to a np.complex64.
            param_resolver: A ParamResolver for determining values of
                Symbols.
            extensions: Extensions that will be applied while trying to
                decompose the circuit's gates into XmonGates. If None, this
                uses the default of xmon_gate_ext.

        Returns:
            SimulatorIterator that steps through the simulation, simulating
            each moment and returning a StepResult for each moment.
        """
        param_resolver = param_resolver or ParamResolver({})
        xmon_circuit, _ = self._to_xmon_circuit(program,
                                                extensions or xmon_gate_ext)
        return simulator_iterator(xmon_circuit,
                                  options or Options(),
                                  qubit_order,
                                  initial_state,
                                  param_resolver)

    def _to_xmon_circuit(self, circuit: Circuit,
                         extensions: Extensions = None
                         ) -> Tuple[Circuit, Set[str]]:
        # TODO: Use one optimization pass.
        xmon_circuit = Circuit(circuit.moments)
        ConvertToXmonGates(extensions).optimize_circuit(xmon_circuit)
        DropEmptyMoments().optimize_circuit(xmon_circuit)
        keys = find_measurement_keys(xmon_circuit)
        return xmon_circuit, keys


def simulator_iterator(
        circuit: Circuit,
        options: 'Options' = Options(),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray]=0,
        param_resolver: ParamResolver = ParamResolver({}),
) -> Iterator['StepResult']:
    """Iterator over TrialResults from Moments of a Circuit.

    This should rarely be instantiated directly, instead prefer to create an
    Simulator and use methods on that object to get an iterator.

    Args:
        circuit: The circuit to simulate; must contain xmon gates only.
        options: Options configuring the simulation.
        qubit_order: Determines the canonical ordering of the qubits used to
            define the order of amplitudes in the wave function.
        initial_state: If this is an int, the state is set to the computational
            basis state corresponding corresponding to the integer. Note that
            the low bit of the integer corresponds to the value of the first
            qubit as determined by the basis argument.

            If this is a np.ndarray it is the full initial state.
            In this case it must be the correct size, be normalized (an L2
            norm of 1), and be safely castable to a np.complex64.
        param_resolver: A ParamResolver for determining values ofs
            Symbols.

    Yields:
        StepResults from simulating a Moment of the Circuit.

    Raises:
        TypeError: if the circuit contains gates that are not XmonGates or
            composite gates made of XmonGates.
    """
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
        circuit.qubits())
    qubit_map = {q: i for i, q in enumerate(reversed(qubits))}
    if isinstance(initial_state, np.ndarray):
        initial_state = initial_state.astype(dtype=np.complex64,
                                             casting='safe')

    with xmon_stepper.Stepper(
            num_qubits=len(qubits),
            num_prefix_qubits=options.num_prefix_qubits,
            initial_state=initial_state,
            min_qubits_before_shard=options.min_qubits_before_shard
    ) as stepper:
        for moment in circuit.moments:
            measurements = defaultdict(list)  # type: Dict[str, List[bool]]
            phase_map = {}  # type: Dict[Tuple[int, ...], float]
            for op in moment.operations:
                gate = op.gate
                if isinstance(gate, xmon_gates.ExpZGate):
                    index = qubit_map[op.qubits[0]]
                    phase_map[(index,)] = param_resolver.value_of(
                        gate.half_turns)
                elif isinstance(gate, xmon_gates.Exp11Gate):
                    index0 = qubit_map[op.qubits[0]]
                    index1 = qubit_map[op.qubits[1]]
                    phase_map[(index0, index1)] = (
                        param_resolver.value_of(gate.half_turns))
                elif isinstance(gate, xmon_gates.ExpWGate):
                    index = qubit_map[op.qubits[0]]
                    stepper.simulate_w(
                        index=index,
                        half_turns=param_resolver.value_of(gate.half_turns),
                        axis_half_turns=param_resolver.value_of(
                            gate.axis_half_turns))
                elif isinstance(gate, xmon_gates.XmonMeasurementGate):
                    invert_mask = gate.invert_mask or len(op.qubits) * (False,)
                    for qubit, invert in zip(op.qubits, invert_mask):
                        index = qubit_map[qubit]
                        result = stepper.simulate_measurement(index)
                        if invert:
                            result = not result
                        measurements[gate.key].append(result)
                else:
                    raise TypeError('{!r} is not supported by the '
                                    'xmon simulator.'.format(gate))
            stepper.simulate_phases(phase_map)
            yield StepResult(stepper, qubit_map, measurements)


def find_measurement_keys(circuit: Circuit) -> Set[str]:
    keys = set()  # type: Set[str]
    for moment in circuit.moments:
        for op in moment.operations:
            if isinstance(op.gate, xmon_gates.XmonMeasurementGate):
                key = op.gate.key
                if key in keys:
                    raise ValueError('Repeated Measurement key {}'.format(key))
                keys.add(key)
    return keys


class StepResult:
    """Results of a step of the simulator.

    Attributes:
        qubit_map: A map from the Qubits in the Circuit to the the index
            of this qubit for a canonical ordering. This canonical ordering is
            used to define the state (see the state() method).
        measurements: A dictionary from measurement gate key to measurement
            results, ordered by the qubits that the measurement operates on.
    """

    def __init__(
            self,
            stepper: xmon_stepper.Stepper,
            qubit_map: Dict,
            measurements: Dict[str, List[bool]]) -> None:
        self.qubit_map = qubit_map or {}
        self.measurements = measurements or defaultdict(list)
        self._stepper = stepper

    def state(self) -> np.ndarray:
        """Return the state (wave function) at this point in the computation.

        The state is returned in the computational basis with these basis
        states defined by the qubit_map. In particular the value in the
        qubit_map is the index of the qubit, and these are translated into
        binary vectors where the last qubit is the 1s bit of the index, the
        second-to-last is the 2s bit of the index, and so forth (i.e. big
        endian ordering).

        Example:
             qubit_map: {QubitA: 0, QubitB: 1, QubitC: 2}
             Then the returned vector will have indices mapped to qubit basis
             states like the following table
               |   | QubitA | QubitB | QubitC |
               +---+--------+--------+--------+
               | 0 |   0    |   0    |   0    |
               | 1 |   0    |   0    |   1    |
               | 2 |   0    |   1    |   0    |
               | 3 |   0    |   1    |   1    |
               | 4 |   1    |   0    |   0    |
               | 5 |   1    |   0    |   1    |
               | 6 |   1    |   1    |   0    |
               | 7 |   1    |   1    |   1    |
               +---+--------+--------+--------+
        """
        return self._stepper.current_state

    def set_state(self, state: Union[int, np.ndarray]):
        """Updates the state of the simulator to the given new state.

        Args:
            state: If this is an int, then this is the state to reset
            the stepper to, expressed as an integer of the computational basis.
            Integer to bitwise indices is little endian. Otherwise if this is
            a np.ndarray this must be the correct size and have dtype of
            np.complex64.

        Raises:
            ValueError if the state is incorrectly sized or not of the correct
            dtype.
        """
        self._stepper.reset_state(state)
