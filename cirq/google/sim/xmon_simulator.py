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

"""XmonSimulator for the Google's Xmon class quantum computers.

The simulator can be used to run all of a Circuit or to step through the
simulation Moment by Moment. The simulator requires that all gates used in
the circuit are either an XmonGate or are CompositionOperations or have a
known unitary which can be decomposed into XmonGates. Measurement gates
must all have unique string keys.

A simple example:
    circuit = Circuit([Moment([X(q1), X(q2)]), Moment([CZ(q1, q2)])])
    sim = XmonSimulator()
    results = sim.run(circuit)

Note that there are two types of methods for the simulator.  "Run" methods
mimic what the quantum hardware provides, and, for example, do not give
access to the wave function.  "Simulate" methods give access to the wave
function, i.e. one can retrieve the final wave function from the simulation
via.
    final_state = sim.simulate(circuit).final_state
"""
import re
import math
import itertools
import collections
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
from cirq.ops import raw_types
from cirq.schedules import Schedule
from cirq.study import ParamResolver, Sweep, Sweepable, TrialResult


def pretty_state(state, decimals=2):
    """
    Returns the wavefunction as a string in Dirac notation.

    For example:
    state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex64)
    print(pretty_state(result.final_state)) -> 0.71|0⟩ + 0.71|1⟩
    """

    perm_list = ["".join(seq) for seq in itertools.product(
        "01", repeat=int(len(state)).bit_length()-1)]
    wvf_string = ""

    for x in range(len(perm_list)):

        rounded_elem = round(state[x].real, decimals) + \
            round(state[x].imag, decimals)

        if rounded_elem != 0:
            wvf_string += str(rounded_elem) + \
                "|{}⟩ + ".format(perm_list[x])

    wvf_string = re.split(r'(\s+)', wvf_string)[:-4]
    wvf_string = "".join(wvf_string)

    return wvf_string


class XmonOptions:
    """XmonOptions for the XmonSimulator.

    Attributes:
        num_prefix_qubits: Sharding of the wave function is performed over 2
            raised to this value number of qubits.
        min_qubits_before_shard: Sharding will be done only for this number
            of qubits or more. The default is 18.
        use_processes: Whether or not to use processes instead of threads.
            Processes can improve the performance slightly (varies by machine
            but on the order of 10 percent faster).  However this varies
            significantly by architecture, and processes should not be used
            for interactive use on Windows.
    """

    def __init__(self,
                 num_shards: int=None,
                 min_qubits_before_shard: int=18,
                 use_processes: bool=False) -> None:
        """XmonSimulator options constructor.

        Args:
            num_shards: sharding will be done for the greatest value of a
                power of two less than this value. If None, the default will
                be used which is the smallest power of two less than or equal
                to the number of CPUs.
            min_qubits_before_shard: Sharding will be done only for this number
                of qubits or more. The default is 18.
            use_processes: Whether or not to use processes instead of threads.
                Processes can improve the performance slightly (varies by
                machine but on the order of 10 percent faster).  However this
                varies significantly by architecture, and processes should not
                be used for interactive python use on Windows.
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
        self.use_processes = use_processes


class XmonSimulateTrialResult:
    """Results of a simulation of the XmonSimulator.

    Unlike TrialResult these results contain the final state (wave function)
    of the system.

    Attributes:
        params: A ParamResolver of settings used for this result.
        measurements: A dictionary from measurement gate key to measurement
            results. Measurement results are a numpy ndarray of actual boolean
            measurement results (ordered by the qubits acted on by the
            measurement gate.)
        final_state: The final state (wave function) of the system after the
            trial finishes.
    """

    def __init__(self,
                 params: ParamResolver,
                 measurements: Dict[str, np.ndarray],
                 final_state: np.ndarray) -> None:
        self.params = params
        self.measurements = measurements
        self.final_state = final_state

    def __repr__(self):
        return ('XmonSimulateTrialResult(params={!r}, '
                'measurements={!r}, '
                'final_state={!r})').format(self.params,
                                            self.measurements,
                                            self.final_state)

    def __str__(self):
        def bitstring(vals):
            return ''.join('1' if v else '0' for v in vals)

        results = sorted(
            [(key, bitstring(val)) for key, val in self.measurements.items()])
        return ' '.join(
            ['{}={}'.format(key, val) for key, val in results])

    def pretty_state(self, decimals=2):
        return pretty_state(self.final_state, decimals)


class XmonSimulator:
    """XmonSimulator for Xmon class quantum circuits.

    This simulator has different methods for different types of simulations.
    For simulations that mimic the quantum hardware, the run methods are
    provided:
        run
        run_sweep
    These methods do not return or give access to the full wave function.

    To get access to the wave function during a simulation, including being
    able to set the wave function, the simulate methods are provided:
        simulate
        simulate_sweep
        simulate_moment_steps (for stepping through a circuit moment by moment)
    """

    def __init__(self, options: XmonOptions = None) -> None:
        """Construct a XmonSimulator.

        Args:
            options: XmonOptions configuring the simulation.
        """
        self.options = options or XmonOptions()

    def run(
        self,
        circuit: Circuit,
        param_resolver: ParamResolver = ParamResolver({}),
        repetitions: int = 1,
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        extensions: Extensions = None,
    ) -> TrialResult:
        """Runs the entire supplied Circuit, mimicking the quantum hardware.

        If one wants access to the wave function (both setting and getting),
        the "simulate" methods should be used.

        The initial state of the  run methods is the all zeros state in the
        computational basis.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            extensions: Extensions that will be applied while trying to
                decompose the circuit's gates into XmonGates. If None, this
                uses the default of xmon_gate_ext.

        Returns:
            TrialResult for a run.
        """
        return self.run_sweep(circuit, [param_resolver], repetitions,
                              qubit_order, extensions or xmon_gate_ext)[0]

    def run_sweep(
            self,
            program: Union[Circuit, Schedule],
            params: Sweepable = ParamResolver({}),
            repetitions: int = 1,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            extensions: Extensions = None
    ) -> List[TrialResult]:
        """Runs the entire supplied Circuit, mimicking the quantum hardware.

        If one wants access to the wave function (both setting and getting),
        the "simulate" methods should be used.

        The initial state of the  run methods is the all zeros state in the
        computational basis.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            extensions: Extensions that will be applied while trying to
                decompose the circuit's gates into XmonGates. If None, this
                uses the default of xmon_gate_ext.

        Returns:
            TrialResult list for this run; one for each possible parameter
            resolver.
        """
        circuit = (
            program if isinstance(program, Circuit) else program.to_circuit())
        param_resolvers = self._to_resolvers(params or ParamResolver({}))

        trial_results = []  # type: List[TrialResult]
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        for param_resolver in param_resolvers:
            xmon_circuit, keys = self._to_xmon_circuit(
                circuit,
                param_resolver,
                extensions or xmon_gate_ext)
            if xmon_circuit.are_all_measurements_terminal():
                measurements = self._run_sweep_sample(xmon_circuit, repetitions,
                                                      qubit_order)
            else:
                measurements = self._run_sweep_repeat(keys, xmon_circuit,
                                                      repetitions, qubit_order)
            trial_results.append(TrialResult(
                params=param_resolver,
                repetitions=repetitions,
                measurements={k: np.array(v) for k, v in measurements.items()}
            ))
        return trial_results

    def _run_sweep_repeat(self, keys, circuit, repetitions, qubit_order):
        measurements = {
            k: [] for k in keys}  # type: Dict[str, List[np.ndarray]]
        for _ in range(repetitions):
            all_step_results = _simulator_iterator(
                circuit,
                self.options,
                qubit_order,
                initial_state=0)
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k].append(np.array(v, dtype=bool))
        return measurements

    def _run_sweep_sample(self, circuit, repetitions, qubit_order):
        all_step_results = _simulator_iterator(
            circuit,
            self.options,
            qubit_order,
            initial_state=0,
            perform_measurements=False)
        step_result = None
        for step_result in all_step_results:
            pass
        return _sample_measurements(circuit,
                                    step_result,
                                    repetitions)

    def simulate(
        self,
        circuit: Circuit,
        param_resolver: ParamResolver = ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
        extensions: Extensions = None,
    ) -> XmonSimulateTrialResult:
        """Simulates the entire supplied Circuit.

        This method returns the final wave function.

        Args:
            circuit: The circuit to simulate.
            param_resolver: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state.
                Otherwise  if this is a np.ndarray it is the full initial
                state. In this case it must be the correct size, be normalized
                (an L2 norm of 1), and be safely castable to a np.complex64.
            extensions: Extensions that will be applied while trying to
                decompose the circuit's gates into XmonGates. If None, this
                uses the default of xmon_gate_ext.

        Returns:
            XmonSimulateTrialResults for the simulation. Includes the final
            wave function.
        """
        return self.simulate_sweep(circuit, [param_resolver], qubit_order,
                                   initial_state,
                                   extensions or xmon_gate_ext)[0]

    def simulate_sweep(
        self,
        program: Union[Circuit, Schedule],
        params: Sweepable = ParamResolver({}),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray] = 0,
        extensions: Extensions = None
    ) -> List[XmonSimulateTrialResult]:
        """Simulates the entire supplied Circuit.

        Args:
            program: The circuit or schedule to simulate.
            params: Parameters to run with the program.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state.
                Otherwise if this is a np.ndarray it is the full initial state.
                In this case it must be the correct size, be normalized (an L2
                norm of 1), and be safely castable to a np.complex64.
            extensions: Extensions that will be applied while trying to
                decompose the circuit's gates into XmonGates. If None, this
                uses the default of xmon_gate_ext.

        Returns:
            List of XmonSimulatorTrialResults for this run, one for each
            possible parameter resolver.
        """
        circuit = (
            program if isinstance(program, Circuit) else program.to_circuit())
        param_resolvers = self._to_resolvers(params or ParamResolver({}))

        trial_results = []  # type: List[XmonSimulateTrialResult]
        qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
        for param_resolver in param_resolvers:
            xmon_circuit, _ = self._to_xmon_circuit(
                circuit,
                param_resolver,
                extensions or xmon_gate_ext)
            measurements = {}  # type: Dict[str, np.ndarray]
            all_step_results = _simulator_iterator(
                xmon_circuit,
                self.options,
                qubit_order,
                initial_state)
            step_result = None
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k] = np.array(v, dtype=bool)
            if step_result:
                final_state = step_result.state()
            else:
                # Empty circuit, so final state should be initial state.
                num_qubits = len(qubit_order.order_for(circuit.all_qubits()))
                final_state = xmon_stepper.decode_initial_state(initial_state,
                                                                num_qubits)
            trial_results.append(XmonSimulateTrialResult(
                params=param_resolver,
                measurements=measurements,
                final_state=final_state))
        return trial_results

    def simulate_moment_steps(
            self,
            program: Circuit,
            options: 'XmonOptions' = None,
            qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
            initial_state: Union[int, np.ndarray]=0,
            param_resolver: ParamResolver = None,
            extensions: Extensions = None) -> Iterator['XmonStepResult']:
        """Returns an iterator of XmonStepResults for each moment simulated.

        Args:
            program: The Circuit to simulate.
            options: XmonOptions configuring the simulation.
            qubit_order: Determines the canonical ordering of the qubits used to
                define the order of amplitudes in the wave function.
            initial_state: If an int, the state is set to the computational
                basis state corresponding to this state.
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
            each moment and returning a XmonStepResult for each moment.
        """
        param_resolver = param_resolver or ParamResolver({})
        xmon_circuit, _ = self._to_xmon_circuit(program,
                                                param_resolver,
                                                extensions or xmon_gate_ext)
        return _simulator_iterator(xmon_circuit,
                                   options or XmonOptions(),
                                   qubit_order,
                                   initial_state)

    def _to_resolvers(self, sweepable: Sweepable) -> List[ParamResolver]:
        if isinstance(sweepable, ParamResolver):
            return [sweepable]
        elif isinstance(sweepable, Sweep):
            return list(sweepable)
        elif isinstance(sweepable, collections.Iterable):
            iterable = cast(collections.Iterable, sweepable)
            return list(iterable) if isinstance(next(iter(iterable)),
                                                ParamResolver) else sum(
                [list(s) for s in iterable], [])
        raise TypeError('Unexpected Sweepable type')

    def _to_xmon_circuit(self, circuit: Circuit,
                         param_resolver: ParamResolver,
                         extensions: Extensions = None
                         ) -> Tuple[Circuit, Set[str]]:
        converter = ConvertToXmonGates(extensions)
        extensions = converter.extensions

        # TODO: Use one optimization pass.
        xmon_circuit = circuit.with_parameters_resolved_by(
            param_resolver, extensions)
        converter.optimize_circuit(xmon_circuit)
        DropEmptyMoments().optimize_circuit(xmon_circuit)
        keys = find_measurement_keys(xmon_circuit)
        return xmon_circuit, keys


def _simulator_iterator(
        circuit: Circuit,
        options: 'XmonOptions' = XmonOptions(),
        qubit_order: ops.QubitOrderOrList = ops.QubitOrder.DEFAULT,
        initial_state: Union[int, np.ndarray]=0,
        perform_measurements: bool=True,
) -> Iterator['XmonStepResult']:
    """Iterator over XmonStepResult from Moments of a Circuit.

    This should rarely be instantiated directly, instead prefer to create an
    XmonSimulator and use methods on that object to get an iterator.

    Args:
        circuit: The circuit to simulate. Must contain only xmon gates with no
            unresolved parameters.
        options: XmonOptions configuring the simulation.
        qubit_order: Determines the canonical ordering of the qubits used to
            define the order of amplitudes in the wave function.
        initial_state: If this is an int, the state is set to the computational
            basis state corresponding to the integer. Note that
            the low bit of the integer corresponds to the value of the first
            qubit as determined by the basis argument.

            If this is a np.ndarray it is the full initial state.
            In this case it must be the correct size, be normalized (an L2
            norm of 1), and be safely castable to a np.complex64.
        perform_measurements: Whether or not to perform the measurements in
            the circuit. Should only be set to False when optimizing for
            sampling over the measurements.

    Yields:
        XmonStepResults from simulating a Moment of the Circuit.

    Raises:
        TypeError: if the circuit contains gates that are not XmonGates or
            composite gates made of XmonGates.
    """
    qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
        circuit.all_qubits())
    qubit_map = {q: i for i, q in enumerate(reversed(qubits))}
    if isinstance(initial_state, np.ndarray):
        initial_state = initial_state.astype(dtype=np.complex64,
                                             casting='safe')

    with xmon_stepper.Stepper(
            num_qubits=len(qubits),
            num_prefix_qubits=options.num_prefix_qubits,
            initial_state=initial_state,
            min_qubits_before_shard=options.min_qubits_before_shard,
            use_processes=options.use_processes
    ) as stepper:
        for moment in circuit:
            measurements = collections.defaultdict(
                list)  # type: Dict[str, List[bool]]
            phase_map = {}  # type: Dict[Tuple[int, ...], float]
            for op in moment.operations:
                gate = cast(ops.GateOperation, op).gate
                if isinstance(gate, xmon_gates.ExpZGate):
                    index = qubit_map[op.qubits[0]]
                    phase_map[(index,)] = cast(float, gate.half_turns)
                elif isinstance(gate, ops.Rot11Gate):
                    index0 = qubit_map[op.qubits[0]]
                    index1 = qubit_map[op.qubits[1]]
                    phase_map[(index0, index1)] = cast(float, gate.half_turns)
                elif isinstance(gate, xmon_gates.ExpWGate):
                    index = qubit_map[op.qubits[0]]
                    stepper.simulate_w(
                        index=index,
                        half_turns=gate.half_turns,
                        axis_half_turns=gate.axis_half_turns)
                elif isinstance(gate, ops.MeasurementGate):
                    if perform_measurements:
                        invert_mask = (
                            gate.invert_mask or len(op.qubits) * (False,))
                        for qubit, invert in zip(op.qubits, invert_mask):
                            index = qubit_map[qubit]
                            result = stepper.simulate_measurement(index)
                            if invert:
                                result = not result
                            measurements[cast(str, gate.key)].append(result)
                else:
                    raise TypeError('{!r} is not supported by the '
                                    'xmon simulator.'.format(gate))
            stepper.simulate_phases(phase_map)
            yield XmonStepResult(stepper, qubit_map, measurements)


def _sample_measurements(circuit: Circuit,
                         step_result: 'XmonStepResult',
                         repetitions: int) -> Dict[str, List]:
    """Sample from measurements in the given circuit.

    This should only be called if the circuit has only terminal measurements.

    Args:
        circuit: The circuit to sample from.
        step_result: The XmonStepResult from which to sample. This should be
            the step at the end of the circuit. Can be None if no steps were
            taken.
        repetitions: The number of time to sample.

    Returns:
        A dictionary from the measurement keys to the measurement results.
        These results are lists of lists, with the outer list corresponding to
        the repetition, and the inner list corresponding to the qubits as
        ordered in the measurement gate.
    """
    if step_result is None:
        return {}
    bounds = {}
    all_qubits = []  # type: List[raw_types.QubitId]
    current_index = 0
    for _, op, gate in circuit.findall_operations_with_gate_type(
            ops.MeasurementGate):
        key = gate.key
        bounds[key] = (current_index, current_index + len(op.qubits))
        all_qubits.extend(op.qubits)
        current_index += len(op.qubits)
    sample = step_result.sample(all_qubits, repetitions)
    return {k: [x[s:e] for x in sample] for k, (s, e) in bounds.items()}


def find_measurement_keys(circuit: Circuit) -> Set[str]:
    keys = set()  # type: Set[str]
    for _, _, gate in circuit.findall_operations_with_gate_type(
            ops.MeasurementGate):
        key = gate.key
        if key in keys:
            raise ValueError('Repeated Measurement key {}'.format(key))
        keys.add(key)
    return keys


class XmonStepResult:
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
        self.measurements = measurements or collections.defaultdict(list)
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

    def pretty_state(self, decimals=2):
        return pretty_state(self.state(), decimals)

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

    def sample(self, qubits: List[raw_types.QubitId], repetitions: int=1):
        """Samples from the wave function at this point in the computation.

        Note that this does not collapse the wave function.

        Returns:
            Measurement results with True corresponding to the |1> state.
            The outer list is for repetitions, and the inner corresponds to
            measurements ordered by the supplied qubits.
        """
        return self._stepper.sample_measurements(
            indices=[self.qubit_map[q] for q in qubits],
            repetitions=repetitions)
