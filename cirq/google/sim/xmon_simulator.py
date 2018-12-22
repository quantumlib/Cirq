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
the circuit are either native to the xmon architecture (i.e. cause
`cirq.google.is_native_xmon_op` to return true) or else can be decomposed into
such operations (by being composite or having a known unitary). Measurement
gates must all have unique string keys.

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
import math
import collections
from typing import cast, Dict, Iterator, List, Set, Union
from typing import Tuple  # pylint: disable=unused-import

import numpy as np

from cirq import circuits, ops, study, protocols, optimizers
from cirq.sim import simulator
from cirq.google import convert_to_xmon_gates
from cirq.google.sim import xmon_stepper


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


class XmonSimulator(simulator.SimulatesSamples,
                    simulator.SimulatesIntermediateWaveFunction):
    """XmonSimulator for Xmon class quantum circuits.

    This simulator has different methods for different types of simulations.

    For simulations that mimic the quantum hardware, the run methods are
    defined in the SimulatesSamples interface:
        run
        run_sweep
    These methods do not return or give access to the full wave function.

    To get access to the wave function during a simulation, including being
    able to set the wave function, the simulate methods are defined in the
    SimulatesFinalWaveFunction interface:
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

    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int,
    ) -> Dict[str, List[np.ndarray]]:
        """See definition in `cirq.SimulatesSamples`."""
        xmon_circuit, keys = self._to_xmon_circuit(
            circuit,
            param_resolver)
        if xmon_circuit.are_all_measurements_terminal():
            return self._run_sweep_sample(xmon_circuit, repetitions)
        else:
            return self._run_sweep_repeat(keys, xmon_circuit, repetitions)

    def _run_sweep_repeat(self, keys, circuit, repetitions):
        measurements = {k: [] for k in
                        keys}  # type: Dict[str, List[np.ndarray]]
        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                circuit,
                qubit_order=ops.QubitOrder.DEFAULT,
                initial_state=0)
            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    measurements[k].append(np.array(v, dtype=bool))
        return {k: np.array(v) for k, v in measurements.items()}

    def _run_sweep_sample(self, circuit, repetitions):
        all_step_results = self._base_iterator(
            circuit,
            qubit_order=ops.QubitOrder.DEFAULT,
            initial_state=0,
            perform_measurements=False)
        step_result = None
        for step_result in all_step_results:
            pass
        if step_result is None:
            return {}
        measurement_ops = [op for _, op, _ in
                           circuit.findall_operations_with_gate_type(
                                   ops.MeasurementGate)]
        return step_result.sample_measurement_ops(measurement_ops, repetitions)

    def _simulator_iterator(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        qubit_order: ops.QubitOrderOrList,
        initial_state: Union[int, np.ndarray],
        perform_measurements: bool = True,
    ) -> Iterator['XmonStepResult']:
        """See definition in `cirq.SimulatesIntermediateWaveFunction`."""
        param_resolver = param_resolver or study.ParamResolver({})
        xmon_circuit, _ = self._to_xmon_circuit(circuit, param_resolver)
        return self._base_iterator(xmon_circuit,
                                   qubit_order,
                                   initial_state,
                                   perform_measurements)


    def _base_iterator(
        self,
        circuit: circuits.Circuit,
        qubit_order: ops.QubitOrderOrList,
        initial_state: Union[int, np.ndarray],
        perform_measurements: bool=True,
    ) -> Iterator['XmonStepResult']:
        """See _simulator_iterator."""
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
            circuit.all_qubits())
        qubit_map = {q: i for i, q in enumerate(reversed(qubits))}
        if isinstance(initial_state, np.ndarray):
            initial_state = initial_state.astype(dtype=np.complex64,
                                                 casting='safe')
        with xmon_stepper.Stepper(
            num_qubits=len(qubits),
            num_prefix_qubits=self.options.num_prefix_qubits,
            initial_state=initial_state,
            min_qubits_before_shard=self.options.min_qubits_before_shard,
            use_processes=self.options.use_processes
        ) as stepper:
            for moment in circuit:
                measurements = collections.defaultdict(
                    list)  # type: Dict[str, List[bool]]
                phase_map = {}  # type: Dict[Tuple[int, ...], float]
                for op in moment.operations:
                    gate = cast(ops.GateOperation, op).gate
                    if isinstance(gate, ops.ZPowGate):
                        index = qubit_map[op.qubits[0]]
                        phase_map[(index,)] = cast(float, gate.exponent)
                    elif isinstance(gate, ops.CZPowGate):
                        index0 = qubit_map[op.qubits[0]]
                        index1 = qubit_map[op.qubits[1]]
                        phase_map[(index0, index1)] = cast(float,
                                                           gate.exponent)
                    elif isinstance(gate, ops.XPowGate):
                        index = qubit_map[op.qubits[0]]
                        stepper.simulate_w(
                            index=index,
                            half_turns=gate.exponent,
                            axis_half_turns=0)
                    elif isinstance(gate, ops.YPowGate):
                        index = qubit_map[op.qubits[0]]
                        stepper.simulate_w(
                            index=index,
                            half_turns=gate.exponent,
                            axis_half_turns=0.5)
                    elif isinstance(gate, ops.PhasedXPowGate):
                        index = qubit_map[op.qubits[0]]
                        stepper.simulate_w(
                            index=index,
                            half_turns=gate.exponent,
                            axis_half_turns=gate.phase_exponent)
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
                        # coverage: ignore
                        raise TypeError('{!r} is not supported by the '
                                        'xmon simulator.'.format(gate))
                stepper.simulate_phases(phase_map)
                yield XmonStepResult(stepper, qubit_map, measurements)

    def _to_xmon_circuit(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver
    ) -> Tuple[circuits.Circuit, Set[str]]:
        # TODO: Use one optimization pass.
        xmon_circuit = protocols.resolve_parameters(circuit, param_resolver)
        convert_to_xmon_gates.ConvertToXmonGates().optimize_circuit(
            xmon_circuit)
        optimizers.DropEmptyMoments().optimize_circuit(xmon_circuit)
        keys = find_measurement_keys(xmon_circuit)
        return xmon_circuit, keys


def find_measurement_keys(circuit: circuits.Circuit) -> Set[str]:
    keys = set()  # type: Set[str]
    for _, _, gate in circuit.findall_operations_with_gate_type(
            ops.MeasurementGate):
        key = gate.key
        if key in keys:
            raise ValueError('Repeated Measurement key {}'.format(key))
        keys.add(key)
    return keys


class XmonStepResult(simulator.StepResult):
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
            measurements: Dict[str, np.ndarray]) -> None:
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

    def sample(self, qubits: List[ops.QubitId], repetitions: int=1):
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
