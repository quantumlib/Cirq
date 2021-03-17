from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from cirq.circuits import Circuit, PointOptimizer, PointOptimizationSummary
from cirq.ops import (
    FSimGate,
    Gate,
    MeasurementGate,
    Operation,
    PhasedFSimGate,
    Qid,
    QubitOrderOrList,
    SingleQubitGate,
    WaitGate,
)
from cirq.sim import (
    Simulator,
    SimulatesSamples,
    SimulatesIntermediateStateVector,
    StateVectorStepResult,
)
from cirq.study import ParamResolver
from cirq.value import RANDOM_STATE_OR_SEED_LIKE, parse_random_state

from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationRequest,
    PhaseCalibratedFSimGate,
    IncompatibleMomentError,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization,
    SQRT_ISWAP_PARAMETERS,
    try_convert_sqrt_iswap_to_fsim,
)


ParametersDriftGenerator = Callable[[Qid, Qid, FSimGate], PhasedFSimCharacterization]
PhasedFsimDictParameters = Dict[
    Tuple[Qid, Qid], Union[Dict[str, float], PhasedFSimCharacterization]
]


class PhasedFSimEngineSimulator(SimulatesSamples, SimulatesIntermediateStateVector):
    """Wrapper on top of cirq.Simulator that allows to simulate calibration requests.

    This simulator introduces get_calibrations which allows to simulate
    cirq.google.run_characterizations requests. The returned calibration results represent the
    internal state of a simulator. Circuits which are run on this simulator are modified to account
    for the changes in the unitary parameters as described by the calibration results.

    Attributes:
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
        characterization.
    """

    def __init__(
        self,
        simulator: Simulator,
        *,
        drift_generator: ParametersDriftGenerator,
        gates_translator: Callable[
            [Gate], Optional[PhaseCalibratedFSimGate]
        ] = try_convert_sqrt_iswap_to_fsim,
    ) -> None:
        """Initializes the PhasedFSimEngineSimulator.

        Args:
            simulator: cirq.Simulator that all the simulation requests are delegated to.
            drift_generator: Callable that generates the imperfect parameters for each pair of
                qubits and the gate. They are later used for simulation.
            gates_translator: Function that translates a gate to a supported FSimGate which will
                undergo characterization.
        """
        self._simulator = simulator
        self._drift_generator = drift_generator
        self._drifted_parameters: Dict[Tuple[Qid, Qid, FSimGate], PhasedFSimCharacterization] = {}
        self.gates_translator = gates_translator

    @classmethod
    def create_with_ideal_sqrt_iswap(
        cls,
        *,
        simulator: Optional[Simulator] = None,
    ) -> 'PhasedFSimEngineSimulator':
        """Creates a PhasedFSimEngineSimulator that simulates ideal FSimGate(theta=π/4, phi=0).

        Attributes:
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.

        Returns:
            New PhasedFSimEngineSimulator instance.
        """

        def sample_gate(_1: Qid, _2: Qid, gate: FSimGate) -> PhasedFSimCharacterization:
            assert isinstance(gate, FSimGate), f'Expected FSimGate, got {gate}'
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(
                gate.phi, 0.0
            ), f'Expected ISWAP ** -0.5 like gate, got {gate}'
            return PhasedFSimCharacterization(
                theta=np.pi / 4, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0
            )

        if simulator is None:
            simulator = Simulator()

        return cls(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @classmethod
    def create_with_random_gaussian_sqrt_iswap(
        cls,
        mean: PhasedFSimCharacterization = SQRT_ISWAP_PARAMETERS,
        *,
        simulator: Optional[Simulator] = None,
        sigma: PhasedFSimCharacterization = PhasedFSimCharacterization(
            theta=0.02, zeta=0.05, chi=0.05, gamma=0.05, phi=0.02
        ),
        random_or_seed: RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> 'PhasedFSimEngineSimulator':
        """Creates a PhasedFSimEngineSimulator that introduces a random deviation from the mean.

        The random deviations are described by a Gaussian distribution of a given mean and sigma,
        for each angle respectively.

        Each gate for each pair of qubits retains the sampled values for the entire simulation, even
        when used multiple times within a circuit.

        Attributes:
            mean: The mean value for each unitary angle. All parameters must be provided.
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.
            sigma: The standard deviation for each unitary angle. For sigma parameters that are
                None, the mean value will be used without any sampling.

        Returns:
            New PhasedFSimEngineSimulator instance.
        """

        if mean.any_none():
            raise ValueError(f'All mean values must be provided, got mean of {mean}')

        rand = parse_random_state(random_or_seed)

        def sample_value(gaussian_mean: Optional[float], gaussian_sigma: Optional[float]) -> float:
            assert gaussian_mean is not None
            if gaussian_sigma is None:
                return gaussian_mean
            return rand.normal(gaussian_mean, gaussian_sigma)

        def sample_gate(_1: Qid, _2: Qid, gate: FSimGate) -> PhasedFSimCharacterization:
            assert isinstance(gate, FSimGate), f'Expected FSimGate, got {gate}'
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(
                gate.phi, 0.0
            ), f'Expected ISWAP ** -0.5 like gate, got {gate}'

            return PhasedFSimCharacterization(
                theta=sample_value(mean.theta, sigma.theta),
                zeta=sample_value(mean.zeta, sigma.zeta),
                chi=sample_value(mean.chi, sigma.chi),
                gamma=sample_value(mean.gamma, sigma.gamma),
                phi=sample_value(mean.phi, sigma.phi),
            )

        if simulator is None:
            simulator = Simulator()

        return cls(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @classmethod
    def create_from_dictionary_sqrt_iswap(
        cls,
        parameters: PhasedFsimDictParameters,
        *,
        simulator: Optional[Simulator] = None,
        ideal_when_missing_gate: bool = False,
        ideal_when_missing_parameter: bool = False,
    ) -> 'PhasedFSimEngineSimulator':
        """Creates PhasedFSimEngineSimulator with fixed drifts.

        Args:
            parameters: Parameters to use for each gate. All keys must be stored in canonical order,
                when the first qubit is not greater than the second one.
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.
            ideal_when_missing_gate: When set and parameters for some gate for a given pair of
                qubits are not specified in the parameters dictionary then the
                FSimGate(theta=π/4, phi=0) gate parameters will be used. When not set and this
                situation occurs, ValueError is thrown during simulation.
            ideal_when_missing_parameter: When set and some parameter for some gate for a given pair
                of qubits is specified then the matching parameter of FSimGate(theta=π/4, phi=0)
                gate will be used. When not set and this situation occurs, ValueError is thrown
                during simulation.

        Returns:
            New PhasedFSimEngineSimulator instance.
        """

        def sample_gate(a: Qid, b: Qid, gate: FSimGate) -> PhasedFSimCharacterization:
            assert isinstance(gate, FSimGate), f'Expected FSimGate, got {gate}'
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(
                gate.phi, 0.0
            ), f'Expected ISWAP ** -0.5 like gate, got {gate}'

            if (a, b) in parameters:
                pair_parameters = parameters[(a, b)]
                if not isinstance(pair_parameters, PhasedFSimCharacterization):
                    pair_parameters = PhasedFSimCharacterization(**pair_parameters)
            elif (b, a) in parameters:
                pair_parameters = parameters[(b, a)]
                if not isinstance(pair_parameters, PhasedFSimCharacterization):
                    pair_parameters = PhasedFSimCharacterization(**pair_parameters)
                pair_parameters = pair_parameters.parameters_for_qubits_swapped()
            elif ideal_when_missing_gate:
                pair_parameters = SQRT_ISWAP_PARAMETERS
            else:
                raise ValueError(f'Missing parameters for pair {(a, b)}')

            if pair_parameters.any_none():
                if not ideal_when_missing_parameter:
                    raise ValueError(
                        f'Missing parameter value for pair {(a, b)}, '
                        f'parameters={pair_parameters}'
                    )
                pair_parameters = pair_parameters.merge_with(SQRT_ISWAP_PARAMETERS)

            return pair_parameters

        for a, b in parameters:
            if a > b:
                raise ValueError(
                    f'All qubit pairs must be given in canonical order where the first qubit is '
                    f'less than the second, got {a} > {b}'
                )

        if simulator is None:
            simulator = Simulator()

        return cls(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @classmethod
    def create_from_characterizations_sqrt_iswap(
        cls,
        characterizations: Iterable[PhasedFSimCalibrationResult],
        *,
        simulator: Optional[Simulator] = None,
        ideal_when_missing_gate: bool = False,
        ideal_when_missing_parameter: bool = False,
    ) -> 'PhasedFSimEngineSimulator':
        """Creates PhasedFSimEngineSimulator with fixed drifts from the characterizations results.

        Args:
            characterizations: Characterization results which are source of the parameters for
                each gate.
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.
            ideal_when_missing_gate: When set and parameters for some gate for a given pair of
                qubits are not specified in the parameters dictionary then the
                FSimGate(theta=π/4, phi=0) gate parameters will be used. When not set and this
                situation occurs, ValueError is thrown during simulation.
            ideal_when_missing_parameter: When set and some parameter for some gate for a given pair
                of qubits is specified then the matching parameter of FSimGate(theta=π/4, phi=0)
                gate will be used. When not set and this situation occurs, ValueError is thrown
                during simulation.

        Returns:
            New PhasedFSimEngineSimulator instance.
        """

        parameters: PhasedFsimDictParameters = {}
        for characterization in characterizations:
            gate = characterization.gate
            if (
                not isinstance(gate, FSimGate)
                or not np.isclose(gate.theta, np.pi / 4)
                or not np.isclose(gate.phi, 0.0)
            ):
                raise ValueError(f'Expected ISWAP ** -0.5 like gate, got {gate}')

            for (a, b), pair_parameters in characterization.parameters.items():
                if a > b:
                    a, b = b, a
                    pair_parameters = pair_parameters.parameters_for_qubits_swapped()
                if (a, b) in parameters:
                    raise ValueError(
                        f'Pair ({(a, b)}) appears in multiple moments, multi-moment '
                        f'simulation is not supported.'
                    )
                parameters[(a, b)] = pair_parameters

        if simulator is None:
            simulator = Simulator()

        return cls.create_from_dictionary_sqrt_iswap(
            parameters,
            simulator=simulator,
            ideal_when_missing_gate=ideal_when_missing_gate,
            ideal_when_missing_parameter=ideal_when_missing_parameter,
        )

    def final_state_vector(self, program: Circuit) -> np.array:
        result = self.simulate(program)
        return result.state_vector()

    def get_calibrations(
        self, requests: Sequence[PhasedFSimCalibrationRequest]
    ) -> List[PhasedFSimCalibrationResult]:
        """Retrieves the calibration that matches the requests

        Args:
            requests: Calibration requests to obtain.

        Returns:
            Calibration results that reflect the internal state of simulator.
        """
        results = []
        for request in requests:
            if isinstance(request, FloquetPhasedFSimCalibrationRequest):
                options = request.options
                characterize_theta = options.characterize_theta
                characterize_zeta = options.characterize_zeta
                characterize_chi = options.characterize_chi
                characterize_gamma = options.characterize_gamma
                characterize_phi = options.characterize_phi
            else:
                raise ValueError(f'Unsupported calibration request {request}')

            translated = self.gates_translator(request.gate)
            if translated is None:
                raise ValueError(f'Calibration request contains unsupported gate {request.gate}')

            parameters = {}
            for a, b in request.pairs:
                drifted = self.create_gate_with_drift(a, b, translated)
                parameters[a, b] = PhasedFSimCharacterization(
                    theta=drifted.theta if characterize_theta else None,
                    zeta=drifted.zeta if characterize_zeta else None,
                    chi=drifted.chi if characterize_chi else None,
                    gamma=drifted.gamma if characterize_gamma else None,
                    phi=drifted.phi if characterize_phi else None,
                )

            results.append(
                PhasedFSimCalibrationResult(
                    parameters=parameters, gate=request.gate, options=options
                )
            )

        return results

    def create_gate_with_drift(
        self, a: Qid, b: Qid, gate_calibration: PhaseCalibratedFSimGate
    ) -> PhasedFSimGate:
        """Generates a gate with drift for a given gate.

        Args:
            a: The first qubit.
            b: The second qubit.
            gate_calibration: Reference gate together with a phase information.

        Returns:
            A modified gate that includes the drifts induced by internal state of the simulator.
        """
        gate = gate_calibration.engine_gate
        if (a, b, gate) in self._drifted_parameters:
            parameters = self._drifted_parameters[(a, b, gate)]
        elif (b, a, gate) in self._drifted_parameters:
            parameters = self._drifted_parameters[(b, a, gate)].parameters_for_qubits_swapped()
        else:
            parameters = self._drift_generator(a, b, gate)
            self._drifted_parameters[(a, b, gate)] = parameters

        return gate_calibration.as_characterized_phased_fsim_gate(parameters)

    def _run(
        self, circuit: Circuit, param_resolver: ParamResolver, repetitions: int
    ) -> Dict[str, np.ndarray]:
        converted = _convert_to_circuit_with_drift(self, circuit)
        return self._simulator._run(converted, param_resolver, repetitions)

    def _base_iterator(
        self,
        circuit: Circuit,
        qubit_order: QubitOrderOrList,
        initial_state: Any,
    ) -> Iterator[StateVectorStepResult]:
        converted = _convert_to_circuit_with_drift(self, circuit)
        return self._simulator._base_iterator(converted, qubit_order, initial_state)


class _PhasedFSimConverter(PointOptimizer):
    def __init__(self, simulator: PhasedFSimEngineSimulator) -> None:
        super().__init__()
        self._simulator = simulator

    def optimization_at(
        self, circuit: Circuit, index: int, op: Operation
    ) -> Optional[PointOptimizationSummary]:

        if isinstance(op.gate, (MeasurementGate, SingleQubitGate, WaitGate)):
            new_op = op
        else:
            if op.gate is None:
                raise IncompatibleMomentError(f'Operation {op} has a missing gate')
            translated = self._simulator.gates_translator(op.gate)
            if translated is None:
                raise IncompatibleMomentError(
                    f'Moment contains non-single qubit operation ' f'{op} with unsupported gate'
                )

            a, b = op.qubits
            new_op = self._simulator.create_gate_with_drift(a, b, translated).on(a, b)

        return PointOptimizationSummary(clear_span=1, clear_qubits=op.qubits, new_operations=new_op)


def _convert_to_circuit_with_drift(
    simulator: PhasedFSimEngineSimulator, circuit: Circuit
) -> Circuit:
    circuit_with_drift = Circuit(circuit)
    converter = _PhasedFSimConverter(simulator)
    converter.optimize_circuit(circuit_with_drift)
    return circuit_with_drift
