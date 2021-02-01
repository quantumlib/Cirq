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
    cast,
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
    SparseSimulatorStep,
    StepResult,
)
from cirq.study import ParamResolver
from cirq.value import RANDOM_STATE_OR_SEED_LIKE, parse_random_state

from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationRequest,
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
    """

    def __init__(
        self,
        simulator: Simulator,
        *,
        drift_generator: ParametersDriftGenerator,
        gates_translator: Callable[[Gate], Optional[FSimGate]] = try_convert_sqrt_iswap_to_fsim,
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
        self._gates_translator = gates_translator
        self._drifted_parameters: Dict[Tuple[Qid, Qid, FSimGate], PhasedFSimCharacterization] = {}

    @staticmethod
    def create_with_ideal_sqrt_iswap(
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
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(
                gate.phi, 0.0
            ), f'Expected ISWAP ** -0.5 like gate, got {gate}'
            return PhasedFSimCharacterization(
                theta=np.pi / 4, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0
            )

        if simulator is None:
            simulator = Simulator()

        return PhasedFSimEngineSimulator(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @staticmethod
    def create_with_random_gaussian_sqrt_iswap(
        mean: PhasedFSimCharacterization,
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
        weh used multiple times within a circuit.

        Attributes:
            mean: The mean value for each unitary angle. All parameters must be provided.
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.
            sigma: The standard deviation for each unitary angle. When some sigma parameter is None,
                the mean value will be always used for that parameter.

        Returns:
            New PhasedFSimEngineSimulator instance.
        """

        def sample_gate(_1: Qid, _2: Qid, gate: FSimGate) -> PhasedFSimCharacterization:
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(
                gate.phi, 0.0
            ), f'Expected ISWAP ** -0.5 like gate, got {gate}'

            def sample_value(
                gaussian_mean: Optional[float], gaussian_sigma: Optional[float]
            ) -> float:
                assert gaussian_mean is not None
                if gaussian_sigma is None:
                    return gaussian_mean
                return rand.normal(gaussian_mean, gaussian_sigma)

            return PhasedFSimCharacterization(
                theta=sample_value(mean.theta, sigma.theta),
                zeta=sample_value(mean.zeta, sigma.zeta),
                chi=sample_value(mean.chi, sigma.chi),
                gamma=sample_value(mean.gamma, sigma.gamma),
                phi=sample_value(mean.phi, sigma.phi),
            )

        if mean.any_none():
            raise ValueError(f'All mean values must be provided, got mean of {mean}')

        rand = parse_random_state(random_or_seed)

        if simulator is None:
            simulator = Simulator()

        return PhasedFSimEngineSimulator(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @staticmethod
    def create_from_dictionary_sqrt_iswap(
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
                of qubits are is specified then the matching parameter of FSimGate(theta=π/4, phi=0)
                gate will be used. When not set and this situation occurs, ValueError is thrown
                during simulation.

        Returns:
            New PhasedFSimEngineSimulator instance.
        """

        def sample_gate(a: Qid, b: Qid, gate: FSimGate) -> PhasedFSimCharacterization:
            assert np.isclose(gate.theta, np.pi / 4) and np.isclose(
                gate.phi, 0.0
            ), f'Expected ISWAP ** -0.5 like gate, got {gate}'

            pair = (a, b) if a < b else (b, a)

            if pair in parameters:
                pair_parameters = parameters[pair]
                if not isinstance(pair_parameters, PhasedFSimCharacterization):
                    pair_parameters = PhasedFSimCharacterization(**pair_parameters)

                if pair_parameters.any_none():
                    if not ideal_when_missing_parameter:
                        raise ValueError(
                            f'Missing parameter value for pair {pair}, '
                            f'parameters={pair_parameters}'
                        )
                    pair_parameters = pair_parameters.merge_with(SQRT_ISWAP_PARAMETERS)
            elif ideal_when_missing_gate:
                pair_parameters = SQRT_ISWAP_PARAMETERS
            else:
                raise ValueError(f'Missing parameters for pair {pair}')

            return pair_parameters

        for a, b in parameters:
            if a > b:
                raise ValueError(
                    f'All qubit pairs must be given in canonical order where the first qubit is '
                    f'less than the second, got {a} > {b}'
                )

        if simulator is None:
            simulator = Simulator()

        return PhasedFSimEngineSimulator(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @staticmethod
    def create_from_characterizations_sqrt_iswap(
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
                of qubits are is specified then the matching parameter of FSimGate(theta=π/4, phi=0)
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

        return PhasedFSimEngineSimulator.create_from_dictionary_sqrt_iswap(
            parameters,
            simulator=simulator,
            ideal_when_missing_gate=ideal_when_missing_gate,
            ideal_when_missing_parameter=ideal_when_missing_parameter,
        )

    def final_state_vector(self, program: Circuit) -> np.array:
        result = self.simulate(program)
        return cast(SparseSimulatorStep, result).state_vector()

    def get_calibrations(
        self, requests: Sequence[PhasedFSimCalibrationRequest]
    ) -> List[PhasedFSimCalibrationResult]:
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

            translated_gate = self._gates_translator(request.gate)
            if translated_gate is None:
                raise ValueError(f'Calibration request contains unsupported gate {request.gate}')

            parameters = {}
            for a, b in request.pairs:
                drifted = self._create_gate(a, b, translated_gate)
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

    def _run(
        self, circuit: Circuit, param_resolver: ParamResolver, repetitions: int
    ) -> Dict[str, np.ndarray]:
        converted = self._convert_to_circuit_with_drift(circuit)
        return self._simulator._run(converted, param_resolver, repetitions)

    def _base_iterator(
        self,
        circuit: Circuit,
        qubit_order: QubitOrderOrList,
        initial_state: Any,
    ) -> Iterator[StepResult]:
        converted = self._convert_to_circuit_with_drift(circuit)
        return self._simulator._base_iterator(converted, qubit_order, initial_state)

    def _convert_to_circuit_with_drift(self, circuit: Circuit) -> Circuit:
        copied = Circuit(circuit)
        converter = self._PhasedFSimConverter(self)
        converter.optimize_circuit(copied)
        return copied

    def _create_gate(self, a: Qid, b: Qid, gate: FSimGate) -> PhasedFSimGate:
        if (a, b, gate) in self._drifted_parameters:
            parameters = self._drifted_parameters[(a, b, gate)]
        elif (b, a, gate) in self._drifted_parameters:
            parameters = self._drifted_parameters[(b, a, gate)].parameters_for_qubits_swapped()
        else:
            parameters = self._drift_generator(a, b, gate)
            self._drifted_parameters[(a, b, gate)] = parameters
        return PhasedFSimGate(**parameters.asdict())

    class _PhasedFSimConverter(PointOptimizer):
        def __init__(self, outer: 'PhasedFSimEngineSimulator') -> None:
            super().__init__()
            self._outer = outer

        def optimization_at(
            self, circuit: Circuit, index: int, op: Operation
        ) -> Optional[PointOptimizationSummary]:

            if isinstance(op.gate, (MeasurementGate, SingleQubitGate, WaitGate)):
                new_op = op
            else:
                if op.gate is None:
                    raise IncompatibleMomentError(f'Operation {op} has a missing gate')
                translated_gate = self._outer._gates_translator(op.gate)
                if translated_gate is None:
                    raise IncompatibleMomentError(
                        f'Moment contains non-single qubit operation ' f'{op} with unsupported gate'
                    )
                a, b = op.qubits
                new_op = self._outer._create_gate(a, b, translated_gate).on(a, b)

            return PointOptimizationSummary(
                clear_span=1, clear_qubits=op.qubits, new_operations=new_op
            )
