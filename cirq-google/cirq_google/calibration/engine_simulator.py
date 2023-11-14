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
from typing import (
    Any,
    Callable,
    cast,
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

import cirq
from cirq import value
from cirq_google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationRequest,
    PhaseCalibratedFSimGate,
    IncompatibleMomentError,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization,
    SQRT_ISWAP_INV_PARAMETERS,
    try_convert_gate_to_fsim,
    try_convert_sqrt_iswap_to_fsim,
)

ParametersDriftGenerator = Callable[[cirq.Qid, cirq.Qid, cirq.FSimGate], PhasedFSimCharacterization]
PhasedFsimDictParameters = Dict[
    Tuple[cirq.Qid, cirq.Qid], Union[Dict[str, float], PhasedFSimCharacterization]
]


class PhasedFSimEngineSimulator(cirq.SimulatesIntermediateStateVector[cirq.SparseSimulatorStep]):
    """Wrapper on top of cirq.Simulator that allows to simulate calibration requests.

    This simulator introduces get_calibrations which allows to simulate
    cirq_google.run_characterizations requests. The returned calibration results represent the
    internal state of a simulator. Circuits which are run on this simulator are modified to account
    for the changes in the unitary parameters as described by the calibration results.

    Attributes:
        gates_translator: Function that translates a gate to a supported FSimGate which will undergo
        characterization.
    """

    def __init__(
        self,
        simulator: cirq.Simulator,
        *,
        drift_generator: ParametersDriftGenerator,
        gates_translator: Callable[
            [cirq.Gate], Optional[PhaseCalibratedFSimGate]
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
        super().__init__()
        self._simulator = simulator
        self._drift_generator = drift_generator
        self._drifted_parameters: Dict[
            Tuple[cirq.Qid, cirq.Qid, cirq.FSimGate], PhasedFSimCharacterization
        ] = {}
        self.gates_translator = gates_translator

    @classmethod
    def create_with_ideal_sqrt_iswap(
        cls, *, simulator: Optional[cirq.Simulator] = None
    ) -> 'PhasedFSimEngineSimulator':
        """Creates a PhasedFSimEngineSimulator that simulates ideal FSimGate(theta=π/4, phi=0).

        Attributes:
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.

        Returns:
            New PhasedFSimEngineSimulator instance.
        """

        def sample_gate(
            _1: cirq.Qid, _2: cirq.Qid, gate: cirq.FSimGate
        ) -> PhasedFSimCharacterization:
            _assert_inv_sqrt_iswap_like(gate)
            return PhasedFSimCharacterization(
                theta=np.pi / 4, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0
            )

        if simulator is None:
            simulator = cirq.Simulator()

        return cls(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @classmethod
    def create_with_random_gaussian_sqrt_iswap(
        cls,
        mean: PhasedFSimCharacterization = SQRT_ISWAP_INV_PARAMETERS,
        *,
        simulator: Optional[cirq.Simulator] = None,
        sigma: PhasedFSimCharacterization = PhasedFSimCharacterization(
            theta=0.02, zeta=0.05, chi=0.05, gamma=0.05, phi=0.02
        ),
        random_or_seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
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

        Raises:
            ValueError: If not all mean values were supplied.
        """

        if mean.any_none():
            raise ValueError(f'All mean values must be provided, got mean of {mean}')

        rand = value.parse_random_state(random_or_seed)

        def sample_value(gaussian_mean: Optional[float], gaussian_sigma: Optional[float]) -> float:
            assert gaussian_mean is not None
            if gaussian_sigma is None:
                return gaussian_mean
            return rand.normal(gaussian_mean, gaussian_sigma)

        def sample_gate(
            _1: cirq.Qid, _2: cirq.Qid, gate: cirq.FSimGate
        ) -> PhasedFSimCharacterization:
            _assert_inv_sqrt_iswap_like(gate)

            return PhasedFSimCharacterization(
                theta=sample_value(mean.theta, sigma.theta),
                zeta=sample_value(mean.zeta, sigma.zeta),
                chi=sample_value(mean.chi, sigma.chi),
                gamma=sample_value(mean.gamma, sigma.gamma),
                phi=sample_value(mean.phi, sigma.phi),
            )

        if simulator is None:
            simulator = cirq.Simulator()

        return cls(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @classmethod
    def create_from_dictionary_sqrt_iswap(
        cls,
        parameters: PhasedFsimDictParameters,
        *,
        simulator: Optional[cirq.Simulator] = None,
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

        Raises:
            ValueError: If missing parameters for the given pair of qubits.
        """

        def sample_gate(
            a: cirq.Qid, b: cirq.Qid, gate: cirq.FSimGate
        ) -> PhasedFSimCharacterization:
            _assert_inv_sqrt_iswap_like(gate)

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
                pair_parameters = SQRT_ISWAP_INV_PARAMETERS
            else:
                raise ValueError(f'Missing parameters for pair {(a, b)}')

            if pair_parameters.any_none():
                if not ideal_when_missing_parameter:
                    raise ValueError(
                        f'Missing parameter value for pair {(a, b)}, '
                        f'parameters={pair_parameters}'
                    )
                pair_parameters = pair_parameters.merge_with(SQRT_ISWAP_INV_PARAMETERS)

            return pair_parameters

        for a, b in parameters:
            if a > b:
                raise ValueError(
                    f'All qubit pairs must be given in canonical order where the first qubit is '
                    f'less than the second, got {a} > {b}'
                )

        if simulator is None:
            simulator = cirq.Simulator()

        return cls(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_sqrt_iswap_to_fsim
        )

    @classmethod
    def create_from_dictionary(
        cls,
        parameters: Dict[
            Tuple[cirq.Qid, cirq.Qid], Dict[cirq.FSimGate, Union[PhasedFSimCharacterization, Dict]]
        ],
        *,
        simulator: Optional[cirq.Simulator] = None,
    ) -> 'PhasedFSimEngineSimulator':
        """Creates PhasedFSimEngineSimulator with fixed drifts.

        Args:
            parameters: maps every pair of qubits and engine gate on that pair to a
                characterization for that gate.
            simulator: Simulator object to use. When None, a new instance of cirq.Simulator() will
                be created.

        Returns:
            New PhasedFSimEngineSimulator instance.

        Raises:
            ValueError: If missing parameters for the given pair of qubits.
        """

        for a, b in parameters.keys():
            if a > b:
                raise ValueError(
                    f'All qubit pairs must be given in canonical order where the first qubit is '
                    f'less than the second, got {a} > {b}'
                )

        def sample_gate(
            a: cirq.Qid, b: cirq.Qid, gate: cirq.FSimGate
        ) -> PhasedFSimCharacterization:
            pair_parameters = None
            swapped = False
            if (a, b) in parameters:
                pair_parameters = parameters[(a, b)].get(gate)
            elif (b, a) in parameters:
                pair_parameters = parameters[(b, a)].get(gate)
                swapped = True

            if pair_parameters is None:
                raise ValueError(f'Missing parameters for value for pair {(a, b)} and gate {gate}.')
            if not isinstance(pair_parameters, PhasedFSimCharacterization):
                pair_parameters = PhasedFSimCharacterization(**pair_parameters)
            if swapped:
                pair_parameters = pair_parameters.parameters_for_qubits_swapped()

            return pair_parameters

        if simulator is None:
            simulator = cirq.Simulator()
        return cls(
            simulator, drift_generator=sample_gate, gates_translator=try_convert_gate_to_fsim
        )

    @classmethod
    def create_from_characterizations_sqrt_iswap(
        cls,
        characterizations: Iterable[PhasedFSimCalibrationResult],
        *,
        simulator: Optional[cirq.Simulator] = None,
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

        Raises:
            ValueError: If the gate was not a gate like `ISWAP ** -0.5` or the pair of qubits it
                acts on appears in multiple different moments.
        """

        parameters: PhasedFsimDictParameters = {}
        for characterization in characterizations:
            gate = characterization.gate
            _assert_inv_sqrt_iswap_like(gate)

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
            simulator = cirq.Simulator()

        return cls.create_from_dictionary_sqrt_iswap(
            parameters,
            simulator=simulator,
            ideal_when_missing_gate=ideal_when_missing_gate,
            ideal_when_missing_parameter=ideal_when_missing_parameter,
        )

    def final_state_vector(self, program: cirq.Circuit) -> np.ndarray:
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

        Raises:
            ValueError: If supplied type of request is not supported or if the request contains
                and unsupported gate.
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
                    theta=cast(float, drifted.theta) if characterize_theta else None,
                    zeta=cast(float, drifted.zeta) if characterize_zeta else None,
                    chi=cast(float, drifted.chi) if characterize_chi else None,
                    gamma=cast(float, drifted.gamma) if characterize_gamma else None,
                    phi=cast(float, drifted.phi) if characterize_phi else None,
                )

            results.append(
                PhasedFSimCalibrationResult(
                    parameters=parameters, gate=request.gate, options=options
                )
            )

        return results

    def create_gate_with_drift(
        self, a: cirq.Qid, b: cirq.Qid, gate_calibration: PhaseCalibratedFSimGate
    ) -> cirq.PhasedFSimGate:
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

    def run_sweep_iter(
        self, program: cirq.AbstractCircuit, params: cirq.Sweepable, repetitions: int = 1
    ) -> Iterator[cirq.Result]:
        converted = _convert_to_circuit_with_drift(self, program)
        yield from self._simulator.run_sweep_iter(converted, params, repetitions)

    def simulate(
        self,
        program: cirq.AbstractCircuit,
        param_resolver: cirq.ParamResolverOrSimilarType = None,
        qubit_order: cirq.QubitOrderOrList = cirq.QubitOrder.DEFAULT,
        initial_state: Any = None,
    ) -> cirq.StateVectorTrialResult:
        converted = _convert_to_circuit_with_drift(self, program)
        return self._simulator.simulate(converted, param_resolver, qubit_order, initial_state)

    def _create_partial_simulation_state(
        self,
        initial_state: Union[int, cirq.StateVectorSimulationState],
        qubits: Sequence[cirq.Qid],
        classical_data: cirq.ClassicalDataStore,
    ) -> cirq.StateVectorSimulationState:
        # Needs an implementation since it's abstract but will never actually be called.
        raise NotImplementedError()

    def _create_step_result(self, sim_state: cirq.SimulationStateBase) -> cirq.SparseSimulatorStep:
        # Needs an implementation since it's abstract but will never actually be called.
        raise NotImplementedError()


def _convert_to_circuit_with_drift(
    simulator: PhasedFSimEngineSimulator, circuit: cirq.AbstractCircuit
) -> cirq.Circuit:
    def map_func(op: cirq.Operation, _) -> cirq.Operation:
        if op.gate is None:
            raise IncompatibleMomentError(f'Operation {op} has a missing gate')

        if (
            isinstance(op.gate, (cirq.MeasurementGate, cirq.WaitGate))
            or cirq.num_qubits(op.gate) == 1
        ):
            return op

        translated = simulator.gates_translator(op.gate)
        if translated is None:
            raise IncompatibleMomentError(
                f'Moment contains non-single qubit operation {op} with unsupported gate'
            )

        a, b = op.qubits
        return simulator.create_gate_with_drift(a, b, translated).on(a, b)

    return cirq.map_operations(circuit, map_func).unfreeze(copy=False)


def _assert_inv_sqrt_iswap_like(gate: cirq.Gate):
    assert isinstance(gate, cirq.FSimGate), f'Expected FSimGate, got {gate}'

    if cirq.is_parameterized(gate):
        raise ValueError("Only unparameterized gates are supported. Gate: {gate}.")
    theta = gate.theta
    phi = gate.phi
    assert isinstance(theta, float) and isinstance(phi, float)
    assert np.isclose(theta, np.pi / 4) and np.isclose(
        phi, 0.0
    ), f'Expected ISWAP ** -0.5 like gate, got {gate}'
