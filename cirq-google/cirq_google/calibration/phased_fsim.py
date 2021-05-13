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
import abc
import collections
import dataclasses
import functools
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    TYPE_CHECKING,
    Generic,
    Union,
    cast,
)

import numpy as np
import pandas as pd

from cirq.circuits import Circuit
from cirq.devices import GridQubit
from cirq.experiments.xeb_fitting import (
    XEBPhasedFSimCharacterizationOptions,
)
from cirq.ops import (
    FSimGate,
    Gate,
    ISwapPowGate,
    Moment,
    Operation,
    PhasedFSimGate,
    PhasedISwapPowGate,
    Qid,
    TwoQubitGate,
    rz,
)
from cirq_google.api import v2
from cirq_google.engine import Calibration, CalibrationLayer, CalibrationResult, Engine, EngineJob

if TYPE_CHECKING:
    import cirq
    import cirq_google

    # Workaround for mypy custom dataclasses (python/mypy#5406)
    from dataclasses import dataclass as json_serializable_dataclass
else:
    from cirq.protocols import json_serializable_dataclass


_FLOQUET_PHASED_FSIM_HANDLER_NAME = 'floquet_phased_fsim_characterization'
_XEB_PHASED_FSIM_HANDLER_NAME = 'xeb_phased_fsim_characterization'
_DEFAULT_XEB_CYCLE_DEPTHS = (5, 25, 50, 100, 200, 300)

T = TypeVar('T')

RequestT = TypeVar('RequestT', bound='PhasedFSimCalibrationRequest')


# Workaround for: https://github.com/python/mypy/issues/5858
def lru_cache_typesafe(func: Callable[..., T]) -> T:
    return functools.lru_cache(maxsize=None)(func)  # type: ignore


def _create_pairs_from_moment(moment: Moment) -> Tuple[Tuple[Tuple[Qid, Qid], ...], Gate]:
    """Creates instantiation parameters from a Moment.

    Given a moment, creates a tuple of pairs of qubits and the
    gate for instantiation of a sub-class of PhasedFSimCalibrationRequest,
    Sub-classes of PhasedFSimCalibrationRequest can call this function
    to implement a from_moment function.
    """
    gate = None
    pairs: List[Tuple[Qid, Qid]] = []
    for op in moment:
        if op.gate is None:
            raise ValueError('All gates in request object must be two qubit gates: {op}')
        if gate is None:
            gate = op.gate
        elif gate != op.gate:
            raise ValueError('All gates in request object must be identical {gate}!={op.gate}')
        if len(op.qubits) != 2:
            raise ValueError('All gates in request object must be two qubit gates: {op}')
        pairs.append((op.qubits[0], op.qubits[1]))
    if gate is None:
        raise ValueError('No gates found to create request {moment}')
    return tuple(pairs), gate


@json_serializable_dataclass(frozen=True)
class PhasedFSimCharacterization:
    """Holder for the unitary angles of the cirq.PhasedFSimGate.

    This class stores five unitary parameters (θ, ζ, χ, γ and φ) that describe the
    cirq.PhasedFSimGate which is the most general particle conserving two-qubit gate. The unitary
    of the underlying gate is:

        [[1,                        0,                       0,                0],
         [0,    exp(-i(γ + ζ)) cos(θ), -i exp(-i(γ - χ)) sin(θ),               0],
         [0, -i exp(-i(γ + χ)) sin(θ),    exp(-i(γ - ζ)) cos(θ),               0],
         [0,                        0,                       0,  exp(-i(2γ + φ))]]

    The parameters θ, γ and φ are symmetric and parameters ζ and χ asymmetric under the qubits
    exchange.

    All the angles described by this class are optional and can be left unknown. This is relevant
    for characterization routines that characterize only subset of the gate parameters. All the
    angles are assumed to take a fixed numerical values which reflect the current state of the
    characterized gate.

    This class supports JSON serialization and deserialization.

    Attributes:
        theta: θ angle in radians or None when unknown.
        zeta: ζ angle in radians or None when unknown.
        chi: χ angle in radians or None when unknown.
        gamma: γ angle in radians or None when unknown.
        phi: φ angle in radians or None when unknown.
    """

    theta: Optional[float] = None
    zeta: Optional[float] = None
    chi: Optional[float] = None
    gamma: Optional[float] = None
    phi: Optional[float] = None

    def asdict(self) -> Dict[str, float]:
        """Converts parameters to a dictionary that maps angle names to values."""
        return dataclasses.asdict(self)

    def all_none(self) -> bool:
        """Returns True if all the angles are None"""
        return (
            self.theta is None
            and self.zeta is None
            and self.chi is None
            and self.gamma is None
            and self.phi is None
        )

    def any_none(self) -> bool:
        """Returns True if any the angle is None"""
        return (
            self.theta is None
            or self.zeta is None
            or self.chi is None
            or self.gamma is None
            or self.phi is None
        )

    def parameters_for_qubits_swapped(self) -> 'PhasedFSimCharacterization':
        """Parameters for the gate with qubits swapped between each other.

        The angles theta, gamma and phi are kept unchanged. The angles zeta and chi are negated for
        the gate with swapped qubits.

        Returns:
            New instance with angles adjusted for swapped qubits.
        """
        return PhasedFSimCharacterization(
            theta=self.theta,
            zeta=-self.zeta if self.zeta is not None else None,
            chi=-self.chi if self.chi is not None else None,
            gamma=self.gamma,
            phi=self.phi,
        )

    def merge_with(self, other: 'PhasedFSimCharacterization') -> 'PhasedFSimCharacterization':
        """Substitutes missing parameter with values from other.

        Args:
            other: Parameters to use for None values.

        Returns:
            New instance of PhasedFSimCharacterization with values from this instance if they are
            set or values from other when some parameter is None.
        """
        return PhasedFSimCharacterization(
            theta=other.theta if self.theta is None else self.theta,
            zeta=other.zeta if self.zeta is None else self.zeta,
            chi=other.chi if self.chi is None else self.chi,
            gamma=other.gamma if self.gamma is None else self.gamma,
            phi=other.phi if self.phi is None else self.phi,
        )

    def override_by(self, other: 'PhasedFSimCharacterization') -> 'PhasedFSimCharacterization':
        """Overrides other parameters that are not None.

        Args:
            other: Parameters to use for override.

        Returns:
            New instance of PhasedFSimCharacterization with values from other if set (values from
            other that are not None). Otherwise the current values are used.
        """
        return other.merge_with(self)


SQRT_ISWAP_INV_PARAMETERS = PhasedFSimCharacterization(
    theta=np.pi / 4, zeta=0.0, chi=0.0, gamma=0.0, phi=0.0
)


class PhasedFSimCalibrationOptions(abc.ABC, Generic[RequestT]):
    """Base class for calibration-specific options passed together with the requests."""

    @abc.abstractmethod
    def create_phased_fsim_request(
        self,
        pairs: Tuple[Tuple[Qid, Qid], ...],
        gate: Gate,
    ) -> RequestT:
        """Create a PhasedFSimCalibrationRequest of the correct type for these options.

        Args:
            pairs: Set of qubit pairs to characterize. A single qubit can appear on at most one
                pair in the set.
            gate: Gate to characterize for each qubit pair from pairs. This must be a supported gate
                which can be described cirq.PhasedFSim gate. This gate must be serialized by the
                cirq_google.SerializableGateSet used
        """


@dataclasses.dataclass
class PhasedFSimCalibrationResult:
    """The PhasedFSimGate characterization result.

    Attributes:
        parameters: Map from qubit pair to characterization result. For each pair of characterized
            quibts a and b either only (a, b) or only (b, a) is present.
        gate: Characterized gate for each qubit pair. This is copied from the matching
            PhasedFSimCalibrationRequest and is included to preserve execution context.
        options: The options used to gather this result.
        project_id: Google's job project id.
        program_id: Google's job program id.
        job_id: Google's job job id.
    """

    parameters: Dict[Tuple[Qid, Qid], PhasedFSimCharacterization]
    gate: Gate
    options: PhasedFSimCalibrationOptions
    project_id: Optional[str] = None
    program_id: Optional[str] = None
    job_id: Optional[str] = None
    _engine_job: Optional[EngineJob] = None
    _calibration: Optional[Calibration] = None

    def override(self, parameters: PhasedFSimCharacterization) -> 'PhasedFSimCalibrationResult':
        """Creates the new results with certain parameters overridden for all characterizations.

        This functionality can be used to zero-out the corrected angles and do the analysis on
        remaining errors.

        Args:
            parameters: Parameters that will be used when overriding. The angles of that object
                which are not None will be used to replace current parameters for every pair stored.

        Returns:
            New instance of PhasedFSimCalibrationResult with certain parameters overriden.
        """
        return PhasedFSimCalibrationResult(
            parameters={
                pair: pair_parameters.override_by(parameters)
                for pair, pair_parameters in self.parameters.items()
            },
            gate=self.gate,
            options=self.options,
        )

    def get_parameters(self, a: Qid, b: Qid) -> Optional['PhasedFSimCharacterization']:
        """Returns parameters for a qubit pair (a, b) or None when unknown."""
        if (a, b) in self.parameters:
            return self.parameters[(a, b)]
        elif (b, a) in self.parameters:
            return self.parameters[(b, a)].parameters_for_qubits_swapped()
        else:
            return None

    @property
    def engine_job(self) -> Optional[EngineJob]:
        """The cirq_google.EngineJob associated with this calibration request.

        Available only when project_id, program_id and job_id attributes are present.
        """
        if self._engine_job is None and self.project_id and self.program_id and self.job_id:
            engine = Engine(project_id=self.project_id)
            self._engine_job = engine.get_program(self.program_id).get_job(self.job_id)
        return self._engine_job

    @property
    def engine_calibration(self) -> Optional[Calibration]:
        """The underlying device calibration that was used for this user-specific calibration.

        This is a cached property that triggers a network call at the first use.
        """
        if self._calibration is None and self.engine_job is not None:
            self._calibration = self.engine_job.get_calibration()
        return self._calibration

    @classmethod
    def _create_parameters_dict(
        cls,
        parameters: List[Tuple[Qid, Qid, PhasedFSimCharacterization]],
    ) -> Dict[Tuple[Qid, Qid], PhasedFSimCharacterization]:
        """Utility function to create parameters from JSON.

        Can be used from child classes to instantiate classes in a _from_json_dict_
        method."""
        return {(q_a, q_b): params for q_a, q_b, params in parameters}

    @classmethod
    def _from_json_dict_(
        cls,
        **kwargs,
    ) -> 'PhasedFSimCalibrationResult':
        """Magic method for the JSON serialization protocol.

        Converts serialized dictionary into a dict suitable for
        class instantiation."""
        del kwargs['cirq_type']
        kwargs['parameters'] = cls._create_parameters_dict(kwargs['parameters'])
        return cls(**kwargs)

    def _json_dict_(self) -> Dict[str, Any]:
        """Magic method for the JSON serialization protocol."""
        return {
            'cirq_type': 'PhasedFSimCalibrationResult',
            'gate': self.gate,
            'parameters': [(q_a, q_b, params) for (q_a, q_b), params in self.parameters.items()],
            'options': self.options,
            'project_id': self.project_id,
            'program_id': self.program_id,
            'job_id': self.job_id,
        }


def merge_matching_results(
    results: Iterable[PhasedFSimCalibrationResult],
) -> Optional[PhasedFSimCalibrationResult]:
    """Merges a collection of results into a single result.

    Args:
        results: List of results to merge. They must be compatible with each other: all gate and
            options fields must be equal and every characterized pair must be present only in one of
            the characterizations.

    Returns:
        New PhasedFSimCalibrationResult that contains all the parameters from every result in
        results or None when the results list is empty.
    """
    all_parameters: Dict[Tuple[Qid, Qid], PhasedFSimCharacterization] = {}
    common_gate = None
    common_options = None
    for result in results:
        if common_gate is None:
            common_gate = result.gate
        elif common_gate != result.gate:
            raise ValueError(
                f'Only matching results can be merged, got gates {common_gate} and {result.gate}'
            )

        if common_options is None:
            common_options = result.options
        elif common_options != result.options:
            raise ValueError(
                f'Only matching results can be merged, got options {common_options} and '
                f'{result.options}'
            )

        if not all_parameters.keys().isdisjoint(result.parameters):
            raise ValueError(f'Only results with disjoint parameters sets can be merged')

        all_parameters.update(result.parameters)

    if common_gate is None or common_options is None:
        return None

    return PhasedFSimCalibrationResult(all_parameters, common_gate, common_options)


class PhasedFSimCalibrationError(Exception):
    """Error that indicates the calibration failure."""


# We have to relax a mypy constraint, see https://github.com/python/mypy/issues/5374
@dataclasses.dataclass(frozen=True)  # type: ignore
class PhasedFSimCalibrationRequest(abc.ABC):
    """Description of the request to characterize PhasedFSimGate.

    Attributes:
        pairs: Set of qubit pairs to characterize. A single qubit can appear on at most one pair in
            the set.
        gate: Gate to characterize for each qubit pair from pairs. This must be a supported gate
            which can be described cirq.PhasedFSim gate. This gate must be serialized by the
            cirq_google.SerializableGateSet used
    """

    pairs: Tuple[Tuple[Qid, Qid], ...]
    gate: Gate  # Any gate which can be described by cirq.PhasedFSim
    options: PhasedFSimCalibrationOptions

    # Workaround for: https://github.com/python/mypy/issues/1362
    @property  # type: ignore
    @lru_cache_typesafe
    def qubit_to_pair(self) -> MutableMapping[Qid, Tuple[Qid, Qid]]:
        """Returns mapping from qubit to a qubit pair that it belongs to."""
        # Returning mutable mapping as a cached result because it's hard to get a frozen dictionary
        # in Python...
        return collections.ChainMap(*({q: pair for q in pair} for pair in self.pairs))

    @abc.abstractmethod
    def to_calibration_layer(self) -> CalibrationLayer:
        """Encodes this characterization request in a CalibrationLayer object."""

    @abc.abstractmethod
    def parse_result(
        self, result: CalibrationResult, job: Optional[EngineJob] = None
    ) -> PhasedFSimCalibrationResult:
        """Decodes the characterization result issued for this request."""


@json_serializable_dataclass(frozen=True)
class XEBPhasedFSimCalibrationOptions(PhasedFSimCalibrationOptions):
    """Options for configuring a PhasedFSim calibration using XEB.

    XEB uses the fidelity of random circuits to characterize PhasedFSim gates. The parameters
    of the gate are varied by a classical optimizer to maximize the observed fidelities.

    Args:
        n_library_circuits: The number of distinct, two-qubit random circuits to use in our
            library of random circuits. This should be the same order of magnitude as
            `n_combinations`.
        n_combinations: We take each library circuit and randomly assign it to qubit pairs.
            This parameter controls the number of random combinations of the two-qubit random
            circuits we execute. Higher values increase the precision of estimates but linearly
            increase experimental runtime.
        cycle_depths: We run the random circuits at these cycle depths to fit an exponential
            decay in the fidelity.
        fatol: The absolute convergence tolerance for the objective function evaluation in
            the Nelder-Mead optimization. This controls the runtime of the classical
            characterization optimization loop.
        xatol: The absolute convergence tolerance for the parameter estimates in
            the Nelder-Mead optimization. This controls the runtime of the classical
            characterization optimization loop.
        fsim_options: An instance of `XEBPhasedFSimCharacterizationOptions` that controls aspects
            of the PhasedFSim characterization like initial guesses and which angles to
            characterize.
    """

    n_library_circuits: int = 20
    n_combinations: int = 10
    cycle_depths: Tuple[int, ...] = _DEFAULT_XEB_CYCLE_DEPTHS
    fatol: Optional[float] = 5e-3
    xatol: Optional[float] = 5e-3

    fsim_options: XEBPhasedFSimCharacterizationOptions = XEBPhasedFSimCharacterizationOptions()

    def to_args(self) -> Dict[str, Any]:
        """Convert this dataclass to an `args` dictionary suitable for sending to the Quantum
        Engine calibration API."""
        args: Dict[str, Any] = {
            'n_library_circuits': self.n_library_circuits,
            'n_combinations': self.n_combinations,
            'cycle_depths': '_'.join(f'{cd:d}' for cd in self.cycle_depths),
        }
        if self.fatol is not None:
            args['fatol'] = self.fatol
        if self.xatol is not None:
            args['xatol'] = self.xatol

        fsim_options = dataclasses.asdict(self.fsim_options)
        fsim_options = {k: v for k, v in fsim_options.items() if v is not None}
        args.update(fsim_options)
        return args

    def create_phased_fsim_request(
        self,
        pairs: Tuple[Tuple[Qid, Qid], ...],
        gate: Gate,
    ) -> 'XEBPhasedFSimCalibrationRequest':
        return XEBPhasedFSimCalibrationRequest(pairs=pairs, gate=gate, options=self)

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        del kwargs['cirq_type']
        kwargs['cycle_depths'] = tuple(kwargs['cycle_depths'])
        return cls(**kwargs)


@json_serializable_dataclass(frozen=True)
class LocalXEBPhasedFSimCalibrationOptions(XEBPhasedFSimCalibrationOptions):
    """Options for configuring a PhasedFSim calibration using a local version of XEB.

    XEB uses the fidelity of random circuits to characterize PhasedFSim gates. The parameters
    of the gate are varied by a classical optimizer to maximize the observed fidelities.

    These "Local" options (corresponding to `LocalXEBPhasedFSimCalibrationRequest`) instruct
    `cirq_google.run_calibrations` to execute XEB analysis locally (not via the quantum
    engine). As such, `run_calibrations` can work with any `cirq.Sampler`, not just
    `QuantumEngineSampler`.

    Args:
        n_library_circuits: The number of distinct, two-qubit random circuits to use in our
            library of random circuits. This should be the same order of magnitude as
            `n_combinations`.
        n_combinations: We take each library circuit and randomly assign it to qubit pairs.
            This parameter controls the number of random combinations of the two-qubit random
            circuits we execute. Higher values increase the precision of estimates but linearly
            increase experimental runtime.
        cycle_depths: We run the random circuits at these cycle depths to fit an exponential
            decay in the fidelity.
        fatol: The absolute convergence tolerance for the objective function evaluation in
            the Nelder-Mead optimization. This controls the runtime of the classical
            characterization optimization loop.
        xatol: The absolute convergence tolerance for the parameter estimates in
            the Nelder-Mead optimization. This controls the runtime of the classical
            characterization optimization loop.
        fsim_options: An instance of `XEBPhasedFSimCharacterizationOptions` that controls aspects
            of the PhasedFSim characterization like initial guesses and which angles to
            characterize.
        n_processes: The number of multiprocessing processes to analyze the XEB characterization
            data. By default, we use a value equal to the number of CPU cores. If `1` is specified,
            multiprocessing is not used.
    """

    n_processes: Optional[int] = None

    def create_phased_fsim_request(
        self,
        pairs: Tuple[Tuple[Qid, Qid], ...],
        gate: Gate,
    ):
        return LocalXEBPhasedFSimCalibrationRequest(pairs=pairs, gate=gate, options=self)


@json_serializable_dataclass(frozen=True)
class FloquetPhasedFSimCalibrationOptions(PhasedFSimCalibrationOptions):
    """Options specific to Floquet PhasedFSimCalibration.

    Some angles require another angle to be characterized first so result might have more angles
    characterized than requested here.

    Attributes:
        characterize_theta: Whether to characterize θ angle.
        characterize_zeta: Whether to characterize ζ angle.
        characterize_chi: Whether to characterize χ angle.
        characterize_gamma: Whether to characterize γ angle.
        characterize_phi: Whether to characterize φ angle.
        readout_error_tolerance: Threshold for pairwise-correlated readout errors above which the
            calibration will report to fail. Just before each calibration all pairwise two-qubit
            readout errors are checked and when any of the pairs reports an error above the
            threshold, the calibration will fail. This value is a sanity check to determine if
            calibration is reasonable and allows for quick termination if it is not. Set to 1.0 to
            disable readout error checks and None to use default, device-specific thresholds.
    """

    characterize_theta: bool
    characterize_zeta: bool
    characterize_chi: bool
    characterize_gamma: bool
    characterize_phi: bool
    readout_error_tolerance: Optional[float] = None

    def zeta_chi_gamma_correction_override(self) -> PhasedFSimCharacterization:
        """Gives a PhasedFSimCharacterization that can be used to override characterization after
        correcting for zeta, chi and gamma angles.
        """
        return PhasedFSimCharacterization(
            zeta=0.0 if self.characterize_zeta else None,
            chi=0.0 if self.characterize_chi else None,
            gamma=0.0 if self.characterize_gamma else None,
        )

    def create_phased_fsim_request(
        self,
        pairs: Tuple[Tuple[Qid, Qid], ...],
        gate: Gate,
    ) -> 'FloquetPhasedFSimCalibrationRequest':
        return FloquetPhasedFSimCalibrationRequest(pairs=pairs, gate=gate, options=self)


"""Floquet PhasedFSimCalibrationOptions options with all angles characterization requests set to
True."""
ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION = FloquetPhasedFSimCalibrationOptions(
    characterize_theta=True,
    characterize_zeta=True,
    characterize_chi=True,
    characterize_gamma=True,
    characterize_phi=True,
)

"""XEB PhasedFSimCalibrationOptions options with all angles characterization requests set to
True."""
ALL_ANGLES_XEB_PHASED_FSIM_CHARACTERIZATION = XEBPhasedFSimCalibrationOptions(
    fsim_options=XEBPhasedFSimCharacterizationOptions(
        characterize_theta=True,
        characterize_zeta=True,
        characterize_chi=True,
        characterize_gamma=True,
        characterize_phi=True,
    )
)


"""PhasedFSimCalibrationOptions with all but chi angle characterization requests set to True."""
WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION = FloquetPhasedFSimCalibrationOptions(
    characterize_theta=True,
    characterize_zeta=True,
    characterize_chi=False,
    characterize_gamma=True,
    characterize_phi=True,
)


"""PhasedFSimCalibrationOptions with theta, zeta and gamma angles characterization requests set to
True.

Those are the most efficient options that can be used to cancel out the errors by adding the
appropriate single-qubit Z rotations to the circuit. The angles zeta, chi and gamma can be removed
by those additions. The angle chi is disabled because it's not supported by Floquet characterization
currently. The angle theta is set enabled because it is characterized together with zeta and adding
it doesn't cost anything.
"""
THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION = FloquetPhasedFSimCalibrationOptions(
    characterize_theta=True,
    characterize_zeta=True,
    characterize_chi=False,
    characterize_gamma=True,
    characterize_phi=False,
)


@dataclasses.dataclass(frozen=True)
class FloquetPhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):
    """PhasedFSim characterization request specific to Floquet calibration.

    Attributes:
        options: Floquet-specific characterization options.
    """

    options: FloquetPhasedFSimCalibrationOptions

    @classmethod
    def from_moment(cls, moment: Moment, options: FloquetPhasedFSimCalibrationOptions):
        """Creates a FloquetPhasedFSimCalibrationRequest from a Moment.

        Given a `Moment` object, this function extracts out the pairs of
        qubits and the `Gate` used to create a `FloquetPhasedFSimCalibrationRequest`
        object.  The moment must contain only identical two-qubit FSimGates.
        If dissimilar gates are passed in, a ValueError is raised.
        """
        pairs, gate = _create_pairs_from_moment(moment)
        return cls(pairs, gate, options)

    def to_calibration_layer(self) -> CalibrationLayer:
        circuit = Circuit([self.gate.on(*pair) for pair in self.pairs])
        args: Dict[str, Any] = {
            'est_theta': self.options.characterize_theta,
            'est_zeta': self.options.characterize_zeta,
            'est_chi': self.options.characterize_chi,
            'est_gamma': self.options.characterize_gamma,
            'est_phi': self.options.characterize_phi,
            # Experimental option that should always be set to True.
            'readout_corrections': True,
        }
        if self.options.readout_error_tolerance is not None:
            # Maximum error of the diagonal elements of the two-qubit readout confusion matrix.
            args['readout_error_tolerance'] = self.options.readout_error_tolerance
            # Maximum error of the off-diagonal elements of the two-qubit readout confusion matrix.
            args['correlated_readout_error_tolerance'] = _correlated_from_readout_tolerance(
                self.options.readout_error_tolerance
            )
        return CalibrationLayer(
            calibration_type=_FLOQUET_PHASED_FSIM_HANDLER_NAME,
            program=circuit,
            args=args,
        )

    def parse_result(
        self, result: CalibrationResult, job: Optional[EngineJob] = None
    ) -> PhasedFSimCalibrationResult:
        if result.code != v2.calibration_pb2.SUCCESS:
            raise PhasedFSimCalibrationError(result.error_message)

        decoded: Dict[int, Dict[str, Any]] = collections.defaultdict(lambda: {})
        for keys, values in result.metrics['angles'].items():
            for key, value in zip(keys, values):
                match = re.match(r'(\d+)_(.+)', str(key))
                if not match:
                    raise ValueError(f'Unknown metric name {key}')
                index = int(match[1])
                name = match[2]
                decoded[index][name] = value

        parsed = {}
        for data in decoded.values():
            a = v2.qubit_from_proto_id(data['qubit_a'])
            b = v2.qubit_from_proto_id(data['qubit_b'])
            parsed[(a, b)] = PhasedFSimCharacterization(
                theta=data.get('theta_est', None),
                zeta=data.get('zeta_est', None),
                chi=data.get('chi_est', None),
                gamma=data.get('gamma_est', None),
                phi=data.get('phi_est', None),
            )

        return PhasedFSimCalibrationResult(
            parameters=parsed,
            gate=self.gate,
            options=self.options,
            project_id=None if job is None else job.project_id,
            program_id=None if job is None else job.program_id,
            job_id=None if job is None else job.job_id,
        )

    @classmethod
    def _from_json_dict_(
        cls,
        gate: Gate,
        pairs: List[Tuple[Qid, Qid]],
        options: FloquetPhasedFSimCalibrationOptions,
        **kwargs,
    ) -> 'FloquetPhasedFSimCalibrationRequest':
        """Magic method for the JSON serialization protocol.

        Converts serialized dictionary into a dict suitable for
        class instantiation."""
        instantiation_pairs = tuple((q_a, q_b) for q_a, q_b in pairs)
        return cls(instantiation_pairs, gate, options)

    def _json_dict_(self) -> Dict[str, Any]:
        """Magic method for the JSON serialization protocol."""
        return {
            'cirq_type': 'FloquetPhasedFSimCalibrationRequest',
            'pairs': [(pair[0], pair[1]) for pair in self.pairs],
            'gate': self.gate,
            'options': self.options,
        }


def _correlated_from_readout_tolerance(readout_tolerance: float) -> float:
    """Heuristic formula for the off-diagonal confusion matrix error thresholds.

    This is chosen to return 0.3 for readout_tolerance = 0.4 and 1.0 for readout_tolerance = 1.0.
    """
    return max(0.0, min(1.0, 7 / 6 * readout_tolerance - 1 / 6))


def _get_labeled_int(key: str, s: str):
    ma = re.match(rf'{key}_(\d+)$', s)
    if ma is None:
        raise ValueError(f"Could not parse {key} value for {s}")
    return int(ma.group(1))


def _parse_xeb_fidelities_df(metrics: 'cirq_google.Calibration', super_name: str) -> pd.DataFrame:
    """Parse a fidelities DataFrame from Metric protos.

    Args:
        metrics: The metrics from a CalibrationResult
        super_name: The metric name prefix. We will extract information for metrics named like
            "{super_name}_depth_{depth}", so you can have multiple independent DataFrames in
            one CalibrationResult.
    """
    records: List[Dict[str, Union[int, float, Tuple['cirq.Qid', 'cirq.Qid']]]] = []
    for metric_name in metrics.keys():
        ma = re.match(fr'{super_name}_depth_(\d+)$', metric_name)
        if ma is None:
            continue

        for (layer_str, pair_str, qa, qb), (value,) in metrics[metric_name].items():
            records.append(
                {
                    'cycle_depth': int(ma.group(1)),
                    'layer_i': _get_labeled_int('layer', cast(str, layer_str)),
                    'pair_i': _get_labeled_int('pair', cast(str, pair_str)),
                    'fidelity': float(value),
                    'pair': (cast(GridQubit, qa), cast(GridQubit, qb)),
                }
            )
    return pd.DataFrame(records)


def _parse_characterized_angles(
    metrics: 'cirq_google.Calibration',
    super_name: str,
) -> Dict[Tuple['cirq.Qid', 'cirq.Qid'], Dict[str, float]]:
    """Parses characterized angles from Metric protos.

    Args:
        metrics: The metrics from a CalibrationResult
        super_name: The metric name prefix. We extract angle names as "{super_name}_{angle_name}".
    """

    records: Dict[Tuple['cirq.Qid', 'cirq.Qid'], Dict[str, float]] = collections.defaultdict(dict)
    for metric_name in metrics.keys():
        ma = re.match(fr'{super_name}_(\w+)$', metric_name)
        if ma is None:
            continue

        angle_name = ma.group(1)
        for (qa, qb), (value,) in metrics[metric_name].items():
            qa = cast(GridQubit, qa)
            qb = cast(GridQubit, qb)
            value = float(value)
            records[qa, qb][angle_name] = value
    return dict(records)


@json_serializable_dataclass(frozen=True)
class LocalXEBPhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):
    """PhasedFSim characterization request for local cross entropy benchmarking (XEB) calibration.

    A "Local" request (corresponding to `LocalXEBPhasedFSimCalibrationOptions`) instructs
    `cirq_google.run_calibrations` to execute XEB analysis locally (not via the quantum
    engine). As such, `run_calibrations` can work with any `cirq.Sampler`, not just
    `QuantumEngineSampler`.

    Attributes:
        options: local-XEB-specific characterization options.
    """

    options: LocalXEBPhasedFSimCalibrationOptions

    def parse_result(
        self, result: CalibrationResult, job: Optional[EngineJob] = None
    ) -> PhasedFSimCalibrationResult:
        raise NotImplementedError('Not applicable for local calibrations')

    def to_calibration_layer(self) -> CalibrationLayer:
        raise NotImplementedError('Not applicable for local calibrations')

    @classmethod
    def _from_json_dict_(
        cls,
        gate: Gate,
        pairs: List[Tuple[Qid, Qid]],
        options: LocalXEBPhasedFSimCalibrationOptions,
        **kwargs,
    ) -> 'LocalXEBPhasedFSimCalibrationRequest':
        # List -> Tuple
        instantiation_pairs = tuple((q_a, q_b) for q_a, q_b in pairs)
        return cls(instantiation_pairs, gate, options)


@json_serializable_dataclass(frozen=True)
class XEBPhasedFSimCalibrationRequest(PhasedFSimCalibrationRequest):
    """PhasedFSim characterization request for cross entropy benchmarking (XEB) calibration.

    Attributes:
        options: XEB-specific characterization options.
    """

    options: XEBPhasedFSimCalibrationOptions

    def to_calibration_layer(self) -> CalibrationLayer:
        circuit = Circuit([self.gate.on(*pair) for pair in self.pairs])
        return CalibrationLayer(
            calibration_type=_XEB_PHASED_FSIM_HANDLER_NAME,
            program=circuit,
            args=self.options.to_args(),
        )

    def parse_result(
        self, result: CalibrationResult, job: Optional[EngineJob] = None
    ) -> PhasedFSimCalibrationResult:
        if result.code != v2.calibration_pb2.SUCCESS:
            raise PhasedFSimCalibrationError(result.error_message)

        # pylint: disable=unused-variable
        initial_fids = _parse_xeb_fidelities_df(result.metrics, 'initial_fidelities')
        final_fids = _parse_xeb_fidelities_df(result.metrics, 'final_fidelities')
        # pylint: enable=unused-variable

        final_params = {
            pair: PhasedFSimCharacterization(**angles)
            for pair, angles in _parse_characterized_angles(
                result.metrics, 'characterized_angles'
            ).items()
        }

        # TODO: Return initial_fids, final_fids somehow.
        return PhasedFSimCalibrationResult(
            parameters=final_params,
            gate=self.gate,
            options=self.options,
            project_id=None if job is None else job.project_id,
            program_id=None if job is None else job.program_id,
            job_id=None if job is None else job.job_id,
        )

    @classmethod
    def _from_json_dict_(
        cls,
        gate: Gate,
        pairs: List[Tuple[Qid, Qid]],
        options: XEBPhasedFSimCalibrationOptions,
        **kwargs,
    ) -> 'XEBPhasedFSimCalibrationRequest':
        # List -> Tuple
        instantiation_pairs = tuple((q_a, q_b) for q_a, q_b in pairs)
        return cls(instantiation_pairs, gate, options)


class IncompatibleMomentError(Exception):
    """Error that occurs when a moment is not supported by a calibration routine."""


@dataclasses.dataclass(frozen=True)
class PhaseCalibratedFSimGate:
    """Association of a user gate with gate to calibrate.

    This association stores information regarding rotation of the calibrated FSim gate by
    phase_exponent p:

        (Z^-p ⊗ Z^p) FSim (Z^p ⊗ Z^-p).

    The rotation should be reflected back during the compilation after the gate is calibrated and
    is equivalent to the shift of -2πp in the χ angle of PhasedFSimGate.

    Attributes:
        engine_gate: Gate that should be used for calibration purposes.
        phase_exponent: Phase rotation exponent p.
    """

    engine_gate: FSimGate
    phase_exponent: float

    def as_characterized_phased_fsim_gate(
        self, parameters: PhasedFSimCharacterization
    ) -> PhasedFSimGate:
        """Creates a PhasedFSimGate which represents the characterized engine_gate but includes
        deviations in unitary parameters.

        Args:
            parameters: The results of characterization of the engine gate.

        Returns:
            Instance of PhasedFSimGate that executes a gate according to the characterized
            parameters of the engine_gate.
        """
        return PhasedFSimGate(
            theta=parameters.theta,
            zeta=parameters.zeta,
            chi=parameters.chi - 2 * np.pi * self.phase_exponent,
            gamma=parameters.gamma,
            phi=parameters.phi,
        )

    def with_zeta_chi_gamma_compensated(
        self,
        qubits: Tuple[Qid, Qid],
        parameters: PhasedFSimCharacterization,
        *,
        engine_gate: Optional[TwoQubitGate] = None,
    ) -> Tuple[Tuple[Operation, ...], ...]:
        """Creates a composite operation that compensates for zeta, chi and gamma angles of the
        characterization.

        Args:
            qubits: Qubits that the gate should act on.
            parameters: The results of characterization of the engine gate.
            engine_gate: TwoQubitGate that represents the engine gate. When None, the internal
                engine_gate of this instance is used. This argument is useful for testing
                purposes.

        Returns:
            Tuple of tuple of operations that describe the compensated gate. The first index
            iterates over moments of the composed operation.
        """
        assert parameters.zeta is not None, "Zeta value must not be None"
        zeta = parameters.zeta

        assert parameters.gamma is not None, "Gamma value must not be None"
        gamma = parameters.gamma

        assert parameters.chi is not None, "Chi value must not be None"
        chi = parameters.chi + 2 * np.pi * self.phase_exponent

        if engine_gate is None:
            engine_gate = self.engine_gate

        a, b = qubits

        alpha = 0.5 * (zeta + chi)
        beta = 0.5 * (zeta - chi)

        return (
            (rz(0.5 * gamma - alpha).on(a), rz(0.5 * gamma + alpha).on(b)),
            (engine_gate.on(a, b),),
            (rz(0.5 * gamma - beta).on(a), rz(0.5 * gamma + beta).on(b)),
        )


def try_convert_sqrt_iswap_to_fsim(gate: Gate) -> Optional[PhaseCalibratedFSimGate]:
    """Converts an equivalent gate to FSimGate(theta=π/4, phi=0) if possible.

    Args:
        gate: Gate to verify.

    Returns:
        FSimGateCalibration with engine_gate FSimGate(theta=π/4, phi=0) if the provided gate is
        either FSimGate, ISWapPowGate, PhasedFSimGate or PhasedISwapPowGate that is equivalent to
        FSimGate(theta=±π/4, phi=0). None otherwise.
    """
    if isinstance(gate, FSimGate):
        if not np.isclose(gate.phi, 0.0):
            return None
        angle = gate.theta
    elif isinstance(gate, ISwapPowGate):
        angle = -gate.exponent * np.pi / 2
    elif isinstance(gate, PhasedFSimGate):
        if (
            not np.isclose(gate.zeta, 0.0)
            or not np.isclose(gate.chi, 0.0)
            or not np.isclose(gate.gamma, 0.0)
            or not np.isclose(gate.phi, 0.0)
        ):
            return None
        angle = gate.theta
    elif isinstance(gate, PhasedISwapPowGate):
        if not np.isclose(-gate.phase_exponent - 0.5, 0.0):
            return None
        angle = gate.exponent * np.pi / 2
    else:
        return None

    angle_canonical = angle % (2 * np.pi)

    if np.isclose(angle_canonical, np.pi / 4):
        return PhaseCalibratedFSimGate(FSimGate(theta=np.pi / 4, phi=0.0), 0.0)
    elif np.isclose(angle_canonical, 7 * np.pi / 4):
        return PhaseCalibratedFSimGate(FSimGate(theta=np.pi / 4, phi=0.0), 0.5)

    return None
