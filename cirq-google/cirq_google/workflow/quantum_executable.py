import abc
import dataclasses
from dataclasses import dataclass
from typing import Union, Tuple

import cirq.work
from cirq import NamedTopology
from cirq.protocols import dataclass_json_dict


class ExecutableSpec(metaclass=abc.ABCMeta):
    """Specification metadata about an executable.

    Subclasses should add problem-specific fields.
    """

    executable_family: str = NotImplemented
    """A unique name to group executables."""


@dataclass(frozen=True)
class BitstringsMeasurement:
    """Use in-circuit MeasurementGate to collect many repetitions of strings of bits.

    This is the lowest-level measurement type allowed in `QuantumExecutable` and behaves
    identically to the `cirq.Sampler.run` function. The executable's circuit must contain
    explicit measurement gates.

    Args:
        n_repeitions: The number of repetitions to execute the circuit.
    """

    n_repetitions: int

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')


TParamPair = Tuple[cirq.TParamKey, cirq.TParamVal]


@dataclass(frozen=True)
class QuantumExecutable:
    """An executable quantum program.

    This serves a similar purpose to `cirq.Circuit` with some key differences. First, a quantum
    executable contains all the relevant context for execution including parameters as well as
    the desired number of repetitions. Second, this object is immutable. Finally, there are
    optional fields enabling a higher level of abstraction for certain aspects of the executable.

    Attributes:
        circuit: A circuit describing the quantum operations to execute.
        measurement: A description of the type of measurement. A valid option is to use
            MeasurementGate in your circuit and specify
            `measurement=BitstringsMeasurement(n_repetitions)`. Future changes may permit
            different types of measurement.
        params: An immutable version of cirq.ParamResolver represented as a tuple of key value
            pairs.
        spec: Specification metadata about this executable that is not used by the quantum runtime,
            but is persisted in result objects to associate executables with results.
        problem_topology: Description of the multiqubit gate topology present in the circuit.
            If not specified, the circuit must be compatible with the device topology.
        initial_state: How to initialize the quantum system before running `circuit`. If not
            specified, the device will be initialized into the all-zeros state.
    """

    circuit: cirq.FrozenCircuit
    measurement: BitstringsMeasurement
    params: Tuple[TParamPair, ...] = None
    spec: ExecutableSpec = None
    problem_topology: NamedTopology = None
    initial_state: cirq.ProductState = None

    # pylint: disable=missing-raises-doc
    def __init__(
        self,
        circuit: cirq.AbstractCircuit,
        measurement: BitstringsMeasurement,
        params: Union[Tuple[TParamPair, ...], cirq.ParamResolverOrSimilarType] = None,
        spec: ExecutableSpec = None,
        problem_topology: NamedTopology = None,
        initial_state: cirq.ProductState = None,
    ):
        """Initialize the quantum executable.

        The actual fields in this class are immutable, but we allow more liberal input types
        which will be frozen in this __init__ method.

        Args:
            circuit: The circuit. This will be frozen before being set as an attribute.
            measurement: A description of the type of measurement.
            params: A cirq.ParamResolverOrSimilarType which will be frozen into a tuple of
                key value pairs.
            spec: Specification metadata about this executable that is not used by the quantum
                runtime, but is persisted in result objects to associate executables with results.
            problem_topology: Description of the multiqubit gate topology present in the circuit.
                If not specified, the circuit must be compatible with the device topology.
            initial_state: How to initialize the quantum system before running `circuit`. If not
                specified, the device will be initialized into the all-zeros state.
        """

        # We care a lot about mutability in this class. No object is truly immutable in Python,
        # but we can get pretty close by following the example of dataclass(frozen=True), which
        # deletes this class's __setattr__ magic method. To set values ever, we use
        # object.__setattr__ in this __init__ function.
        #
        # We write our own __init__ function to be able to accept a wider range of input formats
        # that can be easily converted to our native, immutable format.
        object.__setattr__(self, 'circuit', circuit.freeze())

        if not isinstance(measurement, BitstringsMeasurement):
            raise ValueError(f"`measurement` should be a BitstringsMeasurement, not {measurement}.")
        object.__setattr__(self, 'measurement', measurement)

        if isinstance(params, tuple) and all(
            isinstance(param_kv, tuple) and len(param_kv) == 2 for param_kv in params
        ):
            frozen_params = params
        elif isinstance(params, list) and all(
            isinstance(param_kv, list) and len(param_kv) == 2 for param_kv in params
        ):
            frozen_params = tuple((k, v) for k, v in params)
        else:
            param_resolver = cirq.ParamResolver(params)
            frozen_params = tuple(param_resolver.param_dict.items())
        object.__setattr__(self, 'params', frozen_params)

        if not isinstance(spec, ExecutableSpec):
            raise ValueError(f"`spec` should be an ExecutableSpec, not {spec}.")
        object.__setattr__(self, 'spec', spec)

        if problem_topology is not None and not isinstance(problem_topology, NamedTopology):
            raise ValueError(
                f"`problem_topology` should be a NamedTopology, " f"not {problem_topology}."
            )
        object.__setattr__(self, 'problem_topology', problem_topology)

        if initial_state is not None and not isinstance(initial_state, cirq.ProductState):
            raise ValueError(f"`initial_state` should be a ProductState, not {initial_state}.")
        object.__setattr__(self, 'initial_state', initial_state)

        object.__setattr__(self, '_hash', hash(dataclasses.astuple(self)))

    def __str__(self):
        return f'QuantumExecutable(spec={self.spec})'

    def _json_dict_(self):
        return dataclass_json_dict(self, namespace='cirq.google')
