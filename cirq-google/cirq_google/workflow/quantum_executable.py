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

"""Data structures for programs executable on a quantum runtime."""

import abc
import dataclasses
from dataclasses import dataclass
from typing import Union, Tuple, Optional, Sequence, cast, Iterable, Dict, Any, List

from cirq import _compat, study
import cirq


class ExecutableSpec(metaclass=abc.ABCMeta):
    """Specification metadata about an executable.

    Subclasses should add problem-specific fields.
    """

    executable_family: str = NotImplemented
    """A unique name to group executables."""


@dataclass(frozen=True)
class KeyValueExecutableSpec(ExecutableSpec):
    """A generic executable spec whose metadata is a list of key-value pairs.

    The key-value pairs define an implicit data schema. Consider defining a problem-specific
    subclass of `ExecutableSpec` instead of using this class to realize the benefits of having
    an explicit schema.

    See Also:
        `KeyValueExecutableSpec.from_dict` will use a dictionary to populate `key_value_pairs`.

    Args:
        executable_family: A unique name to group executables.
        key_value_pairs: A tuple of key-value pairs. The keys should be strings but the values
            can be any immutable object.
    """

    executable_family: str
    key_value_pairs: Tuple[Tuple[str, Any], ...] = ()

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.dataclass_json_dict(self, namespace='cirq.google')

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, executable_family: str) -> 'KeyValueExecutableSpec':
        return cls(
            executable_family=executable_family,
            key_value_pairs=tuple((k, v) for k, v in d.items()),
        )

    @classmethod
    def _from_json_dict_(
        cls, executable_family: str, key_value_pairs: List[List[Union[str, Any]]], **kwargs
    ) -> 'KeyValueExecutableSpec':
        return cls(
            executable_family=executable_family,
            key_value_pairs=tuple((k, v) for k, v in key_value_pairs),
        )

    def __repr__(self) -> str:
        return cirq._compat.dataclass_repr(self, namespace='cirq_google')


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
        return cirq.dataclass_json_dict(self, namespace='cirq.google')

    def __repr__(self):
        return cirq._compat.dataclass_repr(self, namespace='cirq_google')


TParamPair = Tuple[cirq.TParamKey, cirq.TParamVal]


@dataclass(frozen=True)
class QuantumExecutable:
    """An executable quantum program.

    This serves a similar purpose to `cirq.Circuit` with some key differences. First, a quantum
    executable contains all the relevant context for execution including parameters as well as
    the desired number of repetitions. Second, this object is immutable. Finally, there are
    optional fields enabling a higher level of abstraction for certain aspects of the executable.

    Attributes:
        circuit: A `cirq.Circuit` describing the quantum operations to execute.
        measurement: A description of the measurement properties or process.
        params: An immutable `cirq.ParamResolver` (or similar type). It's representation is
            normalized to a tuple of key value pairs.
        spec: Optional `cg.ExecutableSpec` containing metadata about this executable that is not
            used by the quantum runtime, but will be forwarded to all downstream result objects.
        problem_topology: Optional `cirq.NamedTopology` instance specifying the topology of the
            circuit. This is useful when optimizing on-device layout. If none is provided we
            assume `circuit` already has a valid on-device layout.
        initial_state: A `cirq.ProductState` specifying the desired initial state before executing
            `circuit`. If not specified, default to the all-zeros state.
    """

    circuit: cirq.FrozenCircuit
    measurement: BitstringsMeasurement
    params: Optional[Tuple[TParamPair, ...]] = None
    spec: Optional[ExecutableSpec] = None
    problem_topology: Optional[cirq.NamedTopology] = None
    initial_state: Optional[cirq.ProductState] = None

    # pylint: disable=missing-raises-doc
    def __init__(
        self,
        circuit: cirq.AbstractCircuit,
        measurement: BitstringsMeasurement,
        params: Union[Sequence[TParamPair], cirq.ParamResolverOrSimilarType] = None,
        spec: Optional[ExecutableSpec] = None,
        problem_topology: Optional[cirq.NamedTopology] = None,
        initial_state: Optional[cirq.ProductState] = None,
    ):
        """Initialize the quantum executable.

        The actual fields in this class are immutable, but we allow more liberal input types
        which will be frozen in this __init__ method.

        Args:
            circuit: The circuit. This will be frozen before being set as an attribute.
            measurement: A description of the measurement properties or process.
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
        object.__setattr__(self, 'measurement', measurement)

        if isinstance(params, tuple) and all(
            isinstance(param_kv, tuple) and len(param_kv) == 2 for param_kv in params
        ):
            frozen_params = params
        elif isinstance(params, Sequence) and all(
            isinstance(param_kv, Sequence) and len(param_kv) == 2 for param_kv in params
        ):
            frozen_params = tuple((k, v) for k, v in params)
        elif study.resolver._is_param_resolver_or_similar_type(params):
            param_resolver = cirq.ParamResolver(cast(cirq.ParamResolverOrSimilarType, params))
            frozen_params = tuple(param_resolver.param_dict.items())
        else:
            raise ValueError(f"`params` should be a ParamResolverOrSimilarType, not {params}.")
        object.__setattr__(self, 'params', frozen_params)

        object.__setattr__(self, 'spec', spec)
        object.__setattr__(self, 'problem_topology', problem_topology)
        object.__setattr__(self, 'initial_state', initial_state)

        # Hash may be expensive to compute, especially for large circuits.
        # This should be safe since this class should be immutable. This line will
        # also check for hashibility of members at construction time.
        object.__setattr__(self, '_hash', hash(dataclasses.astuple(self)))

    def __str__(self):
        return f'QuantumExecutable(spec={self.spec})'

    def __repr__(self):
        return _compat.dataclass_repr(self, namespace='cirq_google')

    def _json_dict_(self):
        return cirq.dataclass_json_dict(self, namespace='cirq.google')


@dataclass(frozen=True)
class QuantumExecutableGroup:
    """A collection of `QuantumExecutable`s.

    Attributes:
        executables: A tuple of `cg.QuantumExecutable`.
    """

    executables: Tuple[QuantumExecutable, ...]

    def __init__(
        self,
        executables: Sequence[QuantumExecutable],
    ):
        """Initialize and normalize the quantum executable group.

        Args:
             executables: A sequence of `cg.QuantumExecutable` which will be frozen into a
                tuple.
        """

        if not isinstance(executables, tuple):
            executables = tuple(executables)
        object.__setattr__(self, 'executables', executables)

        object.__setattr__(self, '_hash', hash(dataclasses.astuple(self)))

    def __len__(self) -> int:
        return len(self.executables)

    def __iter__(self) -> Iterable[QuantumExecutable]:
        yield from self.executables

    def __str__(self) -> str:
        exe_str = ', '.join(str(exe) for exe in self.executables[:2])
        if len(self.executables) > 2:
            exe_str += ', ...'

        return f'QuantumExecutableGroup(executables=[{exe_str}])'

    def __repr__(self) -> str:
        return _compat.dataclass_repr(self, namespace='cirq_google')

    def __hash__(self) -> int:
        return self._hash  # type: ignore

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.dataclass_json_dict(self, namespace='cirq.google')
