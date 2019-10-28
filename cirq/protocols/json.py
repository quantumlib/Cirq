# Copyright 2019 The Cirq Developers
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

import json
from typing import Union, Any, Dict, Optional, List, Callable, Type, cast, \
    TYPE_CHECKING, Iterable

import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol

from cirq.ops import raw_types  # Tells mypy that the raw_types module exists
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq.ops.pauli_gates
    import cirq.devices.unconstrained_device


class _ResolverCache:
    """Lazily import and build registry to avoid circular imports."""

    def __init__(self):
        self._crd = None

    @property
    def cirq_class_resolver_dictionary(self) -> Dict[str, Type]:
        if self._crd is None:
            import cirq
            from cirq.devices.noise_model import _NoNoiseModel
            from cirq.google.known_devices import _NamedConstantXmonDevice
            from cirq.contrib.quantum_volume import QuantumVolumeResult
            self._crd = {
                'AmplitudeDampingChannel': cirq.AmplitudeDampingChannel,
                'AsymmetricDepolarizingChannel':
                cirq.AsymmetricDepolarizingChannel,
                'BitFlipChannel': cirq.BitFlipChannel,
                'CCXPowGate': cirq.CCXPowGate,
                'CCZPowGate': cirq.CCZPowGate,
                'CNotPowGate': cirq.CNotPowGate,
                'CSwapGate': cirq.CSwapGate,
                'CZPowGate': cirq.CZPowGate,
                'Circuit': cirq.Circuit,
                'DepolarizingChannel': cirq.DepolarizingChannel,
                'ConstantQubitNoiseModel': cirq.ConstantQubitNoiseModel,
                'Duration': cirq.Duration,
                'FSimGate': cirq.FSimGate,
                'DensePauliString': cirq.DensePauliString,
                'MutableDensePauliString': cirq.MutableDensePauliString,
                'GateOperation': cirq.GateOperation,
                'GeneralizedAmplitudeDampingChannel':
                cirq.GeneralizedAmplitudeDampingChannel,
                'GlobalPhaseOperation': cirq.GlobalPhaseOperation,
                'GridQubit': cirq.GridQubit,
                'HPowGate': cirq.HPowGate,
                'ISwapPowGate': cirq.ISwapPowGate,
                'IdentityGate': cirq.IdentityGate,
                'IdentityOperation': cirq.IdentityOperation,
                'LineQubit': cirq.LineQubit,
                'LineQid': cirq.LineQid,
                'MeasurementGate': cirq.MeasurementGate,
                'Moment': cirq.Moment,
                '_NamedConstantXmonDevice': _NamedConstantXmonDevice,
                '_NoNoiseModel': _NoNoiseModel,
                'NamedQubit': cirq.NamedQubit,
                '_PauliX': cirq.ops.pauli_gates._PauliX,
                '_PauliY': cirq.ops.pauli_gates._PauliY,
                '_PauliZ': cirq.ops.pauli_gates._PauliZ,
                'PauliString': cirq.PauliString,
                'PhaseDampingChannel': cirq.PhaseDampingChannel,
                'PhaseFlipChannel': cirq.PhaseFlipChannel,
                'PhaseGradientGate': cirq.PhaseGradientGate,
                'PhasedISwapPowGate': cirq.PhasedISwapPowGate,
                'PhasedXPowGate': cirq.PhasedXPowGate,
                'QuantumFourierTransformGate': cirq.QuantumFourierTransformGate,
                'QuantumVolumeResult': QuantumVolumeResult,
                'ResetChannel': cirq.ResetChannel,
                'SingleQubitPauliStringGateOperation':
                cirq.SingleQubitPauliStringGateOperation,
                'SwapPowGate': cirq.SwapPowGate,
                'SycamoreGate': cirq.SycamoreGate,
                'sympy.Symbol': sympy.Symbol,
                'sympy.Add': lambda args: sympy.Add(*args),
                'sympy.Mul': lambda args: sympy.Mul(*args),
                'sympy.Pow': lambda args: sympy.Pow(*args),
                'sympy.Float': lambda approx: sympy.Float(approx),
                'sympy.Integer': sympy.Integer,
                'sympy.Rational': sympy.Rational,
                'pandas.DataFrame': pd.DataFrame,
                'pandas.Index': pd.Index,
                'pandas.MultiIndex': pd.MultiIndex.from_tuples,
                'TwoQubitMatrixGate': cirq.TwoQubitMatrixGate,
                '_UnconstrainedDevice':
                cirq.devices.unconstrained_device._UnconstrainedDevice,
                'WaitGate': cirq.WaitGate,
                '_QubitAsQid': raw_types._QubitAsQid,
                'XPowGate': cirq.XPowGate,
                'XXPowGate': cirq.XXPowGate,
                'YPowGate': cirq.YPowGate,
                'YYPowGate': cirq.YYPowGate,
                'ZPowGate': cirq.ZPowGate,
                'ZZPowGate': cirq.ZZPowGate,

                # not a cirq class, but treated as one:
                'complex': complex,
            }
        return self._crd


RESOLVER_CACHE = _ResolverCache()


def _cirq_class_resolver(cirq_type: str) -> Union[None, Type]:
    return RESOLVER_CACHE.cirq_class_resolver_dictionary.get(cirq_type, None)


DEFAULT_RESOLVERS = [
    _cirq_class_resolver,
]
"""A default list of 'resolver' functions for use in read_json.

For more information about cirq_type resolution during deserialization
please read the docstring for `cirq.read_json`.

3rd party packages which extend Cirq's JSON serialization API should
provide their own resolver functions. 3rd party resolvers can be
prepended to this list:

    MY_DEFAULT_RESOLVERS = [_resolve_my_classes] \
                           + cirq.protocols.json.DEFAULT_RESOLVERS

    def my_read_json(file_or_fn, resolvers=None):
        if resolvers is None:
            resolvers = MY_DEFAULT_RESOLVERS
        return cirq.read_json(file_or_fn, resolvers=resolvers)
"""


class SupportsJSON(Protocol):
    """An object that can be turned into JSON dictionaries.

    The magic method _json_dict_ must return a trivially json-serializable
    type or other objects that support the SupportsJSON protocol.

    During deserialization, a class must be able to be resolved (see
    the docstring for `read_json`) and must be able to be (re-)constructed
    from the serialized parameters. If the type defines a classmethod
    `_from_json_dict_`, that will be called. Otherwise, the `cirq_type` key
    will be popped from the dictionary and used as kwargs to the type's
    constructor.
    """

    def _json_dict_(self) -> Union[None, NotImplementedType, Dict[Any, Any]]:
        pass


def obj_to_dict_helper(obj: Any,
                       attribute_names: Iterable[str],
                       namespace: Optional[str] = None) -> Dict[str, Any]:
    """Construct a dictionary containing attributes from obj

    This is useful as a helper function in objects implementing the
    SupportsJSON protocol, particularly in the _json_dict_ method.

    In addition to keys and values specified by `attribute_names`, the
    returned dictionary has an additional key "cirq_type" whose value
    is the string name of the type of `obj`.

    Args:
        obj: A python object with attributes to be placed in the dictionary.
        attribute_names: The names of attributes to serve as keys in the
            resultant dictionary. The values will be the attribute values.
        namespace: An optional prefix to the value associated with the
            key "cirq_type". The namespace name will be joined with the
            class name via a dot (.)
    """
    if namespace is not None:
        prefix = '{}.'.format(namespace)
    else:
        prefix = ''

    d = {'cirq_type': prefix + obj.__class__.__name__}
    for attr_name in attribute_names:
        d[attr_name] = getattr(obj, attr_name)
    return d


class CirqEncoder(json.JSONEncoder):
    """Extend json.JSONEncoder to support Cirq objects.

    This supports custom serialization. For details, see the documentation
    for the SupportsJSON protocol.

    In addition to serializing objects that implement the SupportsJSON
    protocol, this encoder deals with common, basic types:

     - Python complex numbers get saved as a dictionary keyed by 'real'
       and 'imag'.
     - Numpy ndarrays are converted to lists to use the json module's
       built-in support for lists.
     - Preliminary support for Sympy objects. Currently only sympy.Symbol.
       See https://github.com/quantumlib/Cirq/issues/2014
    """

    def default(self, o):
        if hasattr(o, '_json_dict_'):
            return o._json_dict_()
        if isinstance(o, complex):
            return {
                'cirq_type': 'complex',
                'real': o.real,
                'imag': o.imag,
            }
        if isinstance(o, np.ndarray):
            return o.tolist()

        # TODO: More support for sympy
        #       https://github.com/quantumlib/Cirq/issues/2014
        if isinstance(o, sympy.Symbol):
            return obj_to_dict_helper(o, ['name'], namespace='sympy')

        if isinstance(o, (sympy.Add, sympy.Mul, sympy.Pow)):
            return obj_to_dict_helper(o, ['args'], namespace='sympy')

        if isinstance(o, sympy.Integer):
            return {'cirq_type': 'sympy.Integer', 'i': o.p}

        if isinstance(o, sympy.Float):
            return {'cirq_type': 'sympy.Float', 'approx': float(o)}

        if isinstance(o, sympy.Rational):
            return {
                'cirq_type': 'sympy.Rational',
                'p': o.p,
                'q': o.q,
            }

        if isinstance(o, pd.MultiIndex):
            return {
                'cirq_type': 'pandas.MultiIndex',
                'tuples': list(o),
                'names': list(o.names),
            }

        if isinstance(o, pd.Index):
            return {
                'cirq_type': 'pandas.Index',
                'data': list(o),
                'name': o.name,
            }

        if isinstance(o, pd.DataFrame):
            cols = [o[col].tolist() for col in o.columns]
            rows = list(zip(*cols))
            return {
                'cirq_type': 'pandas.DataFrame',
                'data': rows,
                'columns': o.columns,
                'index': o.index,
            }

        return super().default(o)  # coverage: ignore


def _cirq_object_hook(d, resolvers: List[Callable[[str], Union[None, Type]]]):
    if 'cirq_type' not in d:
        return d

    for resolver in resolvers:
        cls = resolver(d['cirq_type'])
        if cls is not None:
            break
    else:
        raise ValueError("Could not resolve type '{}' "
                         "during deserialization".format(d['cirq_type']))

    if hasattr(cls, '_from_json_dict_'):
        return cls._from_json_dict_(**d)

    del d['cirq_type']
    return cls(**d)


def to_json(obj: Any, file_or_fn, *, indent=2, cls=CirqEncoder):
    """Write a JSON file containing a representation of obj.

    The object may be a cirq object or have data members that are cirq
    objects which implement the SupportsJSON protocol.

    Args:
        obj: An object which can be serialized to a JSON representation.
        file_or_fn: A filename (if a string), otherwise a file-like
            object to serve as the destination for `obj`.
        indent: Pretty-print the resulting file with this indent level.
            Passed to json.dump.
        cls: Passed to json.dump; the default value of CirqEncoder
            enables the serialization of Cirq objects which implement
            the SupportsJSON protocol. To support serialization of 3rd
            party classes, prefer adding the _json_dict_ magic method
            to your classes rather than overriding this default.
    """
    if isinstance(file_or_fn, str):
        with open(file_or_fn, 'w') as actually_a_file:
            return json.dump(obj, actually_a_file, indent=indent, cls=cls)

    return json.dump(obj, file_or_fn, indent=indent, cls=cls)


def read_json(
        file_or_fn,
        resolvers: Optional[List[Callable[[str], Union[None, Type]]]] = None):
    """Read a JSON file that optionally contains cirq objects.

    Args:
        file_or_fn: A filename (if a string), otherwise a file-like
            object from which we read in a representation of an object.
        resolvers: A list of functions that are called in order to turn
            the serialized `cirq_type` string into a constructable class.
            By default, top-level cirq objects that implement the SupportsJSON
            protocol are supported. You can extend the list of supported types
            by pre-pending custom resolvers. Each resolver should return `None`
            to indicate that it cannot resolve the given cirq_type and that
            the next resolver should be tried.
    """
    if resolvers is None:
        # This cast is required because mypy does not accept
        # assigning an expression of type T to a variable of type
        # Optional[T]. This cast may hide actual bugs, so be careful.
        resolvers = cast(Optional[List[Callable[[str], Union[None, Type]]]],
                         DEFAULT_RESOLVERS)

    def obj_hook(x):
        return _cirq_object_hook(x, resolvers)

    if isinstance(file_or_fn, str):
        with open(file_or_fn, 'r') as file:
            return json.load(file, object_hook=obj_hook)

    return json.load(file_or_fn, object_hook=obj_hook)
