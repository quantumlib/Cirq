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
import dataclasses
import gzip
import json
import numbers
import pathlib
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    IO,
    Iterable,
    List,
    Optional,
    overload,
    Sequence,
    Set,
    Type,
    Union,
)

import numpy as np
import pandas as pd
import sympy
from typing_extensions import Protocol

from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType

ObjectFactory = Union[Type, Callable[..., Any]]


class JsonResolver(Protocol):
    """Protocol for json resolver functions passed to read_json."""

    def __call__(self, cirq_type: str) -> Optional[ObjectFactory]:
        ...


def _lazy_resolver(dict_factory: Callable[[], Dict[str, ObjectFactory]]) -> JsonResolver:
    """A lazy JsonResolver based on a dict_factory.

    It only calls dict_factory when the first key is accessed.

    Args:
        dict_factory: a callable that generates an instance of the
          class resolution map - it is assumed to be cached
    """

    def json_resolver(cirq_type: str) -> Optional[ObjectFactory]:
        return dict_factory().get(cirq_type, None)

    return json_resolver


DEFAULT_RESOLVERS: List[JsonResolver] = []
"""A default list of 'JsonResolver' functions for use in read_json.

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


def _internal_register_resolver(dict_factory: Callable[[], Dict[str, ObjectFactory]]) -> None:
    """Register a resolver based on a dict factory for lazy initialization.

    Cirq modules are the ones referred in cirq/__init__.py. If a Cirq module
    wants to expose JSON serializable objects, it should register itself using
    this method to be supported by the protocol. See for example
    cirq/__init__.py or cirq/google/__init__.py.

    As Cirq modules are imported by cirq/__init__.py, they are different from
    3rd party packages, and as such SHOULD NEVER rely on storing a
    separate resolver based on DEAFULT_RESOLVERS because that will cause a
    partial DEFAULT_RESOLVER to be used by that module. What it contains will
    depend on where in cirq/__init__.py the module is imported first, as some
    modules might not had the chance to register themselves yet.

    Args:
        dict_factory: the callable that returns the actual dict for type names
            to types (ObjectFactory)
    """
    DEFAULT_RESOLVERS.append(_lazy_resolver(dict_factory))


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

    @doc_private
    def _json_dict_(self) -> Union[None, NotImplementedType, Dict[Any, Any]]:
        pass


def obj_to_dict_helper(
    obj: Any, attribute_names: Iterable[str], namespace: Optional[str] = None
) -> Dict[str, Any]:
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


# Copying the Python API, whose usage of `repr` annoys pylint.
# pylint: disable=redefined-builtin
def json_serializable_dataclass(
    _cls: Optional[Type] = None,
    *,
    namespace: Optional[str] = None,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
):
    """
    Create a dataclass that supports JSON serialization

    This function defers to the ordinary ``dataclass`` decorator but appends
    the ``_json_dict_`` protocol method which automatically determines
    the appropriate fields from the dataclass.

    Args:
        namespace: An optional prefix to the value associated with the
            key "cirq_type". The namespace name will be joined with the
            class name via a dot (.)
        init, repr, eq, order, unsafe_hash, frozen: Forwarded to the
            ``dataclass`` constructor.
    """

    def wrap(cls):
        cls = dataclasses.dataclass(
            cls, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen
        )

        cls._json_dict_ = lambda obj: obj_to_dict_helper(
            obj, [f.name for f in dataclasses.fields(cls)], namespace=namespace
        )

        return cls

    # _cls is used to deduce if we're being called as
    # @json_serializable_dataclass or @json_serializable_dataclass().
    if _cls is None:
        # We're called with parens.
        return wrap

    # We're called as @dataclass without parens.
    return wrap(_cls)


# pylint: enable=redefined-builtin


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
        # Object with custom method?
        if hasattr(o, '_json_dict_'):
            return o._json_dict_()

        # Sympy object? (Must come before general number checks.)
        # TODO: More support for sympy
        # Github issue: https://github.com/quantumlib/Cirq/issues/2014
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

        # A basic number object?
        if isinstance(o, numbers.Integral):
            return int(o)
        if isinstance(o, numbers.Real):
            return float(o)
        if isinstance(o, numbers.Complex):
            return {
                'cirq_type': 'complex',
                'real': o.real,
                'imag': o.imag,
            }

        # Numpy object?
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()

        # Pandas object?
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


def _cirq_object_hook(d, resolvers: Sequence[JsonResolver], context_map: Dict[str, Any]):
    if 'cirq_type' not in d:
        return d

    if d['cirq_type'] == '_SerializedKey':
        return _SerializedKey.read_from_context(context_map, **d)

    if d['cirq_type'] == '_SerializedContext':
        _SerializedContext.update_context(context_map, **d)
        return None

    if d['cirq_type'] == '_ContextualSerialization':
        return _ContextualSerialization.deserialize_with_context(**d)

    for resolver in resolvers:
        cls = resolver(d['cirq_type'])
        if cls is not None:
            break
    else:
        raise ValueError(
            "Could not resolve type '{}' during deserialization".format(d['cirq_type'])
        )

    from_json_dict = getattr(cls, '_from_json_dict_', None)
    if from_json_dict is not None:
        return from_json_dict(**d)

    del d['cirq_type']
    return cls(**d)


class SerializableByKey(SupportsJSON):
    """Protocol for objects that can be serialized to a key + context."""

    @doc_private
    def _serialization_name_(self) -> str:
        """Returns a human-readable string identifier for this object.

        This identifier need not be globally unique; a unique suffix will be
        dynamically generated to create the serialization key for this object.
        """


class _SerializedKey(SupportsJSON):
    """Internal object for holding a SerializableByKey key.

    This is a private type used in contextual serialization. Its deserialization
    is context-dependent, and is not expected to match the original; in other
    words, `cls._from_json_dict_(obj._json_dict_())` does not return
    the original `obj` for this type.
    """

    def __init__(self, key: str):
        self.key = key

    def _json_dict_(self):
        return obj_to_dict_helper(self, ['key'])

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        raise TypeError(f'Internal error: {cls} should never deserialize with _from_json_dict_.')

    @classmethod
    def read_from_context(cls, context_map, key, **kwargs):
        return context_map[key]


class _SerializedContext(SupportsJSON):
    """Internal object for a single SerializableByKey key-to-object mapping.

    This is a private type used in contextual serialization. Its deserialization
    is context-dependent, and is not expected to match the original; in other
    words, `cls._from_json_dict_(obj._json_dict_())` does not return
    the original `obj` for this type.
    """

    def __init__(self, obj: SerializableByKey, suffix: int):
        self.key = f'{obj._serialization_name_()}_{suffix}'
        self.obj = obj

    def _json_dict_(self):
        return obj_to_dict_helper(self, ['key', 'obj'])

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        raise TypeError(f'Internal error: {cls} should never deserialize with _from_json_dict_.')

    @classmethod
    def update_context(cls, context_map, key, obj, **kwargs):
        context_map.update({key: obj})


class _ContextualSerialization(SupportsJSON):
    """Internal object for serializing an object with its context.

    This is a private type used in contextual serialization. Its deserialization
    is context-dependent, and is not expected to match the original; in other
    words, `cls._from_json_dict_(obj._json_dict_())` does not return
    the original `obj` for this type.
    """

    def __init__(self, obj: Any):
        # Context information and the wrapped object are stored together in
        # `object_dag` to ensure consistent serialization ordering.
        self.object_dag = []
        context = []
        for sbk in get_serializable_by_keys(obj):
            if sbk not in context:
                context.append(sbk)
                new_sc = _SerializedContext(sbk, len(context))
                self.object_dag.append(new_sc)
        self.object_dag += [obj]

    def _json_dict_(self):
        return obj_to_dict_helper(self, ['object_dag'])

    @classmethod
    def _from_json_dict_(cls, **kwargs):
        raise TypeError(f'Internal error: {cls} should never deserialize with _from_json_dict_.')

    @classmethod
    def deserialize_with_context(cls, object_dag, **kwargs):
        # The last element of object_dag is the object to be deserialized.
        return object_dag[-1]


def has_serializable_by_keys(obj: Any) -> bool:
    """Returns true if obj contains one or more SerializableByKey objects."""
    if hasattr(obj, '_serialization_name_'):
        return True
    json_dict = getattr(obj, '_json_dict_', lambda: None)()
    if isinstance(json_dict, Dict):
        return any(has_serializable_by_keys(v) for v in json_dict.values())

    # Handle primitive container types.
    if isinstance(obj, Dict):
        return any(has_serializable_by_keys(elem) for pair in obj.items() for elem in pair)
    if hasattr(obj, '__iter__') and not isinstance(obj, str):
        return any(has_serializable_by_keys(elem) for elem in obj)
    return False


def get_serializable_by_keys(obj: Any) -> List[SerializableByKey]:
    """Returns all SerializableByKeys contained by obj.

    Objects are ordered such that nested objects appear before the object they
    are nested inside. This is required to ensure
    """
    result = []
    if hasattr(obj, '_serialization_name_'):
        result.append(obj)
    json_dict = getattr(obj, '_json_dict_', lambda: None)()
    if isinstance(json_dict, Dict):
        for v in json_dict.values():
            result = get_serializable_by_keys(v) + result
    if result:
        return result

    # Handle primitive container types.
    if isinstance(obj, Dict):
        return [sbk for pair in obj.items() for sbk in get_serializable_by_keys(pair)]
    if hasattr(obj, '__iter__') and not isinstance(obj, str):
        return [sbk for v in obj for sbk in get_serializable_by_keys(v)]
    return []


# pylint: disable=function-redefined
@overload
def to_json(
    obj: Any, file_or_fn: Union[IO, pathlib.Path, str], *, indent=2, cls=CirqEncoder
) -> None:
    pass


@overload
def to_json(obj: Any, file_or_fn: None = None, *, indent=2, cls=CirqEncoder) -> str:
    pass


def to_json(
    obj: Any,
    file_or_fn: Union[None, IO, pathlib.Path, str] = None,
    *,
    indent: int = 2,
    cls: Type[json.JSONEncoder] = CirqEncoder,
) -> Optional[str]:
    """Write a JSON file containing a representation of obj.

    The object may be a cirq object or have data members that are cirq
    objects which implement the SupportsJSON protocol.

    Args:
        obj: An object which can be serialized to a JSON representation.
        file_or_fn: A filename (if a string or `pathlib.Path`) to write to, or
            an IO object (such as a file or buffer) to write to, or `None` to
            indicate that the method should return the JSON text as its result.
            Defaults to `None`.
        indent: Pretty-print the resulting file with this indent level.
            Passed to json.dump.
        cls: Passed to json.dump; the default value of CirqEncoder
            enables the serialization of Cirq objects which implement
            the SupportsJSON protocol. To support serialization of 3rd
            party classes, prefer adding the _json_dict_ magic method
            to your classes rather than overriding this default.
    """
    if has_serializable_by_keys(obj):
        obj = _ContextualSerialization(obj)

        class ContextualEncoder(cls):  # type: ignore
            """An encoder with a context map for concise serialization."""

            # These lists populate gradually during serialization. An object
            # with components defined in 'context' will represent those
            # components using their keys instead of inline definition.
            seen: Set[str] = set()

            def default(self, o):
                if not hasattr(o, '_serialization_name_'):
                    return super().default(o)
                for candidate in obj.object_dag[:-1]:
                    if candidate.obj == o:
                        if not candidate.key in ContextualEncoder.seen:
                            ContextualEncoder.seen.add(candidate.key)
                            return candidate.obj._json_dict_()
                        else:
                            return _SerializedKey(candidate.key)._json_dict_()
                raise ValueError("Object mutated during serialization.")  # coverage: ignore

        cls = ContextualEncoder

    if file_or_fn is None:
        return json.dumps(obj, indent=indent, cls=cls)

    if isinstance(file_or_fn, (str, pathlib.Path)):
        with open(file_or_fn, 'w') as actually_a_file:
            json.dump(obj, actually_a_file, indent=indent, cls=cls)
            return None

    json.dump(obj, file_or_fn, indent=indent, cls=cls)
    return None


# pylint: enable=function-redefined


def read_json(
    file_or_fn: Union[None, IO, pathlib.Path, str] = None,
    *,
    json_text: Optional[str] = None,
    resolvers: Optional[Sequence[JsonResolver]] = None,
):
    """Read a JSON file that optionally contains cirq objects.

    Args:
        file_or_fn: A filename (if a string or `pathlib.Path`) to read from, or
            an IO object (such as a file or buffer) to read from, or `None` to
            indicate that `json_text` argument should be used. Defaults to
            `None`.
        json_text: A string representation of the JSON to parse the object from,
            or else `None` indicating `file_or_fn` should be used. Defaults to
            `None`.
        resolvers: A list of functions that are called in order to turn
            the serialized `cirq_type` string into a constructable class.
            By default, top-level cirq objects that implement the SupportsJSON
            protocol are supported. You can extend the list of supported types
            by pre-pending custom resolvers. Each resolver should return `None`
            to indicate that it cannot resolve the given cirq_type and that
            the next resolver should be tried.
    """
    if (file_or_fn is None) == (json_text is None):
        raise ValueError('Must specify ONE of "file_or_fn" or "json".')

    if resolvers is None:
        resolvers = DEFAULT_RESOLVERS

    context_map: Dict[str, 'SerializableByKey'] = {}

    def obj_hook(x):
        return _cirq_object_hook(x, resolvers, context_map)

    if json_text is not None:
        return json.loads(json_text, object_hook=obj_hook)

    if isinstance(file_or_fn, (str, pathlib.Path)):
        with open(file_or_fn, 'r') as file:
            return json.load(file, object_hook=obj_hook)

    return json.load(cast(IO, file_or_fn), object_hook=obj_hook)


def to_json_gzip(
    obj: Any,
    file_or_fn: Union[None, IO, pathlib.Path, str] = None,
    *,
    indent: int = 2,
    cls: Type[json.JSONEncoder] = CirqEncoder,
) -> Optional[bytes]:
    json_str = to_json(obj, indent=indent, cls=cls)
    if isinstance(file_or_fn, (str, pathlib.Path)):
        with gzip.open(file_or_fn, 'wt', encoding='utf-8') as actually_a_file:
            actually_a_file.write(json_str)
            return None

    gzip_data = gzip.compress(bytes(json_str, encoding='utf-8'))  # type: ignore
    if file_or_fn is None:
        return gzip_data

    file_or_fn.write(gzip_data)
    return None


def read_json_gzip(
    file_or_fn: Union[None, IO, pathlib.Path, str] = None,
    *,
    gzip_raw: Optional[bytes] = None,
    resolvers: Optional[Sequence[JsonResolver]] = None,
):
    if (file_or_fn is None) == (gzip_raw is None):
        raise ValueError('Must specify ONE of "file_or_fn" or "gzip_raw".')

    if gzip_raw is not None:
        json_str = gzip.decompress(gzip_raw).decode(encoding='utf-8')
        return read_json(json_text=json_str, resolvers=resolvers)

    with gzip.open(file_or_fn, 'rt') as json_file:  # type: ignore
        return read_json(cast(IO, json_file), resolvers=resolvers)
