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
import datetime
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
    Tuple,
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


def _register_resolver(dict_factory: Callable[[], Dict[str, ObjectFactory]]) -> None:
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

    The magic method `_json_dict_` must return a trivially json-serializable
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


class HasJSONNamespace(Protocol):
    """An object which prepends a namespace to its JSON cirq_type.

    Classes which implement this method have the following cirq_type format:

        f"{obj._json_namespace_()}.{obj.__class__.__name__}

    Classes outside of Cirq or its submodules MUST implement this method to be
    used in type serialization.
    """

    @doc_private
    @classmethod
    def _json_namespace_(cls) -> str:
        pass


def obj_to_dict_helper(obj: Any, attribute_names: Iterable[str]) -> Dict[str, Any]:
    """Construct a dictionary containing attributes from obj

    This is useful as a helper function in objects implementing the
    SupportsJSON protocol, particularly in the `_json_dict_` method.

    In addition to keys and values specified by `attribute_names`, the
    returned dictionary has an additional key "cirq_type" whose value
    is the string name of the type of `obj`.

    Args:
        obj: A python object with attributes to be placed in the dictionary.
        attribute_names: The names of attributes to serve as keys in the
            resultant dictionary. The values will be the attribute values.
    """
    d = {}
    for attr_name in attribute_names:
        d[attr_name] = getattr(obj, attr_name)
    return d


# pylint: enable=redefined-builtin
def dataclass_json_dict(obj: Any) -> Dict[str, Any]:
    """Return a dictionary suitable for `_json_dict_` from a dataclass.

    Dataclasses keep track of their relevant fields, so we can automatically generate these.

    Dataclasses are implemented with somewhat complex metaprogramming, and tooling (PyCharm, mypy)
    have special cases for dealing with classes decorated with @dataclass. There is very little
    support (and no plans for support) for decorators that wrap @dataclass (like
    @cirq.json_serializable_dataclass) or combining additional decorators with @dataclass.
    Although not as elegant, you may want to consider explicitly defining `_json_dict_` on your
    dataclasses which simply `return dataclass_json_dict(self)`.
    """
    attribute_names = [f.name for f in dataclasses.fields(obj)]
    return obj_to_dict_helper(obj, attribute_names)


def _json_dict_with_cirq_type(obj: Any):
    base_dict = obj._json_dict_()
    if 'cirq_type' in base_dict:
        raise ValueError(
            f"Found 'cirq_type': '{base_dict['cirq_type']}' in user-specified _json_dict_. "
            "'cirq_type' is now automatically generated from the class's name and its "
            "_json_namespace_ method as `cirq_type: '[<namespace>.]<class_name>'`."
            "\n\n"
            "Starting in v0.15, custom 'cirq_type' values will trigger an error. "
            "To fix this, remove 'cirq_type' from the class _json_dict_ method and "
            "define _json_namespace_ for the class."
            "\n\n"
            "For backwards compatibility, third-party classes whose old 'cirq_type' value "
            "does not match the new value must appear under BOTH values in the resolver "
            "for that package. For details on defining custom resolvers, see the "
            "DEFAULT_RESOLVER docstring in cirq-core/cirq/protocols/json_serialization.py."
        )
    return {'cirq_type': json_cirq_type(type(obj)), **base_dict}


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
            return _json_dict_with_cirq_type(o)

        # Sympy object? (Must come before general number checks.)
        # TODO: More support for sympy
        # Github issue: https://github.com/quantumlib/Cirq/issues/2014

        if isinstance(o, sympy.Symbol):
            return {'cirq_type': 'sympy.Symbol', 'name': o.name}

        if isinstance(
            o,
            (
                sympy.Add,
                sympy.Mul,
                sympy.Pow,
                sympy.GreaterThan,
                sympy.StrictGreaterThan,
                sympy.LessThan,
                sympy.StrictLessThan,
                sympy.Equality,
                sympy.Unequality,
            ),
        ):
            return {'cirq_type': f'sympy.{o.__class__.__name__}', 'args': o.args}

        if isinstance(o, sympy.Integer):
            return {'cirq_type': 'sympy.Integer', 'i': o.p}

        if isinstance(o, sympy.Float):
            return {'cirq_type': 'sympy.Float', 'approx': float(o)}

        if isinstance(o, sympy.Rational):
            return {'cirq_type': 'sympy.Rational', 'p': o.p, 'q': o.q}

        if isinstance(o, sympy.NumberSymbol):
            # check if `o` is a numeric symbol,
            # i.e. one of the transcendental numbers
            # sympy.pi, sympy.E or sympy.EulerGamma
            # (note that these are singletons).
            if o is sympy.pi:
                return {'cirq_type': 'sympy.pi'}
            if o is sympy.E:
                return {'cirq_type': 'sympy.E'}
            if o is sympy.EulerGamma:
                return {'cirq_type': 'sympy.EulerGamma'}

        # A basic number object?
        if isinstance(o, numbers.Integral):
            return int(o)
        if isinstance(o, numbers.Real):
            return float(o)
        if isinstance(o, numbers.Complex):
            return {'cirq_type': 'complex', 'real': o.real, 'imag': o.imag}

        # Numpy object?
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()

        # Pandas object?
        if isinstance(o, pd.MultiIndex):
            return {'cirq_type': 'pandas.MultiIndex', 'tuples': list(o), 'names': list(o.names)}
        if isinstance(o, pd.Index):
            return {'cirq_type': 'pandas.Index', 'data': list(o), 'name': o.name}
        if isinstance(o, pd.DataFrame):
            cols = [o[col].tolist() for col in o.columns]
            rows = list(zip(*cols))
            return {
                'cirq_type': 'pandas.DataFrame',
                'data': rows,
                'columns': o.columns,
                'index': o.index,
            }

        # datetime
        if isinstance(o, datetime.datetime):
            return {'cirq_type': 'datetime.datetime', 'timestamp': o.timestamp()}

        return super().default(o)  # pragma: no cover


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

    cls = factory_from_json(d['cirq_type'], resolvers=resolvers)
    from_json_dict = getattr(cls, '_from_json_dict_', None)
    if from_json_dict is not None:
        return from_json_dict(**d)

    del d['cirq_type']
    return cls(**d)


class SerializableByKey(SupportsJSON):
    """Protocol for objects that can be serialized to a key + context.

    In serialization, objects that inherit from this type will only be fully
    defined once (the "context"). Thereafter, a unique integer key will be used
    to identify that object.
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

    def __init__(self, obj: SerializableByKey, uid: int):
        self.key = uid
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
    if isinstance(obj, SerializableByKey):
        return True
    json_dict = getattr(obj, '_json_dict_', lambda: None)()
    if isinstance(json_dict, Dict):
        return any(has_serializable_by_keys(v) for v in json_dict.values())

    # Handle primitive container types.
    if isinstance(obj, Dict):
        return any(has_serializable_by_keys(elem) for pair in obj.items() for elem in pair)

    if hasattr(obj, '__iter__') and not isinstance(obj, str):
        # Return False on TypeError because some numpy values
        # (like np.array(1)) have iterable methods
        # yet return a TypeError when there is an attempt to iterate over them
        try:
            return any(has_serializable_by_keys(elem) for elem in obj)
        except TypeError:
            return False
    return False


def get_serializable_by_keys(obj: Any) -> List[SerializableByKey]:
    """Returns all SerializableByKeys contained by obj.

    Objects are ordered such that nested objects appear before the object they
    are nested inside. This is required to ensure SerializableByKeys are only
    fully defined once in serialization.
    """
    result = []
    if isinstance(obj, SerializableByKey):
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


def json_namespace(type_obj: Type) -> str:
    """Returns a namespace for JSON serialization of `type_obj`.

    Types can provide custom namespaces with `_json_namespace_`; otherwise, a
    Cirq type will not include a namespace in its cirq_type. Non-Cirq types
    must provide a namespace for serialization in Cirq.

    Args:
        type_obj: Type to retrieve the namespace from.

    Returns:
        The namespace to prepend `type_obj` with in its JSON cirq_type.

    Raises:
        ValueError: if `type_obj` is not a Cirq type and does not explicitly
            define its namespace with _json_namespace_.
    """
    if hasattr(type_obj, '_json_namespace_'):
        return type_obj._json_namespace_()
    if type_obj.__module__.startswith('cirq'):
        return ''
    raise ValueError(f'{type_obj} is not a Cirq type, and does not define _json_namespace_.')


def json_cirq_type(type_obj: Type) -> str:
    """Returns a string type for JSON serialization of `type_obj`.

    This method is not part of the base serialization path. Together with
    `cirq_type_from_json`, it can be used to provide type-object serialization
    for classes that need it.
    """
    namespace = json_namespace(type_obj)
    if namespace:
        return f'{namespace}.{type_obj.__name__}'
    return type_obj.__name__


def factory_from_json(
    type_str: str, resolvers: Optional[Sequence[JsonResolver]] = None
) -> ObjectFactory:
    """Returns a factory for constructing objects of type `type_str`.

    DEFAULT_RESOLVERS is updated dynamically as cirq submodules are imported.

    Args:
        type_str: string representation of the type to deserialize.
        resolvers: list of JsonResolvers to use in type resolution. If this is
            left blank, DEFAULT_RESOLVERS will be used.

    Returns:
        An ObjectFactory that can be called to construct an object whose type
        matches the name `type_str`.

    Raises:
        ValueError: if type_str does not have a match in `resolvers`.
    """
    resolvers = resolvers if resolvers is not None else DEFAULT_RESOLVERS
    for resolver in resolvers:
        cirq_type = resolver(type_str)
        if cirq_type is not None:
            return cirq_type
    raise ValueError(f"Could not resolve type '{type_str}' during deserialization")


def cirq_type_from_json(type_str: str, resolvers: Optional[Sequence[JsonResolver]] = None) -> Type:
    """Returns a type object for JSON deserialization of `type_str`.

    This method is not part of the base deserialization path. Together with
    `json_cirq_type`, it can be used to provide type-object deserialization
    for classes that need it.

    Args:
        type_str: string representation of the type to deserialize.
        resolvers: list of JsonResolvers to use in type resolution. If this is
            left blank, DEFAULT_RESOLVERS will be used.

    Returns:
        The type object T for which json_cirq_type(T) matches `type_str`.

    Raises:
        ValueError: if type_str does not have a match in `resolvers`, or if the
            match found is a factory method instead of a type.
    """
    cirq_type = factory_from_json(type_str, resolvers)
    if isinstance(cirq_type, type):
        return cirq_type
    # We assume that if factory_from_json returns a factory, there is not
    # another resolver which resolves `type_str` to a type object.
    raise ValueError(f"Type {type_str} maps to a factory method instead of a type.")


# pylint: disable=function-redefined
@overload
def to_json(
    obj: Any,
    file_or_fn: Union[IO, pathlib.Path, str],
    *,
    indent=2,
    separators=None,
    cls=CirqEncoder,
) -> None:
    pass


@overload
def to_json(
    obj: Any, file_or_fn: None = None, *, indent=2, separators=None, cls=CirqEncoder
) -> str:
    pass


def to_json(
    obj: Any,
    file_or_fn: Union[None, IO, pathlib.Path, str] = None,
    *,
    indent: Optional[int] = 2,
    separators: Optional[Tuple[str, str]] = None,
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
        separators: Passed to json.dump; key-value pairs delimiters defined as
            `(item_separator, key_separators)` tuple. Note that any non-standard
            operators (':', ',') will cause `read_json` to fail.
        cls: Passed to json.dump; the default value of CirqEncoder
            enables the serialization of Cirq objects which implement
            the SupportsJSON protocol. To support serialization of 3rd
            party classes, prefer adding the `_json_dict_` magic method
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
                if not isinstance(o, SerializableByKey):
                    return super().default(o)
                for candidate in obj.object_dag[:-1]:
                    if candidate.obj == o:
                        if not candidate.key in ContextualEncoder.seen:
                            ContextualEncoder.seen.add(candidate.key)
                            return _json_dict_with_cirq_type(candidate.obj)
                        else:
                            return _json_dict_with_cirq_type(_SerializedKey(candidate.key))
                raise ValueError("Object mutated during serialization.")  # pragma: no cover

        cls = ContextualEncoder

    if file_or_fn is None:
        return json.dumps(obj, indent=indent, separators=separators, cls=cls)

    if isinstance(file_or_fn, (str, pathlib.Path)):
        with open(file_or_fn, 'w') as actually_a_file:
            json.dump(obj, actually_a_file, indent=indent, cls=cls)
            return None

    json.dump(obj, file_or_fn, indent=indent, separators=separators, cls=cls)
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

    Raises:
        ValueError: If either none of `file_or_fn` and `json_text` is specified,
            or both are specified.
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
    """Write a gzipped JSON file containing a representation of obj.

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
    json_str = to_json(obj, indent=indent, cls=cls)
    if isinstance(file_or_fn, (str, pathlib.Path)):
        with gzip.open(file_or_fn, 'wt', encoding='utf-8') as actually_a_file:
            actually_a_file.write(json_str)
            return None

    gzip_data = gzip.compress(bytes(json_str, encoding='utf-8'))
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
    """Read a gzipped JSON file that optionally contains cirq objects.

    Args:
        file_or_fn: A filename (if a string or `pathlib.Path`) to read from, or
            an IO object (such as a file or buffer) to read from, or `None` to
            indicate that `gzip_raw` argument should be used. Defaults to
            `None`.
        gzip_raw: Bytes representing the raw gzip input to unzip and parse
            or else `None` indicating `file_or_fn` should be used. Defaults to
            `None`.
        resolvers: A list of functions that are called in order to turn
            the serialized `cirq_type` string into a constructable class.
            By default, top-level cirq objects that implement the SupportsJSON
            protocol are supported. You can extend the list of supported types
            by pre-pending custom resolvers. Each resolver should return `None`
            to indicate that it cannot resolve the given cirq_type and that
            the next resolver should be tried.

    Raises:
        ValueError: If either none of `file_or_fn` and `gzip_raw` is specified,
            or both are specified.
    """
    if (file_or_fn is None) == (gzip_raw is None):
        raise ValueError('Must specify ONE of "file_or_fn" or "gzip_raw".')

    if gzip_raw is not None:
        json_str = gzip.decompress(gzip_raw).decode(encoding='utf-8')
        return read_json(json_text=json_str, resolvers=resolvers)

    with gzip.open(file_or_fn, 'rt') as json_file:  # type: ignore
        return read_json(cast(IO, json_file), resolvers=resolvers)
