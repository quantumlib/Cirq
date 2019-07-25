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

import string
import json
from typing import TYPE_CHECKING, Union, Any, Tuple, TypeVar, Optional, Dict, \
    Iterable

import numpy as np

from typing_extensions import Protocol

from cirq.type_workarounds import NotImplementedType

TDefault = TypeVar('TDefault')

RaiseTypeErrorIfNotProvided = ([],)  # type: Any


class SupportsJSON(Protocol):
    """An object that can be turned into JSON dictionaries.

    Returning `NotImplemented` or `None` means "don't know how to turn into
    QASM". In that case fallbacks based on decomposition and known unitaries
    will be used instead.
    """

    def _json_dict_(self) -> Union[None, NotImplementedType, Dict[Any, Any]]:
        pass

def to_json_dict(obj, attribute_names):
    d = {'cirq_type': obj.__class__.__name__}
    for attr_name in attribute_names:
        d[attr_name] = getattr(obj, attr_name)
    return d


class CirqEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, '_json_dict_'):
            return o._json_dict_()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int_):
            return o.item()
        return super().default(o)


def cirq_object_hook(d):
    if 'cirq_type' in d:
        import cirq
        cls = getattr(cirq, d['cirq_type'])

        if hasattr(cls, '_from_json_dict_'):
            return cls._from_json_dict_(**d)

        del d['cirq_type']
        return cls(**d)

    return d


def _to_json(obj: Any, file, *, indent):
    json.dump(obj, file, indent=indent, cls=CirqEncoder)


def to_json(obj: Any, file, *, indent=2):
    if isinstance(file, str):
        with open(file, 'w') as actually_a_file:
            return _to_json(obj, actually_a_file, indent=indent)

    return _to_json(obj, file, indent=indent)


def _read_json(file):
    return json.load(file, object_hook=cirq_object_hook)


def read_json(file_or_fn):
    if isinstance(file_or_fn, str):
        with open(file_or_fn, 'r') as file:
            return _read_json(file)

    return _read_json(file_or_fn)
