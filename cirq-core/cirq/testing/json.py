# Copyright 2020 The Cirq Developers
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
import inspect
import io
import pathlib
from dataclasses import dataclass
from types import ModuleType
from typing import Dict, Iterator, List, Set, Tuple, Type

import numpy as np
import pandas as pd

import cirq
from cirq.protocols.json_serialization import ObjectFactory

# This is the testing framework for json serialization
# The actual tests live in cirq.protocols.json_serialization_test.py.
#
# When registering a new module, it has to come with the following setup:
#  - test data: a new package to contain the test data, see for example
#       cirq/google/json_test_data
#  - test spec: the json_test_data package should export an object named
#       TestSpec, which should be of type ModuleJsonTestSpec, see for example
#       cirq/google/json_test_data/spec.py
#  - resolver cache: a resolver cache for the exposed types - see for example
#       cirq/json_resolver_cache.py or cirq/google/json_resolver_cache.py


@dataclass
class ModuleJsonTestSpec:
    # for test failures, a better representation
    name: str
    # the packages that have the public classes to be checked for serialization
    packages: List[ModuleType]
    # the path for the folder containing the test files
    test_data_path: pathlib.Path
    # these public class names are planned to be serializable but not yet
    not_yet_serializable: List[str]
    # these public class names do not need to be serialized ever
    should_not_be_serialized: List[str]
    # points to the resolver cache's dict for this module
    resolver_cache: Dict[str, ObjectFactory]
    # {DeprecatedClass: deprecation_deadline} pairs to avoid deprecation errors
    # in serialization tests.
    deprecated: Dict[str, str]
    # The unqualified public name is different from the cirq_type field of the json object,
    # usually due to namespacing.
    custom_class_name_to_cirq_type: Dict[str, str] = dataclasses.field(default_factory=dict)
    # Special cases where classes cannot be tested using the normal infrastructure.
    tested_elsewhere: List[str] = dataclasses.field(default_factory=list)

    def __repr__(self):
        return self.name

    def _get_all_public_classes(self) -> Iterator[Tuple[str, Type]]:
        for module in self.packages:
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) or inspect.ismodule(obj):
                    continue

                if name in self.should_not_be_serialized:
                    continue

                if not inspect.isclass(obj):
                    # singletons, for instance
                    obj = obj.__class__

                if name.startswith('_'):
                    continue

                if inspect.isclass(obj) and inspect.isabstract(obj):
                    continue

                name = self.custom_class_name_to_cirq_type.get(name, name)
                yield name, obj

    def find_classes_that_should_serialize(self) -> Set[Tuple[str, Type]]:
        result: Set[Tuple[str, Type]] = set()

        result.update({(name, obj) for name, obj in self._get_all_public_classes()})
        result.update(self.get_resolver_cache_types())

        return result

    def get_resolver_cache_types(self) -> Set[Tuple[str, Type]]:
        result: Set[Tuple[str, Type]] = set()
        for k, v in self.resolver_cache.items():
            if isinstance(v, type):
                result.add((k, v))
        return result

    def get_all_names(self) -> Iterator[str]:
        def not_module_or_function(x):
            return not (inspect.ismodule(x) or inspect.isfunction(x))

        for m in self.packages:
            for name, _ in inspect.getmembers(m, not_module_or_function):
                yield name

        for name, _ in self.get_resolver_cache_types():
            yield name

    def all_test_data_keys(self) -> List[str]:
        seen = set()

        for file in self.test_data_path.iterdir():
            name = str(file.absolute())
            if name.endswith('.json') or name.endswith('.repr'):
                seen.add(name[: -len('.json')])
            elif name.endswith('.json_inward') or name.endswith('.repr_inward'):
                seen.add(name[: -len('.json_inward')])

        return sorted(seen)


def spec_for(module_name: str) -> ModuleJsonTestSpec:
    import importlib.util

    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(f"{module_name} not found")

    test_module_name = f"{module_name}.json_test_data"
    if importlib.util.find_spec(test_module_name) is None:
        raise ValueError(
            f"{module_name} module is missing json_test_data package, please set it up."
        )
    test_module = importlib.import_module(test_module_name)

    if not hasattr(test_module, "TestSpec"):
        raise ValueError(f"{test_module_name} module is missing TestSpec, please set it up.")

    return getattr(test_module, "TestSpec")


def assert_json_roundtrip_works(obj, text_should_be=None, resolvers=None):
    """Tests that the given object can serialized and de-serialized

    Args:
        obj: The object to test round-tripping for.
        text_should_be: An optional argument to assert the JSON serialized
            output.
        resolvers: Any resolvers if testing those other than the default.

    Raises:
        AssertionError: The given object can not be round-tripped according to
            the given arguments.
    """
    buffer = io.StringIO()
    cirq.protocols.to_json(obj, buffer)

    if text_should_be is not None:
        buffer.seek(0)
        text = buffer.read()
        assert text == text_should_be, text

    buffer.seek(0)
    restored_obj = cirq.protocols.read_json(buffer, resolvers=resolvers)
    if isinstance(obj, np.ndarray):
        np.testing.assert_equal(restored_obj, obj)
    elif isinstance(obj, pd.DataFrame):
        pd.testing.assert_frame_equal(restored_obj, obj)
    elif isinstance(obj, pd.Index):
        pd.testing.assert_index_equal(restored_obj, obj)
    else:
        assert restored_obj == obj
