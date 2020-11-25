import pathlib
from dataclasses import dataclass
from typing import List, Type, Dict

from cirq._import import ModuleType


@dataclass
class JsonSerializationTestSpec:
    # for test names
    name: str
    # the module that has the public classes to be checked for serialization
    modules: List[ModuleType]
    # the path for test files
    test_file_path: pathlib.Path
    # these public class names are exposed but
    not_yet_serializable: List[str]
    # these public class names are exposed but do not need to be serialized
    shouldnt_be_serialized: List[str]
    # points to the resolver cache's dict for this module
    resolver_cache: Dict[str, Type]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
