import pathlib
from dataclasses import dataclass
from typing import List

from cirq._import import ModuleType


@dataclass
class JsonSerializationTestSpec:
    # the module that has the public classes to be checked for serialization
    module: ModuleType
    # the path for test files
    test_file_path: pathlib.Path
    # these public class names are exposed but do not need to be serialized
    excluded_class_names: List[str]
