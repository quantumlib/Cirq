from __future__ import annotations

import pathlib

import cirq
from cirq.testing.json import ModuleJsonTestSpec
from cirq.contrib.json import _class_resolver_dictionary

TestSpec = ModuleJsonTestSpec(
    name="cirq.contrib",
    packages=[cirq.contrib],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[],
    should_not_be_serialized=[],
    resolver_cache=_class_resolver_dictionary(),
    deprecated={},
)
