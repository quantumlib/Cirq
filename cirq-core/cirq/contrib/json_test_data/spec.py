# pylint: disable=wrong-or-nonexistent-copyright-notice
from __future__ import annotations

import pathlib

import cirq
from cirq.contrib.json import _class_resolver_dictionary
from cirq.testing.json import ModuleJsonTestSpec

TestSpec = ModuleJsonTestSpec(
    name="cirq.contrib",
    packages=[cirq.contrib],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[],
    should_not_be_serialized=[
        "QuantumVolumeResult",
        "SwapPermutationGate",
        "BayesianNetworkGate",
        "Unique",
        "CircuitDag",
    ],
    resolver_cache={
        k: v
        for k, v in _class_resolver_dictionary().items()
        if k
        not in {
            "QuantumVolumeResult",
            "SwapPermutationGate",
            "BayesianNetworkGate",
            "Unique",
            "CircuitDag",
        }
    },
    deprecated={},
)
