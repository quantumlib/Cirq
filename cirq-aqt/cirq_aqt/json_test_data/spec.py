import pathlib

import cirq_aqt
from cirq_aqt.json_resolver_cache import _class_resolver_dictionary

from cirq.testing.json import ModuleJsonTestSpec

TestSpec = ModuleJsonTestSpec(
    name="cirq_aqt",
    packages=[cirq_aqt],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[],
    should_not_be_serialized=['AQTSimulator', 'AQTSampler', 'AQTSamplerLocalSimulator'],
    resolver_cache=_class_resolver_dictionary(),
    deprecated={},
)
