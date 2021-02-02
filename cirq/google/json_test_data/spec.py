import pathlib

import cirq
from cirq.google.json_resolver_cache import _class_resolver_dictionary

from cirq.testing.json import ModuleJsonTestSpec

TestSpec = ModuleJsonTestSpec(
    name="cirq.google",
    packages=[cirq.google],
    test_data_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[
        'FSIM_GATESET',
        'SYC_GATESET',
        'Sycamore',
        'Sycamore23',
        'SerializableDevice',
        'SerializableGateSet',
        'SQRT_ISWAP_GATESET',
        'SQRT_ISWAP_PARAMETERS',
        'ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION',
        'WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION',
        'XmonDevice',
        'XMON',
    ],
    should_not_be_serialized=[
        'AnnealSequenceSearchStrategy',
        'ConvertToSqrtIswapGates',
        'ConvertToSycamoreGates',
        'ConvertToXmonGates',
        'DeserializingArg',
        'Engine',
        'EngineJob',
        'EngineProcessor',
        'EngineProgram',
        'EngineTimeSlot',
        'NAMED_GATESETS',
        'ProtoVersion',
        'GateOpSerializer',
        'GateOpDeserializer',
        'GreedySequenceSearchStrategy',
        'PhasedFSimEngineSimulator',
        'SerializingArg',
        'QuantumEngineSampler',
    ],
    resolver_cache=_class_resolver_dictionary(),
)
