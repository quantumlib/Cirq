import pathlib

import cirq
from cirq.google.json_resolver_cache import RESOLVER_CACHE

from cirq.testing.json_serialization_test_spec import JsonSerializationTestSpec

TestSpec = JsonSerializationTestSpec(
    name="cirq.google",
    modules=[cirq.google],
    test_file_path=pathlib.Path(__file__).parent,
    not_yet_serializable=[
        'Calibration', 'Sycamore23', 'XMON', 'FSIM_GATESET', 'SYC_GATESET',
        'Sycamore', 'CalibrationLayer', 'SerializableDevice',
        'SerializableGateSet', 'CalibrationResult', 'SQRT_ISWAP_GATESET',
        'XmonDevice', 'SerializableGateSet', 'SerializableDevice', 'XMON'
    ],
    shouldnt_be_serialized=[
        'Engine',
        'EngineJob',
        'EngineProcessor',
        'EngineProgram',
        'EngineTimeSlot',
        'QuantumEngineSampler',
        'NAMED_GATESETS',
        'ConvertToSqrtIswapGates',
        'ProtoVersion',
        'GateOpSerializer',
        'ConvertToXmonGates',
        'ConvertToSycamoreGates',
        'DeserializingArg',
        'AnnealSequenceSearchStrategy',
        'SerializingArg',
        'GateOpDeserializer',
        'GreedySequenceSearchStrategy',
    ],
    resolver_cache=RESOLVER_CACHE.class_resolver_dictionary)
