import pathlib

import cirq
from cirq.google.json_resolver_cache import _class_resolver_dictionary

from cirq.testing.json import ModuleJsonTestSpec

TestSpec = ModuleJsonTestSpec(name="cirq.google",
                              packages=[cirq.google],
                              test_data_path=pathlib.Path(__file__).parent,
                              not_yet_serializable=[
                                  'Calibration', 'CalibrationResult',
                                  'CalibrationLayer', 'FSIM_GATESET',
                                  'SYC_GATESET', 'Sycamore', 'Sycamore23',
                                  'SerializableDevice', 'SerializableGateSet',
                                  'SQRT_ISWAP_GATESET', 'XmonDevice', 'XMON'
                              ],
                              should_not_be_serialized=[
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
                              resolver_cache=_class_resolver_dictionary())
