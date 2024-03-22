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

import pathlib

import cirq
from cirq.json_resolver_cache import _class_resolver_dictionary
from cirq.testing.json import ModuleJsonTestSpec

TestSpec = ModuleJsonTestSpec(
    name="cirq",
    packages=[cirq, cirq.work],
    test_data_path=pathlib.Path(__file__).parent,
    custom_class_name_to_cirq_type={"MSGate": "cirq.MSGate"},
    resolver_cache=_class_resolver_dictionary(),
    not_yet_serializable=[
        'Alignment',
        'AxisAngleDecomposition',
        'CircuitDiagramInfo',
        'CircuitDiagramInfoArgs',
        'CircuitSampleJob',
        'CliffordSimulatorStepResult',
        'CliffordTrialResult',
        'DensityMatrixSimulator',
        'DensityMatrixStepResult',
        'DensityMatrixTrialResult',
        'ExpressionMap',
        'InsertStrategy',
        'KakDecomposition',
        'LinearCombinationOfGates',
        'LinearCombinationOfOperations',
        'PauliSumCollector',
        'PauliSumExponential',
        'PeriodicValue',
        'PointOptimizationSummary',
        'QasmArgs',
        'QasmOutput',
        'QuantumState',
        'QubitOrder',
        'SimulationTrialResult',
        'SimulationTrialResultBase',
        'SparseSimulatorStep',
        'StateVectorMixin',
        'TextDiagramDrawer',
        'Timestamp',
        'TwoQubitGateTabulationResult',
        'StateVectorTrialResult',
        'ZerosSampler',
    ],
    should_not_be_serialized=[
        'ClassicalStateSimulator',
        # Heatmaps
        'Heatmap',
        'TwoQubitInteractionHeatmap',
        # Intermediate states with work buffers and unknown external prng guts.
        'ApplyChannelArgs',
        'ApplyMixtureArgs',
        'ApplyUnitaryArgs',
        'CliffordTableauSimulationState',
        'DensityMatrixSimulationState',
        'SimulationProductState',
        'SimulationState',
        'SimulationStateBase',
        'StabilizerChFormSimulationState',
        'StabilizerSimulationState',
        'StateVectorSimulationState',
        # Abstract base class for creating compilation targets.
        'CompilationTargetGateset',
        'TwoQubitCompilationTargetGateset',
        # Circuit optimizers are function-like. Only attributes
        # are ignore_failures, tolerance, and other feature flags
        'MEASUREMENT_KEY_SEPARATOR',
        'PointOptimizer',
        # Transformers
        'DecompositionContext',
        'TransformerLogger',
        'TransformerContext',
        # Routing utilities
        'HardCodedInitialMapper',
        'LineInitialMapper',
        'MappingManager',
        'RouteCQC',
        # Qubit Managers,
        'SimpleQubitManager',
        'GreedyQubitManager',
        # global objects
        'CONTROL_TAG',
        'PAULI_BASIS',
        'PAULI_STATES',
        # abstract, but not inspect.isabstract():
        'Device',
        'InterchangeableQubitsGate',
        'Pauli',
        'ABCMetaImplementAnyOneOf',
        'SimulatesAmplitudes',
        'SimulatesExpectationValues',
        'SimulatesFinalState',
        'StateVectorStepResult',
        'StepResultBase',
        'UnitSweep',
        'UNIT_SWEEP',
        'NamedTopology',
        # protocols:
        'HasJSONNamespace',
        'SupportsActOn',
        'SupportsActOnQubits',
        'SupportsApplyChannel',
        'SupportsApplyMixture',
        'SupportsApproximateEquality',
        'SupportsCircuitDiagramInfo',
        'SupportsCommutes',
        'SupportsConsistentApplyUnitary',
        'SupportsControlKey',
        'SupportsDecompose',
        'SupportsDecomposeWithQubits',
        'SupportsEqualUpToGlobalPhase',
        'SupportsExplicitHasUnitary',
        'SupportsExplicitNumQubits',
        'SupportsExplicitQidShape',
        'SupportsJSON',
        'SupportsKraus',
        'SupportsMeasurementKey',
        'SupportsMixture',
        'SupportsParameterization',
        'SupportsPauliExpansion',
        'SupportsPhase',
        'SupportsQasm',
        'SupportsQasmWithArgs',
        'SupportsQasmWithArgsAndQubits',
        'SupportsTraceDistanceBound',
        'SupportsUnitary',
        # mypy types:
        'CIRCUIT_LIKE',
        'DURATION_LIKE',
        'JsonResolver',
        'LabelEntity',
        'NOISE_MODEL_LIKE',
        'OP_TREE',
        'PAULI_GATE_LIKE',
        'PAULI_STRING_LIKE',
        'ParamResolverOrSimilarType',
        'PauliSumLike',
        'QUANTUM_STATE_LIKE',
        'QubitOrderOrList',
        'RANDOM_STATE_OR_SEED_LIKE',
        'STATE_VECTOR_LIKE',
        'Sweepable',
        'TParamKey',
        'TParamVal',
        'TParamValComplex',
        'TRANSFORMER',
        'ParamDictType',
        'ParamMappingType',
        # utility:
        'CliffordSimulator',
        'NoiseModelFromNoiseProperties',
        'Simulator',
        'StabilizerSampler',
        'DEFAULT_RESOLVERS',
    ],
    deprecated={},
    tested_elsewhere=[
        # SerializableByKey does not follow common serialization rules.
        # It is tested separately in test_context_serialization.
        'SerializableByKey'
    ],
)
