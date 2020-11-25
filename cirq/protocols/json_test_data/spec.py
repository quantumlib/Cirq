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
    resolver_cache=_class_resolver_dictionary(),
    not_yet_serializable=[
        'AxisAngleDecomposition',
        'CircuitDag',
        'CircuitDiagramInfo',
        'CircuitDiagramInfoArgs',
        'CircuitSampleJob',
        'CliffordSimulatorStepResult',
        'CliffordTrialResult',
        'DensityMatrixSimulator',
        'DensityMatrixSimulatorState',
        'DensityMatrixStepResult',
        'DensityMatrixTrialResult',
        'ExpressionMap',
        'Heatmap',
        'InsertStrategy',
        'IonDevice',
        'KakDecomposition',
        'LinearCombinationOfGates',
        'LinearCombinationOfOperations',
        'Linspace',
        'ListSweep',
        'NeutralAtomDevice',
        'ParallelGateOperation',
        'PauliInteractionGate',
        'PauliStringPhasor',
        'PauliSum',
        'PauliSumCollector',
        'PauliTransform',
        'PeriodicValue',
        'PointOptimizationSummary',
        'Points',
        'Product',
        'QasmArgs',
        'QasmOutput',
        'QubitOrder',
        'QubitPermutationGate',
        'QuilFormatter',
        'QuilOutput',
        'SimulationTrialResult',
        'SingleQubitCliffordGate',
        'SparseSimulatorStep',
        'StateVectorMixin',
        'TextDiagramDrawer',
        'ThreeQubitDiagonalGate',
        'Timestamp',
        'TwoQubitDiagonalGate',
        'UnitSweep',
        'StateVectorSimulatorState',
        'StateVectorTrialResult',
        'WaveFunctionSimulatorState',
        'WaveFunctionTrialResult',
        'ZerosSampler',
        'Zip',
    ],
    should_not_be_serialized=[
        # Intermediate states with work buffers and unknown external prng guts.
        'ActOnCliffordTableauArgs',
        'ActOnStabilizerCHFormArgs',
        'ActOnStateVectorArgs',
        'ApplyChannelArgs',
        'ApplyMixtureArgs',
        'ApplyUnitaryArgs',

        # Circuit optimizers are function-like. Only attributes
        # are ignore_failures, tolerance, and other feature flags
        'ConvertToCzAndSingleGates',
        'ConvertToIonGates',
        'ConvertToNeutralAtomGates',
        'DropEmptyMoments',
        'DropNegligible',
        'EjectPhasedPaulis',
        'EjectZ',
        'ExpandComposite',
        'MergeInteractions',
        'MergeSingleQubitGates',
        'PointOptimizer',
        'SynchronizeTerminalMeasurements',

        # global objects
        'CONTROL_TAG',
        'PAULI_BASIS',
        'PAULI_STATES',

        # abstract, but not inspect.isabstract():
        'Device',
        'InterchangeableQubitsGate',
        'Pauli',
        'SingleQubitGate',
        'ThreeQubitGate',
        'TwoQubitGate',
        'ABCMetaImplementAnyOneOf',

        # protocols:
        'SupportsActOn',
        'SupportsApplyChannel',
        'SupportsApplyMixture',
        'SupportsApproximateEquality',
        'SupportsChannel',
        'SupportsCircuitDiagramInfo',
        'SupportsCommutes',
        'SupportsConsistentApplyUnitary',
        'SupportsDecompose',
        'SupportsDecomposeWithQubits',
        'SupportsEqualUpToGlobalPhase',
        'SupportsExplicitHasUnitary',
        'SupportsExplicitNumQubits',
        'SupportsExplicitQidShape',
        'SupportsJSON',
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
        'NOISE_MODEL_LIKE',
        'OP_TREE',
        'PAULI_GATE_LIKE',
        'PAULI_STRING_LIKE',
        'ParamResolverOrSimilarType',
        'PauliSumLike',
        'QubitOrderOrList',
        'RANDOM_STATE_OR_SEED_LIKE',
        'STATE_VECTOR_LIKE',
        'Sweepable',
        'TParamKey',
        'TParamVal',
        'ParamDictType',

        # utility:
        'CliffordSimulator',
        'Simulator',
        'StabilizerSampler',
        'Unique',
        'DEFAULT_RESOLVERS',
    ])
