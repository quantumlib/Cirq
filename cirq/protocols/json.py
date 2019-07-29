# Copyright 2019 The Cirq Developers
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

import json
from typing import Union, Any, Dict, Optional, List, Callable, Type

import numpy as np
from typing_extensions import Protocol

from cirq.type_workarounds import NotImplementedType


# Note for developers:
# I used this code to originally generate this list, followed by manual
# pruning:
#
# import cirq
#
# for k in sorted(cirq.__dict__.keys()):
#     if k.startswith('__'):
#         continue
#
#     if k[0].isupper():
#         if k.isupper():
#             continue
#         print(f"'{k}': cirq.{k},")

def _cirq_class_resolver(cirq_type: str):
    import cirq
    return {
        # 'AmplitudeDampingChannel': cirq.AmplitudeDampingChannel,
        # 'ApproxPauliStringExpectation': cirq.ApproxPauliStringExpectation,
        # 'AsymmetricDepolarizingChannel': cirq.AsymmetricDepolarizingChannel,
        # 'AxisAngleDecomposition': cirq.AxisAngleDecomposition,
        # 'BitFlipChannel': cirq.BitFlipChannel,
        'CCXPowGate': cirq.CCXPowGate,
        'CCZPowGate': cirq.CCZPowGate,
        'CNotPowGate': cirq.CNotPowGate,
        # 'CSwapGate': cirq.CSwapGate,
        'CZPowGate': cirq.CZPowGate,
        # 'Circuit': cirq.Circuit,
        # 'CircuitDag': cirq.CircuitDag,
        # 'CircuitDiagramInfo': cirq.CircuitDiagramInfo,
        # 'CircuitDiagramInfoArgs': cirq.CircuitDiagramInfoArgs,
        # 'CircuitSampleJob': cirq.CircuitSampleJob,
        # 'ComputeDisplaysResult': cirq.ComputeDisplaysResult,
        # 'ConstantQubitNoiseModel': cirq.ConstantQubitNoiseModel,
        # 'ControlledGate': cirq.ControlledGate,
        # 'ControlledOperation': cirq.ControlledOperation,
        # 'ConvertToCzAndSingleGates': cirq.ConvertToCzAndSingleGates,
        # 'ConvertToIonGates': cirq.ConvertToIonGates,
        # 'ConvertToNeutralAtomGates': cirq.ConvertToNeutralAtomGates,
        # 'DensityMatrixDisplay': cirq.DensityMatrixDisplay,
        # 'DensityMatrixSimulator': cirq.DensityMatrixSimulator,
        # 'DensityMatrixSimulatorState': cirq.DensityMatrixSimulatorState,
        # 'DensityMatrixStepResult': cirq.DensityMatrixStepResult,
        # 'DensityMatrixTrialResult': cirq.DensityMatrixTrialResult,
        # 'DepolarizingChannel': cirq.DepolarizingChannel,
        # 'Device': cirq.Device,
        # 'DropEmptyMoments': cirq.DropEmptyMoments,
        # 'DropNegligible': cirq.DropNegligible,
        # 'Duration': cirq.Duration,
        'EigenGate': cirq.EigenGate,
        # 'EjectPhasedPaulis': cirq.EjectPhasedPaulis,
        # 'EjectZ': cirq.EjectZ,
        # 'ExpandComposite': cirq.ExpandComposite,
        # 'FSimGate': cirq.FSimGate,
        # 'Gate': cirq.Gate,
        'GateOperation': cirq.GateOperation,
        # 'GeneralizedAmplitudeDampingChannel': cirq.GeneralizedAmplitudeDampingChannel,
        # 'GlobalPhaseOperation': cirq.GlobalPhaseOperation,
        'GridQubit': cirq.GridQubit,
        'HPowGate': cirq.HPowGate,
        'ISwapPowGate': cirq.ISwapPowGate,
        # 'IdentityGate': cirq.IdentityGate,
        # 'InsertStrategy': cirq.InsertStrategy,
        # 'InterchangeableQubitsGate': cirq.InterchangeableQubitsGate,
        # 'IonDevice': cirq.IonDevice,
        # 'KakDecomposition': cirq.KakDecomposition,
        'LineQubit': cirq.LineQubit,
        # 'LinearCombinationOfGates': cirq.LinearCombinationOfGates,
        # 'LinearCombinationOfOperations': cirq.LinearCombinationOfOperations,
        # 'LinearDict': cirq.LinearDict,
        # 'Linspace': cirq.Linspace,
        # 'MeasurementGate': cirq.MeasurementGate,
        # 'MergeInteractions': cirq.MergeInteractions,
        # 'MergeSingleQubitGates': cirq.MergeSingleQubitGates,
        # 'Moment': cirq.Moment,
        # 'NamedQubit': cirq.NamedQubit,
        # 'NeutralAtomDevice': cirq.NeutralAtomDevice,
        # 'NoiseModel': cirq.NoiseModel,
        # 'Operation': cirq.Operation,
        # 'ParallelGateOperation': cirq.ParallelGateOperation,
        # 'ParamResolver': cirq.ParamResolver,
        # 'ParamResolverOrSimilarType': cirq.ParamResolverOrSimilarType,
        # 'Pauli': cirq.Pauli,
        'PauliInteractionGate': cirq.PauliInteractionGate,
        # 'PauliString': cirq.PauliString,
        # 'PauliStringExpectation': cirq.PauliStringExpectation,
        # 'PauliStringGateOperation': cirq.PauliStringGateOperation,
        # 'PauliStringPhasor': cirq.PauliStringPhasor,
        # 'PauliSum': cirq.PauliSum,
        # 'PauliSumCollector': cirq.PauliSumCollector,
        # 'PauliSumLike': cirq.PauliSumLike,
        # 'PauliTransform': cirq.PauliTransform,
        # 'PeriodicValue': cirq.PeriodicValue,
        # 'PhaseDampingChannel': cirq.PhaseDampingChannel,
        # 'PhaseFlipChannel': cirq.PhaseFlipChannel,
        # 'PhasedXPowGate': cirq.PhasedXPowGate,
        # 'PointOptimizationSummary': cirq.PointOptimizationSummary,
        # 'PointOptimizer': cirq.PointOptimizer,
        # 'Points': cirq.Points,
        # 'QasmArgs': cirq.QasmArgs,
        # 'QasmOutput': cirq.QasmOutput,
        # 'Qid': cirq.Qid,
        # 'QubitOrder': cirq.QubitOrder,
        # 'QubitOrderOrList': cirq.QubitOrderOrList,
        # 'Rx': cirq.Rx,
        # 'Ry': cirq.Ry,
        # 'Rz': cirq.Rz,
        # 'Sampler': cirq.Sampler,
        # 'SamplesDisplay': cirq.SamplesDisplay,
        # 'Schedule': cirq.Schedule,
        # 'ScheduledOperation': cirq.ScheduledOperation,
        # 'SimulatesFinalState': cirq.SimulatesFinalState,
        # 'SimulatesIntermediateState': cirq.SimulatesIntermediateState,
        # 'SimulatesIntermediateWaveFunction': cirq.SimulatesIntermediateWaveFunction,
        # 'SimulatesSamples': cirq.SimulatesSamples,
        # 'SimulationTrialResult': cirq.SimulationTrialResult,
        # 'Simulator': cirq.Simulator,
        # 'SingleQubitCliffordGate': cirq.SingleQubitCliffordGate,
        # 'SingleQubitGate': cirq.SingleQubitGate,
        # 'SingleQubitMatrixGate': cirq.SingleQubitMatrixGate,
        'SingleQubitPauliStringGateOperation': cirq.SingleQubitPauliStringGateOperation,
        # 'SparseSimulatorStep': cirq.SparseSimulatorStep,
        # 'StateVectorMixin': cirq.StateVectorMixin,
        # 'StepResult': cirq.StepResult,
        # 'SupportsApplyChannel': cirq.SupportsApplyChannel,
        # 'SupportsApproximateEquality': cirq.SupportsApproximateEquality,
        # 'SupportsChannel': cirq.SupportsChannel,
        # 'SupportsCircuitDiagramInfo': cirq.SupportsCircuitDiagramInfo,
        # 'SupportsConsistentApplyUnitary': cirq.SupportsConsistentApplyUnitary,
        # 'SupportsDecompose': cirq.SupportsDecompose,
        # 'SupportsDecomposeWithQubits': cirq.SupportsDecomposeWithQubits,
        # 'SupportsExplicitNumQubits': cirq.SupportsExplicitNumQubits,
        # 'SupportsExplicitQidShape': cirq.SupportsExplicitQidShape,
        # 'SupportsMixture': cirq.SupportsMixture,
        # 'SupportsParameterization': cirq.SupportsParameterization,
        # 'SupportsPhase': cirq.SupportsPhase,
        # 'SupportsQasm': cirq.SupportsQasm,
        # 'SupportsQasmWithArgs': cirq.SupportsQasmWithArgs,
        # 'SupportsQasmWithArgsAndQubits': cirq.SupportsQasmWithArgsAndQubits,
        # 'SupportsTraceDistanceBound': cirq.SupportsTraceDistanceBound,
        # 'SupportsUnitary': cirq.SupportsUnitary,
        'SwapPowGate': cirq.SwapPowGate,
        # 'Sweep': cirq.Sweep,
        # 'Sweepable': cirq.Sweepable,
        # 'TextDiagramDrawer': cirq.TextDiagramDrawer,
        # 'ThreeQubitGate': cirq.ThreeQubitGate,
        # 'Timestamp': cirq.Timestamp,
        # 'TrialResult': cirq.TrialResult,
        # 'TwoQubitGate': cirq.TwoQubitGate,
        # 'TwoQubitMatrixGate': cirq.TwoQubitMatrixGate,
        # 'UnconstrainedDevice': cirq.UnconstrainedDevice,
        # 'Unique': cirq.Unique,
        # 'UnitSweep': cirq.UnitSweep,
        # 'WaveFunctionDisplay': cirq.WaveFunctionDisplay,
        # 'WaveFunctionSimulatorState': cirq.WaveFunctionSimulatorState,
        # 'WaveFunctionStepResult': cirq.WaveFunctionStepResult,
        # 'WaveFunctionTrialResult': cirq.WaveFunctionTrialResult,
        'XPowGate': cirq.XPowGate,
        'XXPowGate': cirq.XXPowGate,
        'YPowGate': cirq.YPowGate,
        'YYPowGate': cirq.YYPowGate,
        'ZPowGate': cirq.ZPowGate,
        'ZZPowGate': cirq.ZZPowGate,
    }[cirq_type]


DEFAULT_RESOLVERS = [
    _cirq_class_resolver,
]


class SupportsJSON(Protocol):
    """An object that can be turned into JSON dictionaries.

    The magic method _json_dict_ must return a trivially json-serializable
    type or other objects that support the SupportsJSON protocol.

    During deserialization, a class must be able to be resolved (see
    the docstring for `read_json`) and must be able to be (re-)constructed
    from the serialized parameters. If the type defines a classmethod
    `_from_json_dict_`, that will be called. Otherwise, the `cirq_type` key
    will be popped from the dictionary and used as kwargs to the type's
    constructor.
    """

    def _json_dict_(self) -> Union[None, NotImplementedType, Dict[Any, Any]]:
        pass


def to_json_dict(obj, attribute_names):
    d = {'cirq_type': obj.__class__.__name__}
    for attr_name in attribute_names:
        d[attr_name] = getattr(obj, attr_name)
    return d


class CirqEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, '_json_dict_'):
            return o._json_dict_()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.int_):
            return o.item()
        return super().default(o)


def _cirq_object_hook(d, resolvers):
    if 'cirq_type' in d:
        for resolver in resolvers:
            cls = resolver(d['cirq_type'])
            if cls is not None:
                break
        else:
            raise ValueError("Could not resolve {}".format(d['cirq_type']))

        if hasattr(cls, '_from_json_dict_'):
            return cls._from_json_dict_(**d)

        del d['cirq_type']
        return cls(**d)

    return d


def to_json(obj: Any, file, *, indent=2, cls=CirqEncoder):
    if isinstance(file, str):
        with open(file, 'w') as actually_a_file:
            return json.dump(obj, actually_a_file, indent=indent, cls=cls)

    return json.dump(obj, file, indent=indent, cls=cls)


def read_json(file_or_fn, resolvers: Optional[List[Callable[[str], Type]]] = None):
    """Read a JSON file that optionally contains cirq objects.

    Args:
        file_or_fn: A filename (if a string), otherwise a file-like
            object.
        resolvers: A list of functions that are called in order to turn
            the serialized `cirq_type` string into a constructable class.
            By default, top-level cirq objects that implement the SupportsJSON
            protocol are supported. You can extend the list of supported types
            by pre-pending custom resolvers. Each resolver should return `None`
            to indicate that it cannot resolve the given cirq_type and that
            the next resolver should be tried.
    """
    if resolvers is None:
        resolvers = DEFAULT_RESOLVERS

    def obj_hook(x):
        return _cirq_object_hook(x, resolvers)

    if isinstance(file_or_fn, str):
        with open(file_or_fn, 'r') as file:
            return json.load(file, object_hook=obj_hook)

    return json.load(file_or_fn, object_hook=obj_hook)
