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
"""Methods for resolving JSON types during serialization."""

from __future__ import annotations

import datetime
import functools
from typing import Any, NamedTuple, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    import cirq
    from cirq.protocols.json_serialization import ObjectFactory

# Needed for backwards compatible named tuples of CrossEntropyResult
CrossEntropyPair = NamedTuple('CrossEntropyPair', [('num_cycle', int), ('xeb_fidelity', float)])
SpecklePurityPair = NamedTuple('SpecklePurityPair', [('num_cycle', int), ('purity', float)])
CrossEntropyResult = NamedTuple(
    'CrossEntropyResult',
    [
        ('data', list[CrossEntropyPair]),
        ('repetitions', int),
        ('purity_data', list[SpecklePurityPair] | None),
    ],
)
CrossEntropyResultDict = NamedTuple(
    'CrossEntropyResultDict', [('results', dict[tuple['cirq.Qid', ...], CrossEntropyResult])]
)


@functools.lru_cache()
def _class_resolver_dictionary() -> dict[str, ObjectFactory]:
    import numpy as np
    import pandas as pd
    import sympy

    import cirq
    from cirq.devices import InsertionNoiseModel, NoiseModelFromNoiseProperties, ThermalNoiseModel
    from cirq.devices.noise_model import _NoNoiseModel
    from cirq.experiments import GridInteractionLayer
    from cirq.ops import raw_types

    TSymPy = TypeVar('TSymPy', bound=sympy.Basic)

    def _boolean_hamiltonian_gate_op(qubit_map, boolean_strs, theta):
        return cirq.BooleanHamiltonianGate(
            parameter_names=list(qubit_map.keys()), boolean_strs=boolean_strs, theta=theta
        ).on(*qubit_map.values())

    def _identity_operation_from_dict(qubits, **kwargs):
        return cirq.identity_each(*qubits)

    def single_qubit_matrix_gate(matrix):
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, dtype=np.complex128)
        return cirq.MatrixGate(matrix, qid_shape=(matrix.shape[0],))

    def two_qubit_matrix_gate(matrix):
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix, dtype=np.complex128)
        return cirq.MatrixGate(matrix, qid_shape=(2, 2))

    def _cross_entropy_result(data, repetitions, **kwargs) -> CrossEntropyResult:
        purity_data = kwargs.get('purity_data', None)
        if purity_data is not None:
            purity_data = [SpecklePurityPair(d, f) for d, f in purity_data]
        return CrossEntropyResult(
            data=[CrossEntropyPair(d, f) for d, f in data],
            repetitions=repetitions,
            purity_data=purity_data,
        )

    def _cross_entropy_result_dict(
        results: list[tuple[list[cirq.Qid], CrossEntropyResult]], **kwargs
    ) -> CrossEntropyResultDict:
        return CrossEntropyResultDict(results={tuple(qubits): result for qubits, result in results})

    def _parallel_gate_op(gate, qubits):
        return cirq.parallel_gate_op(gate, *qubits)

    def _datetime(timestamp: float) -> datetime.datetime:
        # We serialize datetimes (both with ("aware") and without ("naive") timezone information)
        # as unix timestamps. The deserialized datetime will always refer to the
        # same point in time, but will be re-constructed as a timezone-aware object.
        #
        # If `o` is a naive datetime,  o != read_json(to_json(o)) because Python doesn't
        # let you compare aware and naive datetimes.
        return datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)

    def _symmetricalqidpair(qids):
        return frozenset(qids)

    def _sympy_check_args(cls: type[TSymPy], allowed_type: type, *args: Any) -> TSymPy:
        """Instantiate SymPy class after ensuring all arguments are of the `allowed_type` type."""
        for arg in args:
            if not isinstance(arg, allowed_type):
                raise TypeError(f"Expected a serialized SymPy argument, got {type(arg).__name__}.")
        return cls(*args)

    return {
        'AmplitudeDampingChannel': cirq.AmplitudeDampingChannel,
        'AnyIntegerPowerGateFamily': cirq.AnyIntegerPowerGateFamily,
        'AnyUnitaryGateFamily': cirq.AnyUnitaryGateFamily,
        'AsymmetricDepolarizingChannel': cirq.AsymmetricDepolarizingChannel,
        'BitFlipChannel': cirq.BitFlipChannel,
        'BitMaskKeyCondition': cirq.BitMaskKeyCondition,
        'BitstringAccumulator': cirq.work.BitstringAccumulator,
        'BooleanHamiltonianGate': cirq.BooleanHamiltonianGate,
        'CCNotPowGate': cirq.CCNotPowGate,
        'CCXPowGate': cirq.CCXPowGate,
        'CCYPowGate': cirq.CCYPowGate,
        'CCZPowGate': cirq.CCZPowGate,
        'Circuit': cirq.Circuit,
        'CircuitFunction': cirq.CircuitFunction,
        'CircuitOperation': cirq.CircuitOperation,
        'ClassicallyControlledOperation': cirq.ClassicallyControlledOperation,
        'ClassicalDataDictionaryStore': cirq.ClassicalDataDictionaryStore,
        'CliffordGate': cirq.CliffordGate,
        'CliffordState': cirq.CliffordState,
        'CliffordTableau': cirq.CliffordTableau,
        'CNotPowGate': cirq.CNotPowGate,
        'Concat': cirq.Concat,
        'ConstantQubitNoiseModel': cirq.ConstantQubitNoiseModel,
        'ControlledGate': cirq.ControlledGate,
        'ControlledOperation': cirq.ControlledOperation,
        'CSwapGate': cirq.CSwapGate,
        'CXPowGate': cirq.CXPowGate,
        'CYPowGate': cirq.CYPowGate,
        'CZPowGate': cirq.CZPowGate,
        'CZTargetGateset': cirq.CZTargetGateset,
        'DiagonalGate': cirq.DiagonalGate,
        'DensePauliString': cirq.DensePauliString,
        'DepolarizingChannel': cirq.DepolarizingChannel,
        'DeviceMetadata': cirq.DeviceMetadata,
        'Duration': cirq.Duration,
        'For': cirq.For,
        'FrozenCircuit': cirq.FrozenCircuit,
        'FSimGate': cirq.FSimGate,
        'GateFamily': cirq.GateFamily,
        'GateOperation': cirq.GateOperation,
        'Gateset': cirq.Gateset,
        'GeneralizedAmplitudeDampingChannel': cirq.GeneralizedAmplitudeDampingChannel,
        'GlobalPhaseGate': cirq.GlobalPhaseGate,
        'GridDeviceMetadata': cirq.GridDeviceMetadata,
        'GridInteractionLayer': GridInteractionLayer,
        'GridQid': cirq.GridQid,
        'GridQubit': cirq.GridQubit,
        'HPowGate': cirq.HPowGate,
        'ISwapPowGate': cirq.ISwapPowGate,
        'IdentityGate': cirq.IdentityGate,
        'If': cirq.If,
        'InitObsSetting': cirq.work.InitObsSetting,
        'InsertionNoiseModel': InsertionNoiseModel,
        '_InverseCompositeGate': raw_types._InverseCompositeGate,
        'KeyCondition': cirq.KeyCondition,
        'KrausChannel': cirq.KrausChannel,
        'LinearDict': cirq.LinearDict,
        'LineQubit': cirq.LineQubit,
        'LineQid': cirq.LineQid,
        'LineTopology': cirq.LineTopology,
        'Linspace': cirq.Linspace,
        'ListSweep': cirq.ListSweep,
        'cirq.MSGate': cirq.MSGate,
        'MatrixGate': cirq.MatrixGate,
        'MixedUnitaryChannel': cirq.MixedUnitaryChannel,
        'MeasurementKey': cirq.MeasurementKey,
        'MeasurementGate': cirq.MeasurementGate,
        'MeasurementType': cirq.MeasurementType,
        '_MeasurementSpec': cirq.work._MeasurementSpec,
        'Moment': cirq.Moment,
        'MutableDensePauliString': cirq.MutableDensePauliString,
        'MutablePauliString': cirq.MutablePauliString,
        '_NoNoiseModel': _NoNoiseModel,
        'NamedQubit': cirq.NamedQubit,
        'NamedQid': cirq.NamedQid,
        'NoIdentifierQubit': cirq.testing.NoIdentifierQubit,
        'NoiseModelFromNoiseProperties': NoiseModelFromNoiseProperties,
        'ObservableMeasuredResult': cirq.work.ObservableMeasuredResult,
        'OpIdentifier': cirq.OpIdentifier,
        'ParamResolver': cirq.ParamResolver,
        'ParallelGate': cirq.ParallelGate,
        'ParallelGateFamily': cirq.ParallelGateFamily,
        'PauliInteractionGate': cirq.PauliInteractionGate,
        'PauliMeasurementGate': cirq.PauliMeasurementGate,
        'PauliString': cirq.PauliString,
        'PauliStringPhasor': cirq.PauliStringPhasor,
        'PauliStringPhasorGate': cirq.PauliStringPhasorGate,
        'PauliSum': cirq.PauliSum,
        '_PauliX': cirq.ops.pauli_gates._PauliX,
        '_PauliY': cirq.ops.pauli_gates._PauliY,
        '_PauliZ': cirq.ops.pauli_gates._PauliZ,
        'PhaseDampingChannel': cirq.PhaseDampingChannel,
        'PhaseFlipChannel': cirq.PhaseFlipChannel,
        'PhaseGradientGate': cirq.PhaseGradientGate,
        'PhasedFSimGate': cirq.PhasedFSimGate,
        'PhasedISwapPowGate': cirq.PhasedISwapPowGate,
        'PhasedXPowGate': cirq.PhasedXPowGate,
        'PhasedXZGate': cirq.PhasedXZGate,
        'Points': cirq.Points,
        'Product': cirq.Product,
        'ProductState': cirq.ProductState,
        'ProductOfSums': cirq.ProductOfSums,
        'ProjectorString': cirq.ProjectorString,
        'ProjectorSum': cirq.ProjectorSum,
        'QasmUGate': cirq.circuits.qasm_output.QasmUGate,
        '_QubitAsQid': raw_types._QubitAsQid,
        'QuantumFourierTransformGate': cirq.QuantumFourierTransformGate,
        'QubitPermutationGate': cirq.QubitPermutationGate,
        'RandomGateChannel': cirq.RandomGateChannel,
        'RepetitionsStoppingCriteria': cirq.work.RepetitionsStoppingCriteria,
        'ResetChannel': cirq.ResetChannel,
        'Result': cirq.ResultDict,  # Keep support for Cirq < 0.14.
        'ResultDict': cirq.ResultDict,
        'RoutingSwapTag': cirq.RoutingSwapTag,
        'Rx': cirq.Rx,
        'Ry': cirq.Ry,
        'Rz': cirq.Rz,
        'SingleQubitCliffordGate': cirq.SingleQubitCliffordGate,
        'SingleQubitPauliStringGateOperation': cirq.SingleQubitPauliStringGateOperation,
        'SingleQubitReadoutCalibrationResult': cirq.experiments.SingleQubitReadoutCalibrationResult,
        'SetVariable': cirq.SetVariable,
        'SqrtIswapTargetGateset': cirq.SqrtIswapTargetGateset,
        'StabilizerStateChForm': cirq.StabilizerStateChForm,
        'StatePreparationChannel': cirq.StatePreparationChannel,
        'SumOfProducts': cirq.SumOfProducts,
        'SwapPowGate': cirq.SwapPowGate,
        'SympyCondition': cirq.SympyCondition,
        'TaggedOperation': cirq.TaggedOperation,
        'TensoredConfusionMatrices': cirq.TensoredConfusionMatrices,
        'ThermalNoiseModel': ThermalNoiseModel,
        'TiltedSquareLattice': cirq.TiltedSquareLattice,
        'ThreeQubitDiagonalGate': cirq.ThreeQubitDiagonalGate,
        'TrialResult': cirq.ResultDict,  # keep support for Cirq < 0.11.
        'TwoQubitDiagonalGate': cirq.TwoQubitDiagonalGate,
        'TwoQubitGateTabulation': cirq.TwoQubitGateTabulation,
        'TwoQubitNumericalCompiler': cirq.TwoQubitNumericalCompiler,
        '_UnconstrainedDevice': cirq.devices.unconstrained_device._UnconstrainedDevice,
        '_Unit': cirq.study.sweeps._Unit,
        'VariableLineQid': cirq.ops.variable_qid.VariableLineQid,
        'VariableGridQid': cirq.ops.variable_qid.VariableGridQid,
        'VarianceStoppingCriteria': cirq.work.VarianceStoppingCriteria,
        'VirtualTag': cirq.VirtualTag,
        'WaitGate': cirq.WaitGate,
        'While': cirq.While,
        # The formatter keeps putting this back
        'XEBPhasedFSimCharacterizationOptions': cirq.experiments.XEBPhasedFSimCharacterizationOptions,  # noqa: E501
        '_XEigenState': cirq.value.product_state._XEigenState,
        'XPowGate': cirq.XPowGate,
        'XXPowGate': cirq.XXPowGate,
        '_YEigenState': cirq.value.product_state._YEigenState,
        'YPowGate': cirq.YPowGate,
        'YYPowGate': cirq.YYPowGate,
        '_ZEigenState': cirq.value.product_state._ZEigenState,
        'Zip': cirq.Zip,
        'ZipLongest': cirq.ZipLongest,
        'ZPowGate': cirq.ZPowGate,
        'ZZPowGate': cirq.ZZPowGate,
        'UniformSuperpositionGate': cirq.UniformSuperpositionGate,
        # Old types, only supported for backwards-compatibility
        'BooleanHamiltonian': _boolean_hamiltonian_gate_op,  # Removed in v0.15
        'CrossEntropyResult': _cross_entropy_result,  # Removed in v0.16
        'CrossEntropyResultDict': _cross_entropy_result_dict,  # Removed in v0.16
        'IdentityOperation': _identity_operation_from_dict,
        'ParallelGateOperation': _parallel_gate_op,  # Removed in v0.14
        'SingleQubitMatrixGate': single_qubit_matrix_gate,
        'SymmetricalQidPair': _symmetricalqidpair,  # Removed in v0.15
        'TwoQubitMatrixGate': two_qubit_matrix_gate,
        'GlobalPhaseOperation': cirq.global_phase_operation,  # Removed in v0.16
        # not a cirq class, but treated as one:
        'pandas.DataFrame': pd.DataFrame,
        'pandas.Index': pd.Index,
        'pandas.MultiIndex': pd.MultiIndex.from_tuples,
        'sympy.Symbol': lambda name: _sympy_check_args(sympy.Symbol, str, name),
        'sympy.Add': lambda args: _sympy_check_args(sympy.Add, sympy.Basic, *args),
        'sympy.Mul': lambda args: _sympy_check_args(sympy.Mul, sympy.Basic, *args),
        'sympy.Pow': lambda args: _sympy_check_args(sympy.Pow, sympy.Basic, *args),
        'sympy.GreaterThan': lambda args: _sympy_check_args(sympy.GreaterThan, sympy.Basic, *args),
        'sympy.StrictGreaterThan': lambda args: _sympy_check_args(
            sympy.StrictGreaterThan, sympy.Basic, *args
        ),
        'sympy.LessThan': lambda args: _sympy_check_args(sympy.LessThan, sympy.Basic, *args),
        'sympy.StrictLessThan': lambda args: _sympy_check_args(
            sympy.StrictLessThan, sympy.Basic, *args
        ),
        'sympy.Equality': lambda args: _sympy_check_args(sympy.Equality, sympy.Basic, *args),
        'sympy.Unequality': lambda args: _sympy_check_args(sympy.Unequality, sympy.Basic, *args),
        'sympy.And': lambda args: _sympy_check_args(sympy.And, sympy.Basic, *args),
        'sympy.Or': lambda args: _sympy_check_args(sympy.Or, sympy.Basic, *args),
        'sympy.Not': lambda args: _sympy_check_args(sympy.Not, sympy.Basic, *args),
        'sympy.Xor': lambda args: _sympy_check_args(sympy.Xor, sympy.Basic, *args),
        'sympy.Indexed': lambda args: _sympy_check_args(sympy.Indexed, sympy.Basic, *args),
        'sympy.IndexedBase': lambda args: _sympy_check_args(sympy.IndexedBase, sympy.Basic, *args),
        'sympy.Float': lambda approx: _sympy_check_args(sympy.Float, float, approx),
        'sympy.Integer': lambda i: _sympy_check_args(sympy.Integer, int, i),
        'sympy.Rational': lambda p, q: _sympy_check_args(sympy.Rational, int, p, q),
        'sympy.pi': lambda: sympy.pi,
        'sympy.E': lambda: sympy.E,
        'sympy.EulerGamma': lambda: sympy.EulerGamma,
        'complex': complex,
        'datetime.datetime': _datetime,
    }
