import functools
from collections import defaultdict
from typing import Dict, TYPE_CHECKING, Optional

from cirq.protocols.json_serialization import ObjectFactory, register_resolver

if TYPE_CHECKING:
    import cirq.ops.pauli_gates
    import cirq.devices.unconstrained_device


@functools.lru_cache(maxsize=1)
def _class_resolver_dictionary() -> Dict[str, Optional[ObjectFactory]]:
    import cirq
    from cirq.ops import raw_types
    import pandas as pd
    import numpy as np
    from cirq.devices.noise_model import _NoNoiseModel
    from cirq.experiments import (CrossEntropyResult, CrossEntropyResultDict,
                                  GridInteractionLayer)
    from cirq.experiments.grid_parallel_two_qubit_xeb import (
        GridParallelXEBMetadata)
    from cirq.google.devices.known_devices import (_NamedConstantXmonDevice)

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

    import sympy
    d = defaultdict(lambda: None)
    d.update({
        'AmplitudeDampingChannel': cirq.AmplitudeDampingChannel,
        'AsymmetricDepolarizingChannel': cirq.AsymmetricDepolarizingChannel,
        'BitFlipChannel': cirq.BitFlipChannel,
        'ProductState': cirq.ProductState,
        'CCNotPowGate': cirq.CCNotPowGate,
        'CCXPowGate': cirq.CCXPowGate,
        'CCZPowGate': cirq.CCZPowGate,
        'CNotPowGate': cirq.CNotPowGate,
        'ControlledGate': cirq.ControlledGate,
        'ControlledOperation': cirq.ControlledOperation,
        'CSwapGate': cirq.CSwapGate,
        'CXPowGate': cirq.CXPowGate,
        'CZPowGate': cirq.CZPowGate,
        'CrossEntropyResult': CrossEntropyResult,
        'CrossEntropyResultDict': CrossEntropyResultDict,
        'Circuit': cirq.Circuit,
        'CliffordState': cirq.CliffordState,
        'CliffordTableau': cirq.CliffordTableau,
        'DepolarizingChannel': cirq.DepolarizingChannel,
        'ConstantQubitNoiseModel': cirq.ConstantQubitNoiseModel,
        'Duration': cirq.Duration,
        'FrozenCircuit': cirq.FrozenCircuit,
        'FSimGate': cirq.FSimGate,
        'DensePauliString': cirq.DensePauliString,
        'MutableDensePauliString': cirq.MutableDensePauliString,
        'MutablePauliString': cirq.MutablePauliString,
        'GateOperation': cirq.GateOperation,
        'GeneralizedAmplitudeDampingChannel':
        cirq.GeneralizedAmplitudeDampingChannel,
        'GlobalPhaseOperation': cirq.GlobalPhaseOperation,
        'GridInteractionLayer': GridInteractionLayer,
        'GridParallelXEBMetadata': GridParallelXEBMetadata,
        'GridQid': cirq.GridQid,
        'GridQubit': cirq.GridQubit,
        'HPowGate': cirq.HPowGate,
        'ISwapPowGate': cirq.ISwapPowGate,
        'IdentityGate': cirq.IdentityGate,
        'IdentityOperation': _identity_operation_from_dict,
        'InitObsSetting': cirq.work.InitObsSetting,
        'LinearDict': cirq.LinearDict,
        'LineQubit': cirq.LineQubit,
        'LineQid': cirq.LineQid,
        'MatrixGate': cirq.MatrixGate,
        'MeasurementGate': cirq.MeasurementGate,
        '_MeasurementSpec': cirq.work._MeasurementSpec,
        'Moment': cirq.Moment,
        '_XEigenState': cirq.value.product_state._XEigenState,  # type: ignore
        '_YEigenState': cirq.value.product_state._YEigenState,  # type: ignore
        '_ZEigenState': cirq.value.product_state._ZEigenState,  # type: ignore
        '_NamedConstantXmonDevice': _NamedConstantXmonDevice,
        '_NoNoiseModel': _NoNoiseModel,
        'NamedQubit': cirq.NamedQubit,
        'NamedQid': cirq.NamedQid,
        'NoIdentifierQubit': cirq.testing.NoIdentifierQubit,
        '_PauliX': cirq.ops.pauli_gates._PauliX,
        '_PauliY': cirq.ops.pauli_gates._PauliY,
        '_PauliZ': cirq.ops.pauli_gates._PauliZ,
        'ParamResolver': cirq.ParamResolver,
        'PasqalDevice': cirq.pasqal.PasqalDevice,
        'PasqalVirtualDevice': cirq.pasqal.PasqalVirtualDevice,
        'PauliString': cirq.PauliString,
        'PhaseDampingChannel': cirq.PhaseDampingChannel,
        'PhaseFlipChannel': cirq.PhaseFlipChannel,
        'PhaseGradientGate': cirq.PhaseGradientGate,
        'PhasedFSimGate': cirq.PhasedFSimGate,
        'PhasedISwapPowGate': cirq.PhasedISwapPowGate,
        'PhasedXPowGate': cirq.PhasedXPowGate,
        'PhasedXZGate': cirq.PhasedXZGate,
        'RandomGateChannel': cirq.RandomGateChannel,
        'QuantumFourierTransformGate': cirq.QuantumFourierTransformGate,
        'ResetChannel': cirq.ResetChannel,
        'SingleQubitMatrixGate': single_qubit_matrix_gate,
        'SingleQubitPauliStringGateOperation':
        cirq.SingleQubitPauliStringGateOperation,
        'SingleQubitReadoutCalibrationResult':
        cirq.experiments.SingleQubitReadoutCalibrationResult,
        'StabilizerStateChForm': cirq.StabilizerStateChForm,
        'SwapPowGate': cirq.SwapPowGate,
        'TaggedOperation': cirq.TaggedOperation,
        'ThreeDQubit': cirq.pasqal.ThreeDQubit,
        'Result': cirq.Result,
        'TrialResult': cirq.TrialResult,
        'TwoDQubit': cirq.pasqal.TwoDQubit,
        'TwoQubitMatrixGate': two_qubit_matrix_gate,
        '_UnconstrainedDevice':
        cirq.devices.unconstrained_device._UnconstrainedDevice,
        'VirtualTag': cirq.VirtualTag,
        'WaitGate': cirq.WaitGate,
        '_QubitAsQid': raw_types._QubitAsQid,
        'XPowGate': cirq.XPowGate,
        'XXPowGate': cirq.XXPowGate,
        'YPowGate': cirq.YPowGate,
        'YYPowGate': cirq.YYPowGate,
        'ZPowGate': cirq.ZPowGate,
        'ZZPowGate': cirq.ZZPowGate,

        # not a cirq class, but treated as one:
        'pandas.DataFrame': pd.DataFrame,
        'pandas.Index': pd.Index,
        'pandas.MultiIndex': pd.MultiIndex.from_tuples,
        'sympy.Symbol': sympy.Symbol,
        'sympy.Add': lambda args: sympy.Add(*args),
        'sympy.Mul': lambda args: sympy.Mul(*args),
        'sympy.Pow': lambda args: sympy.Pow(*args),
        'sympy.Float': lambda approx: sympy.Float(approx),
        'sympy.Integer': sympy.Integer,
        'sympy.Rational': sympy.Rational,
        'complex': complex,
    })
    return d
