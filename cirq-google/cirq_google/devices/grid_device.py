# Copyright 2022 The Cirq Developers
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

"""Device object representing Google devices with a grid qubit layout."""

from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)
import re
import warnings
from dataclasses import dataclass

import cirq
from cirq_google import ops
from cirq_google import transformers
from cirq_google.api import v2
from cirq_google.devices import known_devices
from cirq_google.experimental import ops as experimental_ops


# Gate family constants used in various parts of GridDevice logic.
_PHASED_XZ_GATE_FAMILY = cirq.GateFamily(cirq.PhasedXZGate)
_MEASUREMENT_GATE_FAMILY = cirq.GateFamily(cirq.MeasurementGate)
_WAIT_GATE_FAMILY = cirq.GateFamily(cirq.WaitGate)

_SYC_FSIM_GATE_FAMILY = ops.FSimGateFamily(gates_to_accept=[ops.SYC])
_SQRT_ISWAP_FSIM_GATE_FAMILY = ops.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP])
_SQRT_ISWAP_INV_FSIM_GATE_FAMILY = ops.FSimGateFamily(gates_to_accept=[cirq.SQRT_ISWAP_INV])
_CZ_FSIM_GATE_FAMILY = ops.FSimGateFamily(gates_to_accept=[cirq.CZ])
_SYC_GATE_FAMILY = cirq.GateFamily(ops.SYC)
_SQRT_ISWAP_GATE_FAMILY = cirq.GateFamily(cirq.SQRT_ISWAP)
_SQRT_ISWAP_INV_GATE_FAMILY = cirq.GateFamily(cirq.SQRT_ISWAP_INV)
_CZ_GATE_FAMILY = cirq.GateFamily(cirq.CZ)
_CZ_POW_GATE_FAMILY = cirq.GateFamily(cirq.CZPowGate)


# TODO(#5050) Add GlobalPhaseGate
# Target gates of `cirq_google.GoogleCZTargetGateset`.
_CZ_TARGET_GATES = [
    _CZ_FSIM_GATE_FAMILY,
    _CZ_GATE_FAMILY,
    _PHASED_XZ_GATE_FAMILY,
    _MEASUREMENT_GATE_FAMILY,
]
# Target gates of cirq.CZTargetGateset with allow_partial_czs=True.
_CZ_POW_TARGET_GATES = [_CZ_POW_GATE_FAMILY, _PHASED_XZ_GATE_FAMILY, _MEASUREMENT_GATE_FAMILY]
# Target gates of `cirq_google.SycamoreTargetGateset`.
_SYC_TARGET_GATES = [
    _SYC_FSIM_GATE_FAMILY,
    _SYC_GATE_FAMILY,
    _PHASED_XZ_GATE_FAMILY,
    _MEASUREMENT_GATE_FAMILY,
]
# Target gates of `cirq.SqrtIswapTargetGateset`
_SQRT_ISWAP_TARGET_GATES = [
    _SQRT_ISWAP_FSIM_GATE_FAMILY,
    _SQRT_ISWAP_GATE_FAMILY,
    _PHASED_XZ_GATE_FAMILY,
    _MEASUREMENT_GATE_FAMILY,
]


# Families of gates which can be applied to any subset of valid qubits.
_VARIADIC_GATE_FAMILIES = [_MEASUREMENT_GATE_FAMILY, _WAIT_GATE_FAMILY]


GateOrFamily = Union[Type[cirq.Gate], cirq.Gate, cirq.GateFamily]


@dataclass
class _GateRepresentations:
    """Contains equivalent representations of a gate in both DeviceSpecification and GridDevice.

    Attributes:
        gate_spec_name: The name of gate type in `GateSpecification`.
        supported_gates: A list of gates that can be serialized into the `GateSpecification` with
            the matching name.
    """

    gate_spec_name: str
    supported_gates: List[cirq.GateFamily]


# Gates recognized by the GridDevice class. This controls the (de)serialization between
# `DeviceSpecification.valid_gates` and `cirq.Gateset`.

# This is a superset of valid gates for a given `GridDevice` instance. The specific gateset depends
# on the underlying device.

# Edit this list to add support for new gates. If a new `_GateRepresentations` is added, add a new
# `GateSpecification` message in cirq-google/cirq_google/api/v2/device.proto.

# Update `_build_compilation_target_gatesets()` if the gate you are updating affects an existing
# CompilationTargetGateset there, or if you'd like to add another `CompilationTargetGateset` to
# allow users to transform their circuits that include your gate.
_GATES: List[_GateRepresentations] = [
    _GateRepresentations(
        gate_spec_name='syc', supported_gates=[_SYC_FSIM_GATE_FAMILY, _SYC_GATE_FAMILY]
    ),
    _GateRepresentations(
        gate_spec_name='sqrt_iswap',
        supported_gates=[_SQRT_ISWAP_FSIM_GATE_FAMILY, _SQRT_ISWAP_GATE_FAMILY],
    ),
    _GateRepresentations(
        gate_spec_name='sqrt_iswap_inv',
        supported_gates=[_SQRT_ISWAP_INV_FSIM_GATE_FAMILY, _SQRT_ISWAP_INV_GATE_FAMILY],
    ),
    _GateRepresentations(
        gate_spec_name='cz', supported_gates=[_CZ_FSIM_GATE_FAMILY, _CZ_GATE_FAMILY]
    ),
    _GateRepresentations(gate_spec_name='cz_pow_gate', supported_gates=[_CZ_POW_GATE_FAMILY]),
    _GateRepresentations(
        gate_spec_name='phased_xz',
        supported_gates=[
            # TODO: Extend support to cirq.IdentityGate.
            cirq.GateFamily(cirq.I),
            cirq.GateFamily(cirq.PhasedXZGate),
            cirq.GateFamily(cirq.XPowGate),
            cirq.GateFamily(cirq.YPowGate),
            cirq.GateFamily(cirq.HPowGate),
            cirq.GateFamily(cirq.PhasedXPowGate),
            cirq.GateFamily(cirq.ops.SingleQubitCliffordGate),
        ],
    ),
    _GateRepresentations(
        gate_spec_name='virtual_zpow',
        supported_gates=[cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=[ops.PhysicalZTag()])],
    ),
    _GateRepresentations(
        gate_spec_name='physical_zpow',
        supported_gates=[cirq.GateFamily(cirq.ZPowGate, tags_to_accept=[ops.PhysicalZTag()])],
    ),
    _GateRepresentations(
        gate_spec_name='coupler_pulse',
        supported_gates=[cirq.GateFamily(experimental_ops.CouplerPulse)],
    ),
    _GateRepresentations(
        gate_spec_name='meas', supported_gates=[cirq.GateFamily(cirq.MeasurementGate)]
    ),
    _GateRepresentations(gate_spec_name='wait', supported_gates=[cirq.GateFamily(cirq.WaitGate)]),
    _GateRepresentations(
        gate_spec_name='fsim_via_model',
        supported_gates=[cirq.GateFamily(cirq.FSimGate, tags_to_accept=[ops.FSimViaModelTag()])],
    ),
    _GateRepresentations(
        gate_spec_name='internal_gate', supported_gates=[cirq.GateFamily(ops.InternalGate)]
    ),
]


def _validate_device_specification(proto: v2.device_pb2.DeviceSpecification) -> None:
    """Raises a ValueError if the `DeviceSpecification` proto is invalid."""

    qubit_set = set()
    for q_name in proto.valid_qubits:
        # Qubit names must be unique.
        if q_name in qubit_set:
            raise ValueError(
                f"Invalid DeviceSpecification: valid_qubits contains duplicate qubit '{q_name}'."
            )
        # Qubit names must be in the form <int>_<int> to be parsed as cirq.GridQubits.
        if re.match(r'^[0-9]+\_[0-9]+$', q_name) is None:
            raise ValueError(
                f"Invalid DeviceSpecification: valid_qubits contains the qubit '{q_name}' which is"
                " not in the GridQubit form '<int>_<int>."
            )
        qubit_set.add(q_name)

    for target_set in proto.valid_targets:
        # Check for unknown qubits in targets.
        for target in target_set.targets:
            for target_id in target.ids:
                if target_id not in proto.valid_qubits:
                    raise ValueError(
                        f"Invalid DeviceSpecification: valid_targets contain qubit '{target_id}'"
                        " which is not in valid_qubits."
                    )

        # Symmetric targets should not have repeated qubits.
        if target_set.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC:
            for target in target_set.targets:
                if len(target.ids) > len(set(target.ids)):
                    raise ValueError(
                        f"Invalid DeviceSpecification: the target set '{target_set.name}' is"
                        " SYMMETRIC but has a target which contains repeated qubits:"
                        f" {target.ids}."
                    )

        # Asymmetric target set type is not expected.
        # While this is allowed by the proto, it has never been set, so it's safe to raise an
        # exception if this is set unexpectedly.
        if target_set.target_ordering == v2.device_pb2.TargetSet.ASYMMETRIC:
            raise ValueError("Invalid DeviceSpecification: target_ordering cannot be ASYMMETRIC.")


def _serialize_gateset_and_gate_durations(
    out: v2.device_pb2.DeviceSpecification,
    gateset: cirq.Gateset,
    gate_durations: Mapping[cirq.GateFamily, cirq.Duration],
) -> v2.device_pb2.DeviceSpecification:
    """Serializes the given gateset and gate durations to DeviceSpecification."""

    gate_specs: Dict[str, v2.device_pb2.GateSpecification] = {}
    for gate_family in gateset.gates:
        gate_spec = v2.device_pb2.GateSpecification()
        gate_rep = next(
            (gr for gr in _GATES for gf in gr.supported_gates if gf == gate_family), None
        )
        if gate_rep is None:
            raise ValueError(f'Unrecognized gate: {gate_family}.')
        gate_name = gate_rep.gate_spec_name

        # Set gate
        getattr(gate_spec, gate_name).SetInParent()

        # Set gate duration
        gate_durations_picos = {
            int(gate_durations[gf].total_picos())
            for gf in gate_rep.supported_gates
            if gf in gate_durations
        }
        if len(gate_durations_picos) > 1:
            raise ValueError(
                'Multiple gate families in the following list exist in the gate duration dict, and '
                f'they are expected to have the same duration value: {gate_rep.supported_gates}'
            )
        elif len(gate_durations_picos) == 1:
            gate_spec.gate_duration_picos = gate_durations_picos.pop()

        # GateSpecification dedup. Multiple gates or GateFamilies in the gateset could map to the
        # same GateSpecification.
        gate_specs[gate_name] = gate_spec

    # Sort by gate name to keep valid_gates stable.
    out.valid_gates.extend(v for _, v in sorted(gate_specs.items()))

    return out


def _deserialize_gateset_and_gate_durations(
    proto: v2.device_pb2.DeviceSpecification,
) -> Tuple[cirq.Gateset, Mapping[cirq.GateFamily, cirq.Duration]]:
    """Deserializes gateset and gate duration from DeviceSpecification."""

    gates_list: List[GateOrFamily] = []
    gate_durations: Dict[cirq.GateFamily, cirq.Duration] = {}

    for gate_spec in proto.valid_gates:
        gate_name = gate_spec.WhichOneof('gate')

        gate_rep = next((gr for gr in _GATES if gr.gate_spec_name == gate_name), None)
        if gate_rep is None:  # pragma: no cover
            warnings.warn(
                f"The DeviceSpecification contains the gate '{gate_name}' which is not recognized"
                " by Cirq and will be ignored. This may be due to an out-of-date Cirq version.",
                UserWarning,
            )
            continue

        gates_list.extend(gate_rep.supported_gates)
        for g in gate_rep.supported_gates:
            gate_durations[g] = cirq.Duration(picos=gate_spec.gate_duration_picos)

    # TODO(#5050) Add GlobalPhaseGate support

    return cirq.Gateset(*gates_list), gate_durations


def _build_compilation_target_gatesets(
    gateset: cirq.Gateset,
) -> Sequence[cirq.CompilationTargetGateset]:
    """Detects compilation target gatesets based on what gates are inside the gateset."""

    # Include a particular target gateset if the device's gateset contains all required gates of
    # the target gateset.
    # Set all remaining gates in the device's gateset as `additional_gates` so that they are not
    # decomposed in the transformation process.
    target_gatesets: List[cirq.CompilationTargetGateset] = []
    if all(gate_family in gateset.gates for gate_family in _CZ_TARGET_GATES):
        target_gatesets.append(
            transformers.GoogleCZTargetGateset(
                additional_gates=list(gateset.gates - set(_CZ_TARGET_GATES))
            )
        )
    if all(gate_family in gateset.gates for gate_family in _SYC_TARGET_GATES):
        # TODO(#5050) SycamoreTargetGateset additional gates
        target_gatesets.append(transformers.SycamoreTargetGateset())
    if all(gate_family in gateset.gates for gate_family in _SQRT_ISWAP_TARGET_GATES):
        target_gatesets.append(
            cirq.SqrtIswapTargetGateset(
                additional_gates=list(gateset.gates - set(_SQRT_ISWAP_TARGET_GATES))
            )
        )
    if all(gate_family in gateset.gates for gate_family in _CZ_POW_TARGET_GATES):
        target_gatesets.append(
            cirq.CZTargetGateset(
                allow_partial_czs=True,
                additional_gates=list(gateset.gates - set(_CZ_POW_TARGET_GATES)),
            )
        )

    return tuple(target_gatesets)


@cirq.value_equality
class GridDevice(cirq.Device):
    """Device object representing Google devices with a grid qubit layout.

    For end users, instances of this class are typically accessed via
    `Engine.get_processor('processor_name').get_device()`.

    This class is compliant with the core `cirq.Device` abstraction. In particular:
        * Device information is captured in the `metadata` property.
        * An instance of `GridDevice` can be used to validate circuits, moments, and operations.

    Example use cases:

        Get an instance of a Google grid device.
        >>> device = cirq_google.engine.create_device_from_processor_id("rainbow")

        Print the grid layout of the device.
        >>> print(device)
                          (3, 2)
                          │
                          │
                 (4, 1)───(4, 2)───(4, 3)
                 │        │        │
                 │        │        │
        (5, 0)───(5, 1)───(5, 2)───(5, 3)───(5, 4)
                 │        │        │        │
                 │        │        │        │
                 (6, 1)───(6, 2)───(6, 3)───(6, 4)───(6, 5)
                          │        │        │        │
                          │        │        │        │
                          (7, 2)───(7, 3)───(7, 4)───(7, 5)───(7, 6)
                                   │        │        │
                                   │        │        │
                                   (8, 3)───(8, 4)───(8, 5)
                                            │
                                            │
                                            (9, 4)

        Determine whether a circuit can be run on the device.
        >>> circuit = cirq.Circuit(cirq.X(cirq.q(5, 1)))
        >>> device.validate_circuit(circuit)  # Raises a ValueError if the circuit is invalid.

        Determine whether an operation can be run on the device.
        >>> operation = cirq.X(cirq.q(5, 1))
        >>> device.validate_operation(operation)  # Raises a ValueError if the operation is invalid.

        Get the `cirq.Gateset` containing valid gates for the device, and inspect the full list
        of valid gates.
        >>> gateset = device.metadata.gateset
        >>> print(gateset)
        Gateset:...

        Determine whether a gate is available on the device.
        >>> gate = cirq.X
        >>> gate in device.metadata.gateset
        True

        * Get a collection of valid qubits on the device.
        >>> device.metadata.qubit_set
        frozenset({...cirq.GridQubit(6, 4)...})

        * Get a collection of valid qubit pairs for two-qubit gates.
        >>> device.metadata.qubit_pairs
        frozenset({...})

        * Get a collection of isolated qubits, i.e. qubits which are not part of any qubit pair.
        >>> device.metadata.isolated_qubits
        frozenset()

        * Get a collection of approximate gate durations for every gate supported by the device.
        >>> device.metadata.gate_durations
        {...cirq.Duration...}

        * Get a collection of valid CompilationTargetGatesets for the device, which can be used to
          transform a circuit to one which only contains gates from a native target gateset
          supported by the device.
        >>> device.metadata.compilation_target_gatesets
        (...cirq_google.GoogleCZTargetGateset...)

        * Assuming valid CompilationTargetGatesets exist for the device, select the first one and
          use it to transform a circuit to one which only contains gates from a native target
          gateset supported by the device.
        >>> circuit = cirq.optimize_for_target_gateset(
        ...     circuit,
        ...     gateset=device.metadata.compilation_target_gatesets[0]
        ... )
        >>> print(circuit)
        (5, 1): ───PhXZ(a=0,x=1,z=0)───

    Notes about CompilationTargetGatesets:

    * If a device contains gates which yield multiple compilation target gatesets, the user can only
      choose one target gateset to compile to. For example, a device may contain both SYC and
      SQRT_ISWAP gates which yield two separate target gatesets, but a circuit can only be compiled
      to either SYC or SQRT_ISWAP for its two-qubit gates, not both.
    * For a given compilation target gateset, gates which are part of the device's gateset but not
      the target gateset are not decomposed. However, they may still be merged with other gates in
      the circuit.
    * A circuit which contains `cirq.WaitGate`s will be dropped if it is transformed using
      CompilationTargetGatesets generated by GridDevice. To better control circuit timing, insert
      WaitGates after the circuit has been transformed.

    Notes for cirq_google internal implementation:

    For Google devices, the
    [DeviceSpecification proto](
        https://github.com/quantumlib/Cirq/blob/main/cirq-google/cirq_google/api/v2/device.proto
    )
    is the main specification for device information surfaced by the Quantum Computing Service.
    Thus, this class should typically be instantiated using a `DeviceSpecification` proto via the
    `from_proto()` class method.
    """

    def __init__(self, metadata: cirq.GridDeviceMetadata):
        """Creates a GridDevice object.

        This constructor should not be used directly outside the class implementation. Use
        `from_proto()` instead.
        """
        self._metadata = metadata

    @classmethod
    def from_proto(cls, proto: v2.device_pb2.DeviceSpecification) -> 'GridDevice':
        """Deserializes the `DeviceSpecification` to a `GridDevice`.

        Args:
            proto: The `DeviceSpecification` proto describing a Google device.

        Raises:
            ValueError: If the given `DeviceSpecification` is invalid. It is invalid if:
                * A `DeviceSpecification.valid_qubits` string is not in the form `<int>_<int>`, thus
                  cannot be parsed as a `cirq.GridQubit`.
                * `DeviceSpecification.valid_targets` refer to qubits which are not in
                  `DeviceSpecification.valid_qubits`.
                * A target set in `DeviceSpecification.valid_targets` has type `SYMMETRIC` but
                  contains targets with repeated qubits, e.g. a qubit pair with a self loop.
        """

        _validate_device_specification(proto)

        # Create qubit set
        all_qubits = {v2.grid_qubit_from_proto_id(q) for q in proto.valid_qubits}

        # Create qubit pair set
        qubit_pairs = [
            (v2.grid_qubit_from_proto_id(target.ids[0]), v2.grid_qubit_from_proto_id(target.ids[1]))
            for ts in proto.valid_targets
            for target in ts.targets
            if len(target.ids) == 2 and ts.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC
        ]

        gateset, gate_durations = _deserialize_gateset_and_gate_durations(proto)

        try:
            metadata = cirq.GridDeviceMetadata(
                qubit_pairs=qubit_pairs,
                gateset=gateset,
                gate_durations=gate_durations if len(gate_durations) > 0 else None,
                all_qubits=all_qubits,
                compilation_target_gatesets=_build_compilation_target_gatesets(gateset),
            )
        except ValueError as ve:  # pragma: no cover
            # Spec errors should have been caught in validation above.
            raise ValueError("DeviceSpecification is invalid.") from ve  # pragma: no cover

        return GridDevice(metadata)

    def to_proto(
        self, out: Optional[v2.device_pb2.DeviceSpecification] = None
    ) -> v2.device_pb2.DeviceSpecification:
        """Serializes the GridDevice to a DeviceSpecification.

        Args:
            out: Optional DeviceSpecification to be populated. Fields are populated in-place.

        Returns:
            The populated DeviceSpecification if out is specified, or the newly created
            DeviceSpecification.
        """
        qubits = self._metadata.qubit_set
        unordered_pairs = [tuple(pair_set) for pair_set in self._metadata.qubit_pairs]
        pairs = sorted((q0, q1) if q0 <= q1 else (q1, q0) for q0, q1 in unordered_pairs)
        gateset = self._metadata.gateset
        gate_durations = self._metadata.gate_durations

        if out is None:
            out = v2.device_pb2.DeviceSpecification()

        # If fields are already filled (i.e. as part of the old DeviceSpecification format), leave
        # them as is. Fields populated in the new format do not conflict with how they were
        # populated in the old format.
        # TODO(#5050) remove empty checks below once deprecated fields in DeviceSpecification are
        # removed.

        if not out.valid_qubits:
            known_devices.populate_qubits_in_device_proto(qubits, out)
        if not out.valid_targets:
            known_devices.populate_qubit_pairs_in_device_proto(pairs, out)
        _serialize_gateset_and_gate_durations(
            out, gateset, {} if gate_durations is None else gate_durations
        )
        _validate_device_specification(out)

        return out

    @classmethod
    def _from_device_information(
        cls,
        *,
        qubit_pairs: Collection[Tuple[cirq.GridQubit, cirq.GridQubit]],
        gateset: cirq.Gateset,
        gate_durations: Optional[Mapping[cirq.GateFamily, cirq.Duration]] = None,
        all_qubits: Optional[Collection[cirq.GridQubit]] = None,
    ) -> 'GridDevice':
        """Constructs a GridDevice using the device information provided.

        EXPERIMENTAL: this method may have changes which are not backward compatible in the future.

        This is a convenience method for constructing a GridDevice given partial gateset and
        gate_duration information: for every distinct gate, only one representation needs to be in
        gateset and gate_duration. The remaining representations will be automatically generated.

        For example, if the input gateset contains only `cirq.PhasedXZGate`, and the input
        gate_durations is `{cirq.GateFamily(cirq.PhasedXZGate): cirq.Duration(picos=3)}`,
        `GridDevice.metadata.gateset` will be

        ```
        cirq.Gateset(cirq.PhasedXZGate, cirq.XPowGate, cirq.YPowGate, cirq.PhasedXPowGate)
        ```

        and `GridDevice.metadata.gate_durations` will be

        ```
        {
            cirq.GateFamily(cirq.PhasedXZGate): cirq.Duration(picos=3),
            cirq.GateFamily(cirq.XPowGate): cirq.Duration(picos=3),
            cirq.GateFamily(cirq.YPowGate): cirq.Duration(picos=3),
            cirq.GateFamily(cirq.PhasedXPowGate): cirq.Duration(picos=3),
        }
        ```

        This method reduces the complexity of constructing `GridDevice` on server side by requiring
        only the bare essential device information.

        Args:
            qubit_pairs: Collection of bidirectional qubit couplings available on the device.
            gateset: The gate set supported by the device.
            gate_durations: Optional mapping from gates supported by the device to their timing
                estimates. Not every gate is required to have an associated duration.
            out: If set, device information will be serialized into this DeviceSpecification.

        Raises:
            ValueError: If a pair contains two identical qubits.
            ValueError: If `gateset` contains invalid GridDevice gates.
            ValueError: If `gate_durations` contains keys which are not in `gateset`.
            ValueError: If multiple gate families in gate_durations can
                represent a particular gate, but they have different durations.
            ValueError: If all_qubits is provided and is not a superset
                of all the qubits found in qubit_pairs.
        """
        metadata = cirq.GridDeviceMetadata(
            qubit_pairs=qubit_pairs,
            gateset=gateset,
            gate_durations=gate_durations,
            all_qubits=all_qubits,
        )
        incomplete_device = GridDevice(metadata)
        # incomplete_device may have incomplete gateset and gate durations information, as described
        # in the docstring.
        # To generate the full gateset and gate durations, we rely on the device deserialization
        # logic by first serializing then deserializing the fake device, to ensure that the
        # resulting device is consistent with one that is deserialized from DeviceSpecification.
        return GridDevice.from_proto(incomplete_device.to_proto())

    @property
    def metadata(self) -> cirq.GridDeviceMetadata:
        """Get metadata information for the device."""
        return self._metadata

    def validate_operation(self, operation: cirq.Operation) -> None:
        """Raises an exception if an operation is not valid.

        An operation is valid if
            * The operation is in the device gateset.
            * The operation targets a valid qubit
            * The operation targets a valid qubit pair, if it is a two-qubit operation.

        Args:
            operation: The operation to validate.

        Raises:
            ValueError: The operation isn't valid for this device.
        """

        if operation not in self._metadata.gateset:
            raise ValueError(f'Operation {operation} contains a gate which is not supported.')

        for q in operation.qubits:
            if isinstance(q, ops.Coupler):
                if any(qc not in self._metadata.qubit_set for qc in q.qubits):
                    raise ValueError(f'Qubits on coupler not on device: {q.qubits}.')
                if frozenset(q.qubits) not in self._metadata.qubit_pairs:
                    raise ValueError(f'Coupler pair is not valid on device: {q.qubits}.')
            elif q not in self._metadata.qubit_set:
                raise ValueError(f'Qubit not on device: {q!r}.')

        if (
            len(operation.qubits) == 2
            and not any(operation in gf for gf in _VARIADIC_GATE_FAMILIES)
            and frozenset(operation.qubits) not in self._metadata.qubit_pairs
        ):
            raise ValueError(f'Qubit pair is not valid on device: {operation.qubits!r}.')

    def __str__(self) -> str:
        diagram = cirq.TextDiagramDrawer()

        qubits = cast(Set[cirq.GridQubit], self._metadata.qubit_set)

        # Don't print out extras newlines if the row/col doesn't start at 0
        min_col = min(q.col for q in qubits)
        min_row = min(q.row for q in qubits)

        for q in qubits:
            info = cirq.circuit_diagram_info(q, default=None)
            qubit_name = info.wire_symbols[0] if info else str(q)
            diagram.write(q.col - min_col, q.row - min_row, qubit_name)

        # Find pairs that are connected by two-qubit gates.
        Pair = Tuple[cirq.GridQubit, cirq.GridQubit]
        pairs = sorted({cast(Pair, tuple(pair)) for pair in self._metadata.qubit_pairs})

        # Draw lines between connected pairs. Limit to horizontal/vertical
        # lines since that is all the diagram drawer can handle.
        for q1, q2 in pairs:
            if q1.row == q2.row or q1.col == q2.col:
                diagram.grid_line(
                    q1.col - min_col, q1.row - min_row, q2.col - min_col, q2.row - min_row
                )

        return diagram.render(horizontal_spacing=3, vertical_spacing=2, use_unicode_characters=True)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Creates ASCII diagram for Jupyter, IPython, etc."""
        # There should never be a cycle, but just in case use the default repr.
        p.text(repr(self) if cycle else str(self))

    def __repr__(self) -> str:
        return f'cirq_google.GridDevice({repr(self._metadata)})'

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self):
        return {'metadata': self._metadata}

    @classmethod
    def _from_json_dict_(cls, metadata, **kwargs):
        return cls(metadata)

    def _value_equality_values_(self):
        return self._metadata
