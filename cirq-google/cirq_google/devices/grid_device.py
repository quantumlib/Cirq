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

from typing import Any, Set, Tuple, cast
import cirq
from cirq_google.api import v2


def _validate_device_specification(proto: v2.device_pb2.DeviceSpecification) -> None:
    """Validates the DeviceSpecification proto.

    Args:
        proto: The DeviceSpecification proto to validate.

    Raises:
        ValueError: If the DeviceSpecification is invalid.

    """

    for target_set in proto.valid_targets:

        # Check for unknown qubits in targets.
        for target in target_set.targets:
            for target_id in target.ids:
                if target_id not in proto.valid_qubits:
                    raise ValueError(
                        f"Invalid DeviceSpecification: valid_targets contain qubit '{target_id}'"
                        " which is not in valid_qubits."
                    )

        # Symmetric and asymmetric targets should not have repeated qubits.
        if (
            target_set.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC
            or target_set.target_ordering == v2.device_pb2.TargetSet.ASYMMETRIC
        ):
            for target in target_set.targets:
                if len(target.ids) > len(set(target.ids)):
                    raise ValueError(
                        f"Invalid DeviceSpecification: the target set '{target_set.name}' is either"
                        " SYMMETRIC or ASYMMETRIC but has a target which contains repeated qubits:"
                        f" {target.ids}."
                    )

        # A SUBSET_PERMUTATION target should contain exactly one qubit.
        if target_set.target_ordering == v2.device_pb2.TargetSet.SUBSET_PERMUTATION:
            for target in target_set.targets:
                if len(target.ids) != 1:
                    raise ValueError(
                        f"Invalid DeviceSpecification: the target set '{target_set.name}' is of"
                        " type SUBSET_PERMUTATION but contains a target which does not have exactly"
                        f" 1 qubit: {target.ids}."
                    )


@cirq.value_equality
class GridDevice(cirq.Device):
    """Device object representing Google devices with a grid qubit layout.

    For end users, instances of this class are typically accessed via
    `Engine.get_processor('processor_name').get_device()`.

    This class is compliant with the core `cirq.Device` abstraction. In particular:
        * Device information is captured in the `metadata` property.
        * An instance of `GridDevice` can be used to validate circuits, moments, and operations.

    Example use cases:

        * Get an instance of a Google grid device.
        >>> device = cirq_google.get_engine().get_processor('weber').get_device()

        * Print the grid layout of the device.
        >>> print(device)

        * Determine whether a circuit can be run on the device.
        >>> device.validate_circuit(circuit)  # Raises an exception if the circuit is invalid.

        * Determine whether an operation can be run on the device.
        >>> device.validate_operation(operation)  # Raises an exception if the operation is invalid.

        * Get the `cirq.Gateset` containing valid gates for the device, and inspect the full list
          of valid gates.
        >>> gateset = device.metadata.gateset
        >>> print(gateset)

        * Determine whether a gate is available on the device.
        >>> gate in device.metadata.gateset

        * Get a collection of valid qubits on the device.
        >>> device.metadata.qubit_set

        * Get a collection of valid qubit pairs for two-qubit gates.
        >>> device.metadata.qubit_pairs

        * Get a collection of isolated qubits, i.e. qubits which are not part of any qubit pair.
        >>> device.metadata.isolated_qubits

        * Get a collection of approximate durations of performing each gate supported by the device.
        >>> device.metadata.gate_durations

        TODO(#5050) Add compilation_target_gatesets example.

    Notes for cirq_google internal implementation:

    For Google devices, the
    [DeviceSpecification proto](
        https://github.com/quantumlib/Cirq/blob/3969c2d3964cea56df33b329f036ba6810683882/cirq-google/cirq_google/api/v2/device.proto#L13
    )
    is the main specification for device information surfaced by the Quantum Computing Service.
    Thus, this class is should be instantiated using a `DeviceSpecification` proto via the
    `from_proto()` class method.
    """

    def __init__(self, metadata: cirq.GridDeviceMetadata):
        """Creates a GridDevice object.

        This constructor typically should not be used directly. Use `from_proto()` instead.
        """
        self._metadata = metadata

    @classmethod
    def from_proto(cls, proto: v2.device_pb2.DeviceSpecification) -> 'GridDevice':
        """Create a `GridDevice` from a DeviceSpecification proto.

        This class only supports `cirq.GridQubit`s and `cirq.NamedQubit`s. If a
        `DeviceSpecification.valid_qubits` string is in the form `<int>_<int>`, it is parsed as a
        GridQubit. Otherwise it is parsed as a NamedQubit.

        Args:
            proto: The `DeviceSpecification` proto describing a Google device.

        Raises:
            ValueError: If the given `DeviceSpecification` is invalid.
        """

        _validate_device_specification(proto)

        # Create qubit set
        all_qubits = [_qid_from_str(q) for q in proto.valid_qubits]

        # Create qubit pair set
        #
        # While the `GateSpecification` proto message contains qubit target references, they are
        # ignored here because the following assumptions make them unnecessary currently:
        # * All valid qubit pairs work for all two-qubit gates.
        # * All valid qubits work for all single-qubit gates.
        # * Measurement gate can always be applied to all subset of qubits.
        #
        qubit_pairs = [
            (_qid_from_str(target.ids[0]), _qid_from_str(target.ids[1]))
            for ts in proto.valid_targets
            for target in ts.targets
            if len(target.ids) == 2 and ts.target_ordering == v2.device_pb2.TargetSet.SYMMETRIC
        ]

        # TODO(#5050) implement gate durations
        try:
            metadata = cirq.GridDeviceMetadata(
                qubit_pairs=qubit_pairs,
                gateset=cirq.Gateset(),  # TODO(#5050) implement
                all_qubits=all_qubits,
            )
        except ValueError as ve:  # coverage: ignore
            # Spec errors should have been caught in validation above.
            raise ValueError("DeviceSpecification is invalid.") from ve  # coverage: ignore

        return GridDevice(metadata)

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
        # TODO(#5050) uncomment once gateset logic is implemented
        # if operation not in self._metadata.gateset:
        #     raise ValueError(f'Operation {operation} is not a supported gate')

        for q in operation.qubits:
            if q not in self._metadata.qubit_set:
                raise ValueError(f'Qubit not on device: {q!r}')

        if (
            len(operation.qubits) == 2
            and frozenset(operation.qubits) not in self._metadata.qubit_pairs
        ):
            raise ValueError(f'Qubit pair is not valid on device: {operation.qubits!r}')

    def __str__(self) -> str:
        # If all qubits are grid qubits, render an appropriate text diagram.
        if all(isinstance(q, cirq.GridQubit) for q in self._metadata.qubit_set):
            diagram = cirq.TextDiagramDrawer()

            qubits = cast(Set[cirq.GridQubit], self._metadata.qubit_set)

            # Don't print out extras newlines if the row/col doesn't start at 0
            min_col = min(q.col for q in qubits)
            min_row = min(q.row for q in qubits)

            for q in qubits:
                diagram.write(q.col - min_col, q.row - min_row, str(q))

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

            return diagram.render(
                horizontal_spacing=3, vertical_spacing=2, use_unicode_characters=True
            )

        return super().__str__()

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Creates ASCII diagram for Jupyter, IPython, etc."""
        # There should never be a cycle, but just in case use the default repr.
        p.text(repr(self) if cycle else str(self))

    def __repr__(self) -> str:
        return f'cirq_google.GridDevice({repr(self._metadata)})'

    def _json_dict_(self):
        return {'metadata': self._metadata}

    @classmethod
    def _from_json_dict_(cls, metadata, **kwargs):
        return cls(metadata)

    def _value_equality_values_(self):
        return self._metadata


def _qid_from_str(id_str: str) -> cirq.Qid:
    """Translates a qubit id string info cirq.Qid objects.

    Tries to translate to GridQubit if possible (e.g. '4_3'), otherwise
    falls back to using NamedQubit.
    """
    try:
        return v2.grid_qubit_from_proto_id(id_str)
    except ValueError:
        return v2.named_qubit_from_proto_id(id_str)
