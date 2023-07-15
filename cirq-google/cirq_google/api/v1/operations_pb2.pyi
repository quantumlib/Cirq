"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import sys

if sys.version_info >= (3, 8):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class Qubit(google.protobuf.message.Message):
    """Identifies a qubit."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    ROW_FIELD_NUMBER: builtins.int
    COL_FIELD_NUMBER: builtins.int
    row: builtins.int
    """row number in grid."""
    col: builtins.int
    """column number in grid."""
    def __init__(
        self,
        *,
        row: builtins.int = ...,
        col: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["col", b"col", "row", b"row"]) -> None: ...

global___Qubit = Qubit

class ParameterizedFloat(google.protobuf.message.Message):
    """A number specified as a constant plus an optional parameter lookup."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    RAW_FIELD_NUMBER: builtins.int
    PARAMETER_KEY_FIELD_NUMBER: builtins.int
    raw: builtins.float
    """A constant value."""
    parameter_key: builtins.str
    """A variable value stored under some parameter key.
    This cannot be the empty string.
    """
    def __init__(
        self,
        *,
        raw: builtins.float = ...,
        parameter_key: builtins.str = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["parameter_key", b"parameter_key", "raw", b"raw", "value", b"value"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["parameter_key", b"parameter_key", "raw", b"raw", "value", b"value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["value", b"value"]) -> typing_extensions.Literal["raw", "parameter_key"] | None: ...

global___ParameterizedFloat = ParameterizedFloat

class ExpW(google.protobuf.message.Message):
    """A single-qubit rotation around an axis on the XY equator of the Bloch sphere.

    This gate is exp(-i * pi * W(theta) * t / 2) where
      W(theta) = cos(pi * theta) X + sin(pi * theta) Y
    or in matrix form
      W(theta) = [[0, cos(pi * theta) - i sin(pi * theta)],
                  [cos(pi * theta) + i sin(pi * theta), 0]]
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TARGET_FIELD_NUMBER: builtins.int
    AXIS_HALF_TURNS_FIELD_NUMBER: builtins.int
    HALF_TURNS_FIELD_NUMBER: builtins.int
    @property
    def target(self) -> global___Qubit:
        """The qubit to rotate."""
    @property
    def axis_half_turns(self) -> global___ParameterizedFloat:
        """The angle of the rotation axis' facing in the XY plane, expressed in
        units of pi. In other words, this is the theta in exp(i pi W(theta) t / 2).
           - 0 is positive-ward along X.
           - 0.5 is positive-ward along Y.
           - 1.0 is negative-ward along X.
           - 1.5 is negative-ward along Y.
        Note that this is periodic with period 2.
        """
    @property
    def half_turns(self) -> global___ParameterizedFloat:
        """The amount to rotate by expressed in units of pi / 2, i.e. the t in
        exp(i pi W(theta) t / 2).
        Note that this is periodic with period 4 (or 2 when ignoring global phase).
        """
    def __init__(
        self,
        *,
        target: global___Qubit | None = ...,
        axis_half_turns: global___ParameterizedFloat | None = ...,
        half_turns: global___ParameterizedFloat | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["axis_half_turns", b"axis_half_turns", "half_turns", b"half_turns", "target", b"target"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["axis_half_turns", b"axis_half_turns", "half_turns", b"half_turns", "target", b"target"]) -> None: ...

global___ExpW = ExpW

class ExpZ(google.protobuf.message.Message):
    """A single-qubit rotation around the Z axis of the Bloch sphere.

    This gate is exp(-i * pi * Z * t / 2) where Z is the Pauli Z matrix,
      Z = [[1, 0], [0, -1]]
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TARGET_FIELD_NUMBER: builtins.int
    HALF_TURNS_FIELD_NUMBER: builtins.int
    @property
    def target(self) -> global___Qubit:
        """The qubit to rotate."""
    @property
    def half_turns(self) -> global___ParameterizedFloat:
        """The amount of the rotation in radians, i.e. the t in
        exp(i * pi * Z * t / 2).
        Note that this is periodic with period 4 (or 2 when ignoring global phase).
        """
    def __init__(
        self,
        *,
        target: global___Qubit | None = ...,
        half_turns: global___ParameterizedFloat | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["half_turns", b"half_turns", "target", b"target"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["half_turns", b"half_turns", "target", b"target"]) -> None: ...

global___ExpZ = ExpZ

class Exp11(google.protobuf.message.Message):
    """A two qubit rotation which acts to phase only the |11> state.

    This gate is exp(i * pi * H  * t) where H = |11><11| or in matrix form
      H = diag(0, 0, 0, 1)
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TARGET1_FIELD_NUMBER: builtins.int
    TARGET2_FIELD_NUMBER: builtins.int
    HALF_TURNS_FIELD_NUMBER: builtins.int
    @property
    def target1(self) -> global___Qubit:
        """The first qubit to interact."""
    @property
    def target2(self) -> global___Qubit:
        """The other qubit to interact."""
    @property
    def half_turns(self) -> global___ParameterizedFloat:
        """The amount of the rotation in units of pi, i.e. the t in
        exp(i * pi * |11><11| * t).
        Note that this is periodic with period 2 (including global phase).
        """
    def __init__(
        self,
        *,
        target1: global___Qubit | None = ...,
        target2: global___Qubit | None = ...,
        half_turns: global___ParameterizedFloat | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["half_turns", b"half_turns", "target1", b"target1", "target2", b"target2"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["half_turns", b"half_turns", "target1", b"target1", "target2", b"target2"]) -> None: ...

global___Exp11 = Exp11

class Measurement(google.protobuf.message.Message):
    """A multi-qubit measurement in the computational basis (|0>, |1>)."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TARGETS_FIELD_NUMBER: builtins.int
    KEY_FIELD_NUMBER: builtins.int
    INVERT_MASK_FIELD_NUMBER: builtins.int
    @property
    def targets(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Qubit]:
        """The qubits to measure."""
    key: builtins.str
    """The key that this measurement's bit will be grouped under.
    Measurement keys must be unique across the circuit.
    """
    @property
    def invert_mask(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.bool]:
        """If not empty, a list of booleans describing whether the results should
        be flipped for each of the qubits above. The length of this vector must
        match the length of the qubits, and the interpretation of whether to
        invert or not matches component-wise this list and the qubits' list.
        """
    def __init__(
        self,
        *,
        targets: collections.abc.Iterable[global___Qubit] | None = ...,
        key: builtins.str = ...,
        invert_mask: collections.abc.Iterable[builtins.bool] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["invert_mask", b"invert_mask", "key", b"key", "targets", b"targets"]) -> None: ...

global___Measurement = Measurement

class Operation(google.protobuf.message.Message):
    """An operation to apply: either a gate or a measurement."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INCREMENTAL_DELAY_PICOSECONDS_FIELD_NUMBER: builtins.int
    EXP_W_FIELD_NUMBER: builtins.int
    EXP_Z_FIELD_NUMBER: builtins.int
    EXP_11_FIELD_NUMBER: builtins.int
    MEASUREMENT_FIELD_NUMBER: builtins.int
    incremental_delay_picoseconds: builtins.int
    """When this operation should be done, relative to the previous operation.
    Use a delay of 0 to apply simultaneous with previous operation.
    (Implies operations must be sorted by application order.)
    """
    @property
    def exp_w(self) -> global___ExpW:
        """A single-qubit rotation around an axis on the XY equator."""
    @property
    def exp_z(self) -> global___ExpZ:
        """A single-qubit rotation around the Z axis."""
    @property
    def exp_11(self) -> global___Exp11:
        """An operation that interacts two qubits, phasing only the 11 state."""
    @property
    def measurement(self) -> global___Measurement:
        """Measures a qubit and indicates where to store the result."""
    def __init__(
        self,
        *,
        incremental_delay_picoseconds: builtins.int = ...,
        exp_w: global___ExpW | None = ...,
        exp_z: global___ExpZ | None = ...,
        exp_11: global___Exp11 | None = ...,
        measurement: global___Measurement | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["exp_11", b"exp_11", "exp_w", b"exp_w", "exp_z", b"exp_z", "measurement", b"measurement", "operation", b"operation"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["exp_11", b"exp_11", "exp_w", b"exp_w", "exp_z", b"exp_z", "incremental_delay_picoseconds", b"incremental_delay_picoseconds", "measurement", b"measurement", "operation", b"operation"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["operation", b"operation"]) -> typing_extensions.Literal["exp_w", "exp_z", "exp_11", "measurement"] | None: ...

global___Operation = Operation
