"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor = ...

# The context for running a quantum program.
class RunContext(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    PARAMETER_SWEEPS_FIELD_NUMBER: builtins.int
    # The parameters for operations in a program.
    @property
    def parameter_sweeps(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___ParameterSweep]: ...
    def __init__(self,
        *,
        parameter_sweeps : typing.Optional[typing.Iterable[global___ParameterSweep]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"parameter_sweeps",b"parameter_sweeps"]) -> None: ...
global___RunContext = RunContext

# Specifies how to repeatedly sample a circuit, with or without sweeping over
# varying parameter-dicts.
class ParameterSweep(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    REPETITIONS_FIELD_NUMBER: builtins.int
    SWEEP_FIELD_NUMBER: builtins.int
    # How many times to sample, for each parameter-dict that is swept over.
    # This must be set to a value strictly greater than zero.
    repetitions: builtins.int = ...
    # Which parameters, that control gates in the circuit, to try.
    #
    # The keys of the parameters generated by this sweep must be a superset
    # of the keys in the program's operation's Args. When this is not specified,
    # no parameterization is assumed (and the program must have no
    # args with symbols).
    @property
    def sweep(self) -> global___Sweep: ...
    def __init__(self,
        *,
        repetitions : builtins.int = ...,
        sweep : typing.Optional[global___Sweep] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"sweep",b"sweep"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"repetitions",b"repetitions",u"sweep",b"sweep"]) -> None: ...
global___ParameterSweep = ParameterSweep

# A sweep over all of the parameters in a program.
class Sweep(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    SWEEP_FUNCTION_FIELD_NUMBER: builtins.int
    SINGLE_SWEEP_FIELD_NUMBER: builtins.int
    @property
    def sweep_function(self) -> global___SweepFunction: ...
    @property
    def single_sweep(self) -> global___SingleSweep: ...
    def __init__(self,
        *,
        sweep_function : typing.Optional[global___SweepFunction] = ...,
        single_sweep : typing.Optional[global___SingleSweep] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"single_sweep",b"single_sweep",u"sweep",b"sweep",u"sweep_function",b"sweep_function"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"single_sweep",b"single_sweep",u"sweep",b"sweep",u"sweep_function",b"sweep_function"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal[u"sweep",b"sweep"]) -> typing.Optional[typing_extensions.Literal["sweep_function","single_sweep"]]: ...
global___Sweep = Sweep

# A function that takes multiple sweeps and produces more sweeps.
class SweepFunction(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    # The type of sweep function.
    class FunctionType(_FunctionType, metaclass=_FunctionTypeEnumTypeWrapper):
        pass
    class _FunctionType:
        V = typing.NewType('V', builtins.int)
    class _FunctionTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_FunctionType.V], builtins.type):
        DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor = ...
        # The function type is not specified. Should never be used.
        FUNCTION_TYPE_UNSPECIFIED = SweepFunction.FunctionType.V(0)
        # A Cartesian product of parameter sweeps.
        #
        # Example of product:
        # If one of the sweeps assigns
        # "a": 0.0
        # "a": 1.0
        # and another assigns
        # "b": 2.0
        # "b": 3.0
        # then the product of these assigns all possible combinations.
        # "a": 0.0, "b": 2.0
        # "a": 0.0, "b": 3.0
        # "a": 1.0, "b": 2.0
        # "a": 1.0, "b": 3.0
        PRODUCT = SweepFunction.FunctionType.V(1)
        # A zip product of parameter sweeps.
        #
        # Example of zip:
        # If one of the sweeps assigns
        # "a": 0.0
        # "a": 1.0
        # and another assigns
        # "b": 2.0
        # "b": 3.0
        # then the product of these assigns
        # "a": 0.0, "b": 2.0
        # "a": 1.0, "b": 3.0
        # Note: if one sweep is shorter, the others will be truncated.
        ZIP = SweepFunction.FunctionType.V(2)

    # The function type is not specified. Should never be used.
    FUNCTION_TYPE_UNSPECIFIED = SweepFunction.FunctionType.V(0)
    # A Cartesian product of parameter sweeps.
    #
    # Example of product:
    # If one of the sweeps assigns
    # "a": 0.0
    # "a": 1.0
    # and another assigns
    # "b": 2.0
    # "b": 3.0
    # then the product of these assigns all possible combinations.
    # "a": 0.0, "b": 2.0
    # "a": 0.0, "b": 3.0
    # "a": 1.0, "b": 2.0
    # "a": 1.0, "b": 3.0
    PRODUCT = SweepFunction.FunctionType.V(1)
    # A zip product of parameter sweeps.
    #
    # Example of zip:
    # If one of the sweeps assigns
    # "a": 0.0
    # "a": 1.0
    # and another assigns
    # "b": 2.0
    # "b": 3.0
    # then the product of these assigns
    # "a": 0.0, "b": 2.0
    # "a": 1.0, "b": 3.0
    # Note: if one sweep is shorter, the others will be truncated.
    ZIP = SweepFunction.FunctionType.V(2)

    FUNCTION_TYPE_FIELD_NUMBER: builtins.int
    SWEEPS_FIELD_NUMBER: builtins.int
    function_type: global___SweepFunction.FunctionType.V = ...
    # The argument sweeps to the function.
    @property
    def sweeps(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Sweep]: ...
    def __init__(self,
        *,
        function_type : global___SweepFunction.FunctionType.V = ...,
        sweeps : typing.Optional[typing.Iterable[global___Sweep]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"function_type",b"function_type",u"sweeps",b"sweeps"]) -> None: ...
global___SweepFunction = SweepFunction

# A set of values to loop over for a particular parameter.
class SingleSweep(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    PARAMETER_KEY_FIELD_NUMBER: builtins.int
    POINTS_FIELD_NUMBER: builtins.int
    LINSPACE_FIELD_NUMBER: builtins.int
    # The parameter key being varied. This cannot be the empty string.
    # These are must appear as string Args in the quantum program.
    parameter_key: typing.Text = ...
    # An explicit list of points to try.
    @property
    def points(self) -> global___Points: ...
    # Uniformly-spaced sampling over a range.
    @property
    def linspace(self) -> global___Linspace: ...
    def __init__(self,
        *,
        parameter_key : typing.Text = ...,
        points : typing.Optional[global___Points] = ...,
        linspace : typing.Optional[global___Linspace] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"linspace",b"linspace",u"points",b"points",u"sweep",b"sweep"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"linspace",b"linspace",u"parameter_key",b"parameter_key",u"points",b"points",u"sweep",b"sweep"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal[u"sweep",b"sweep"]) -> typing.Optional[typing_extensions.Literal["points","linspace"]]: ...
global___SingleSweep = SingleSweep

# A list of explicit values.
class Points(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    POINTS_FIELD_NUMBER: builtins.int
    # The values.
    @property
    def points(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]: ...
    def __init__(self,
        *,
        points : typing.Optional[typing.Iterable[builtins.float]] = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"points",b"points"]) -> None: ...
global___Points = Points

# A range of evenly-spaced values.
#
# Example: if the first_point is 1.0, the last_point is 2.0 ,
# and the num_points is 5, thi corresponds to the points
#   1.0, 1.25, 1.5, 1.75, 2.0
class Linspace(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    FIRST_POINT_FIELD_NUMBER: builtins.int
    LAST_POINT_FIELD_NUMBER: builtins.int
    NUM_POINTS_FIELD_NUMBER: builtins.int
    # The start of the range.
    first_point: builtins.float = ...
    # The end of the range.
    last_point: builtins.float = ...
    # The number of points in the range (including first and last). Must be
    # greater than zero. If it is 1, the first_point and last_point must be
    # the same.
    num_points: builtins.int = ...
    def __init__(self,
        *,
        first_point : builtins.float = ...,
        last_point : builtins.float = ...,
        num_points : builtins.int = ...,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"first_point",b"first_point",u"last_point",b"last_point",u"num_points",b"num_points"]) -> None: ...
global___Linspace = Linspace
