# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import google.protobuf.internal.containers
import google.protobuf.message
import typing

class ParameterSweep(google.protobuf.message.Message):
    repetitions = ... # type: int

    @property
    def sweep(self) -> ProductSweep: ...

    def __init__(self,
        repetitions : typing.Optional[int] = None,
        sweep : typing.Optional[ProductSweep] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ParameterSweep: ...
    def MergeFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
    def CopyFrom(self, other_msg: google.protobuf.message.Message) -> None: ...

class ProductSweep(google.protobuf.message.Message):

    @property
    def factors(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[ZipSweep]: ...

    def __init__(self,
        factors : typing.Optional[typing.Iterable[ZipSweep]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ProductSweep: ...
    def MergeFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
    def CopyFrom(self, other_msg: google.protobuf.message.Message) -> None: ...

class ZipSweep(google.protobuf.message.Message):

    @property
    def sweeps(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[SingleSweep]: ...

    def __init__(self,
        sweeps : typing.Optional[typing.Iterable[SingleSweep]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ZipSweep: ...
    def MergeFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
    def CopyFrom(self, other_msg: google.protobuf.message.Message) -> None: ...

class SingleSweep(google.protobuf.message.Message):
    parameter_key = ... # type: typing.Text

    @property
    def points(self) -> Points: ...

    @property
    def linspace(self) -> Linspace: ...

    def __init__(self,
        parameter_key : typing.Optional[typing.Text] = None,
        points : typing.Optional[Points] = None,
        linspace : typing.Optional[Linspace] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> SingleSweep: ...
    def MergeFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
    def CopyFrom(self, other_msg: google.protobuf.message.Message) -> None: ...

class Points(google.protobuf.message.Message):
    points = ... # type: google.protobuf.internal.containers.RepeatedScalarFieldContainer[float]

    def __init__(self,
        points : typing.Optional[typing.Iterable[float]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Points: ...
    def MergeFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
    def CopyFrom(self, other_msg: google.protobuf.message.Message) -> None: ...

class Linspace(google.protobuf.message.Message):
    first_point = ... # type: float
    last_point = ... # type: float
    num_points = ... # type: int

    def __init__(self,
        first_point : typing.Optional[float] = None,
        last_point : typing.Optional[float] = None,
        num_points : typing.Optional[int] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> Linspace: ...
    def MergeFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
    def CopyFrom(self, other_msg: google.protobuf.message.Message) -> None: ...

class ParameterDict(google.protobuf.message.Message):
    class AssignmentsEntry(google.protobuf.message.Message):
        key = ... # type: typing.Text
        value = ... # type: float

        def __init__(self,
            key : typing.Optional[typing.Text] = None,
            value : typing.Optional[float] = None,
            ) -> None: ...
        @classmethod
        def FromString(cls, s: bytes) -> ParameterDict.AssignmentsEntry: ...
        def MergeFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
        def CopyFrom(self, other_msg: google.protobuf.message.Message) -> None: ...


    @property
    def assignments(self) -> typing.MutableMapping[typing.Text, float]: ...

    def __init__(self,
        assignments : typing.Optional[typing.Mapping[typing.Text, float]] = None,
        ) -> None: ...
    @classmethod
    def FromString(cls, s: bytes) -> ParameterDict: ...
    def MergeFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
    def CopyFrom(self, other_msg: google.protobuf.message.Message) -> None: ...
