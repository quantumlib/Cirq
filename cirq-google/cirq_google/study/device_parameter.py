# Copyright 2023 The Cirq Developers
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

from typing import Any, Dict, Optional, Sequence
import dataclasses
from typing_extensions import Protocol
import cirq


class SupportsDeviceParameter(Protocol):
    """Protocol for using device parameter keys.

    Args:
       path: path of the key to modify, with each sub-folder as a string
           entry in a list.
       idx: If this key is an array, which index to modify.
       value: value of the parameter to be set, if any.
    """

    path: Sequence[str]
    idx: Optional[int] = None
    value: Optional[Any] = None


@dataclasses.dataclass
class DeviceParameter(SupportsDeviceParameter):
    """Class for specifying device parameters.

    For instance, varying the length of pulses, timing, etc.
    This class is intended to be attached to a cirq.Points
    or cirq.Linspace sweep object as a metadata attribute.

    Args:
       path: path of the key to modify, with each sub-folder as a string
           entry in a list.
       idx: If this key is an array, which index to modify.
       value: value of the parameter to be set, if any.
       units: string value of the unit type of the value, if any.
          For instance, "GHz", "MHz", "ns", etc.
    """

    path: Sequence[str]
    idx: Optional[int] = None
    value: Optional[Any] = None
    units: Optional[str] = None

    def __repr__(self) -> str:
        return (
            "cirq_google.study.DeviceParameter("
            f"path={self.path!r}, idx={self.idx}, value={self.value!r}, units={self.units!r})"
        )

    @classmethod
    def _json_namespace_(cls) -> str:
        return "cirq.google"

    @classmethod
    def _from_json_dict_(cls, path, idx, value, **kwargs):
        return DeviceParameter(
            path=path, idx=idx, value=value, units=kwargs["units"] if "units" in kwargs else None
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["path", "idx", "value", "units"])


@dataclasses.dataclass
class Metadata:
    """A dataclass holds extra information for sweeps.

    Args:
        device_parameters: If presents, it means it is reg_param sweep.
        is_const: If true, the associated sweep value will be put in parameters instead of axes.
        label: If presents, use it as column name instead of using self._key.
        unit:  If presents, the values in sweep are treated as values with this unit.
            This is a temporary solution. This can be avoided if the values use tunits.
            In this case, we should not keep unit information in metadata.
    """

    device_parameters: Optional[Sequence[DeviceParameter]] = None
    is_const: bool = False
    label: Optional[str] = None
    unit: Optional[str] = None

    def __repr__(self) -> str:
        return (
            "cirq_google.study.Metadata("
            f"device_parameters={self.device_parameters!r}, is_const={self.is_const}, "
            f"label={self.label!r}, unit={self.unit})"
        )

    @classmethod
    def _json_namespace_(cls) -> str:
        return "cirq.google"

    @classmethod
    def _from_json_dict_(
        cls,
        device_parameters: Optional[Sequence[DeviceParameter]] = None,
        is_const: bool = False,
        label: Optional[str] = None,
        unit: Optional[str] = None,
        **kwargs,
    ):
        return Metadata(device_parameters=device_parameters, is_const=is_const, label=label)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ["device_parameters", "is_const", "label", "unit"])
