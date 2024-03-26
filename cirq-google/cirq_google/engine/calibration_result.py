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
#
import dataclasses
import datetime
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import cirq_google


@dataclasses.dataclass
class CalibrationResult:
    """Python implementation of the proto found in
    cirq_google.api.v2.calibration_pb2.CalibrationLayerResult for use
    in Engine calls.

    Note that, if these fields are not filled out by the calibration API,
    they will be set to the default values in the proto, as defined here:
    https://developers.google.com/protocol-buffers/docs/proto3#default
    These defaults will converted to `None` by the API client.

    Deprecated: Calibrations are no longer supported via cirq.
    """

    code: Any
    error_message: Optional[str]
    token: Optional[str]
    valid_until: Optional[datetime.datetime]
    metrics: 'cirq_google.Calibration'

    @classmethod
    def _from_json_dict_(
        cls,
        code: Any,
        error_message: Optional[str],
        token: Optional[str],
        utc_valid_until: float,
        metrics: 'cirq_google.Calibration',
        **kwargs,
    ) -> 'CalibrationResult':
        """Magic method for the JSON serialization protocol."""
        valid_until = (
            datetime.datetime.utcfromtimestamp(utc_valid_until)
            if utc_valid_until is not None
            else None
        )
        return cls(code, error_message, token, valid_until, metrics)

    def _json_dict_(self) -> Dict[str, Any]:
        """Magic method for the JSON serialization protocol."""
        utc_valid_until = (
            self.valid_until.replace(tzinfo=datetime.timezone.utc).timestamp()
            if self.valid_until is not None
            else None
        )
        return {
            'code': self.code,
            'error_message': self.error_message,
            'token': self.token,
            'utc_valid_until': utc_valid_until,
            'metrics': self.metrics,
        }
