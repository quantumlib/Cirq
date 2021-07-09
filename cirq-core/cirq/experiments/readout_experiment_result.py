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

from typing import Any, Dict, TYPE_CHECKING

import dataclasses

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass(frozen=True)
class ReadoutExperimentResult:
    """Result of estimating single qubit readout error.

    Attributes:
        zero_state_errors: A dictionary from qubit to probability of measuring
            a 1 when the qubit is initialized to |0⟩.
        one_state_errors: A dictionary from qubit to probability of measuring
            a 0 when the qubit is initialized to |1⟩.
        repetitions: The number of repetitions that were used to estimate the
            probabilities.
        timestamp: The time the data was taken, in seconds since the epoch.
    """

    zero_state_errors: Dict['cirq.Qid', float]
    one_state_errors: Dict['cirq.Qid', float]
    repetitions: int
    timestamp: float

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'zero_state_errors': list(self.zero_state_errors.items()),
            'one_state_errors': list(self.one_state_errors.items()),
            'repetitions': self.repetitions,
            'timestamp': self.timestamp,
        }

    @classmethod
    def _from_json_dict_(
        cls, zero_state_errors, one_state_errors, repetitions, timestamp, **kwargs
    ):
        return cls(
            zero_state_errors=dict(zero_state_errors),
            one_state_errors=dict(one_state_errors),
            repetitions=repetitions,
            timestamp=timestamp,
        )

    def __repr__(self) -> str:
        return (
            'cirq.experiments.ReadoutExperimentResult('
            f'zero_state_errors={self.zero_state_errors!r}, '
            f'one_state_errors={self.one_state_errors!r}, '
            f'repetitions={self.repetitions!r}, '
            f'timestamp={self.timestamp!r})'
        )
