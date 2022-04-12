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

from typing import Any

import cirq


def assert_consistent_channel(gate: Any, rtol: float = 1e-5, atol: float = 1e-8):
    assert cirq.has_kraus(gate), f"Given gate {gate!r} does not return True cirq.has_kraus."
    kraus_ops = cirq.kraus(gate)
    assert cirq.is_cptp(kraus_ops=kraus_ops, rtol=rtol, atol=atol), (
        f"Kraus operators for {gate!r} did not sum to identity up to expected tolerances. "
        f"Summed to {sum(m.T.conj() @ m for m in kraus_ops)}"
    )
