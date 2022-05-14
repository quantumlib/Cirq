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

import numpy as np

import cirq


def assert_consistent_channel(gate: Any, rtol: float = 1e-5, atol: float = 1e-8):
    """Asserts that a given gate has Kraus operators and that they are properly normalized."""
    assert cirq.has_kraus(gate), f"Given gate {gate!r} does not return True for cirq.has_kraus."
    kraus_ops = cirq.kraus(gate)
    assert cirq.is_cptp(kraus_ops=kraus_ops, rtol=rtol, atol=atol), (
        f"Kraus operators for {gate!r} did not sum to identity up to expected tolerances. "
        f"Summed to {sum(m.T.conj() @ m for m in kraus_ops)}"
    )


def assert_consistent_mixture(gate: Any, rtol: float = 1e-5, atol: float = 1e-8):
    """Asserts that a given gate is a mixture and the mixture probabilities sum to one."""
    assert cirq.has_mixture(gate), f"Give gate {gate!r} does not return for cirq.has_mixture."
    mixture = cirq.mixture(gate)
    total = np.sum(k for k, v in mixture)
    assert total - 1 <= atol + rtol * np.abs(total), (
        f"The mixture for gate {gate!r} did not return coefficients that sum to 1. Summed to "
        f"{total}."
    )
