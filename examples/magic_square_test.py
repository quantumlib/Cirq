# Copyright 2025 The Cirq Developers
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

from __future__ import annotations

import numpy as np

import cirq
import examples.magic_square as ms


def test_run_magic_square_game() -> None:
    """Test that Alice and Bob win 100% of the time with a noiseless simulator."""

    sampler = cirq.Simulator()
    alice_qubits = cirq.GridQubit.rect(1, 4, 0, 0)  # test with 2 measure qubits
    bob_qubits = cirq.GridQubit.rect(1, 4, 1, 0)
    for add_dd in [False, True]:
        result = ms.run_magic_square_game(sampler, alice_qubits, bob_qubits, add_dd=add_dd)
        assert np.all(result.get_win_matrix() == np.ones((3, 3)))
    ms.main()
