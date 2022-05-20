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

import pytest
import numpy as np
import cirq


def test_assert_consistent_channel_valid():
    channel = cirq.KrausChannel(kraus_ops=(np.array([[0, 1], [0, 0]]), np.array([[1, 0], [0, 0]])))
    cirq.testing.assert_consistent_channel(channel)


def test_assert_consistent_channel_tolerances():
    # This channel is off by 1e-5 from the identity matrix in the consistency condition.
    channel = cirq.KrausChannel(
        kraus_ops=(np.array([[0, np.sqrt(1 - 1e-5)], [0, 0]]), np.array([[1, 0], [0, 0]]))
    )
    # We are comparing to identity, so rtol is same as atol for non-zero entries.
    cirq.testing.assert_consistent_channel(channel, rtol=1e-5, atol=0)
    with pytest.raises(AssertionError):
        cirq.testing.assert_consistent_channel(channel, rtol=1e-6, atol=0)
    cirq.testing.assert_consistent_channel(channel, rtol=0, atol=1e-5)
    with pytest.raises(AssertionError):
        cirq.testing.assert_consistent_channel(channel, rtol=0, atol=1e-6)


def test_assert_consistent_channel_invalid():
    channel = cirq.KrausChannel(kraus_ops=(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [0, 0]])))
    with pytest.raises(AssertionError, match=r"cirq.KrausChannel.*2 1"):
        cirq.testing.assert_consistent_channel(channel)


def test_assert_consistent_channel_not_kraus():
    with pytest.raises(AssertionError, match="12.*has_kraus"):
        cirq.testing.assert_consistent_channel(12)
