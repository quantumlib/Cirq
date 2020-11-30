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
"""A protocol that wouldn't exist if python had __rimul__."""

import pytest

import cirq


def test_act_on_checks():
    class Bad:
        def _act_on_(self, args):
            return False

        def _act_on_fallback_(self, action, allow_decompose):
            return False

    with pytest.raises(ValueError, match="must return True or NotImplemented"):
        _ = cirq.act_on(Bad(), object())

    with pytest.raises(ValueError, match="must return True or NotImplemented"):
        _ = cirq.act_on(object(), Bad())
