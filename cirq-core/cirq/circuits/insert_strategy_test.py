# Copyright 2018 The Cirq Developers
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

import pickle

import pytest

import cirq


def test_repr():
    assert repr(cirq.InsertStrategy.NEW) == 'cirq.InsertStrategy.NEW'


@pytest.mark.parametrize(
    'strategy',
    [
        cirq.InsertStrategy.NEW,
        cirq.InsertStrategy.NEW_THEN_INLINE,
        cirq.InsertStrategy.INLINE,
        cirq.InsertStrategy.EARLIEST,
    ],
    ids=lambda strategy: strategy.name,
)
def test_identity_after_pickling(strategy: cirq.InsertStrategy):
    unpickled_strategy = pickle.loads(pickle.dumps(strategy))
    assert unpickled_strategy is strategy
