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

"""Tests for processor_config classes."""

from __future__ import annotations

import cirq_google as cg

import datetime

def test_get_config_returns_existing_processor_config(self):
    p1 = cg.engine.ProcessorConfig(name="p1", effective_device=None, calibraion=None)
    p2 = cg.engine.ProcessorConfig(name="p2", effective_device=None, calibraion=None)

    snapshot = cg.ProcessorConfigSnapshot(snapshot_id="test_snap", create_time=datetime.datetime.now, run_names=[], processor_configs=[p1, p2])

    assert snapshot.get_config("p2") == p2