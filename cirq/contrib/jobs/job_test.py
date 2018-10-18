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

import cirq
from cirq.contrib.jobs import Job


def test_job_equality():
    eq = cirq.testing.EqualsTester()
    q = cirq.NamedQubit('q')
    q2 = cirq.NamedQubit('q2')

    # Equivalent empty jobs
    eq.add_equality_group(Job(),
                          Job(cirq.Circuit()),
                          Job(cirq.Circuit([])),
                          Job(cirq.Circuit(), cirq.UnitSweep))

    # Equivalent circuit, different instances
    eq.add_equality_group(Job(cirq.Circuit([cirq.Moment([cirq.Z(q)])])),
                          Job(cirq.Circuit([cirq.Moment([cirq.Z(q)])])))
    # Different Circuit
    c = cirq.Circuit([cirq.Moment([cirq.CZ(q, q2)])])
    eq.add_equality_group(Job(c))

    ps1 = cirq.Points('Example', [42.0])
    ps2 = cirq.Points('Example', [42.0])
    ps3 = cirq.Points('Example', [42.0, 1.4])
    eq.add_equality_group(Job(c, ps1, 2), Job(c, ps2, 2))
    eq.add_equality_group(Job(c, ps1, 4))
    eq.add_equality_group(Job(c, ps3, 2))

