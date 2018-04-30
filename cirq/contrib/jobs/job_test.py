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

from cirq import ops
from cirq.circuits.circuit import Circuit
from cirq.circuits.moment import Moment
from cirq.contrib.jobs import Job
from cirq.study import sweeps
from cirq.testing import EqualsTester


def test_job_equality():
    eq = EqualsTester()
    q = ops.QubitId()
    q2 = ops.QubitId()

    # Equivalent empty jobs
    eq.add_equality_group(Job(),
                          Job(Circuit()),
                          Job(Circuit([])),
                          Job(Circuit(), sweeps.Unit))

    # Equivalent circuit, different instances
    eq.add_equality_group(Job(Circuit([Moment([ops.Z(q)])])),
                          Job(Circuit([Moment([ops.Z(q)])])))
    # Different Circuit
    c = Circuit([Moment([ops.CZ(q, q2)])])
    eq.add_equality_group(Job(c))

    ps1 = sweeps.Points('Example', [42.0])
    ps2 = sweeps.Points('Example', [42.0])
    ps3 = sweeps.Points('Example', [42.0, 1.4])
    eq.add_equality_group(Job(c, ps1, 2), Job(c, ps2, 2))
    eq.add_equality_group(Job(c, ps1, 4))
    eq.add_equality_group(Job(c, ps3, 2))

