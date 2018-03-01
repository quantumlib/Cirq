# Copyright 2017 Google LLC
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
from cirq.api.google.v1.params_pb2 import ParameterSweep
from cirq.circuits.circuit import Circuit
from cirq.circuits.moment import Moment
from cirq.contrib import Job
from cirq.testing import EqualsTester


def test_job_equality():
    eq = EqualsTester()
    q = ops.QubitId()
    q2 = ops.QubitId()

    # Equivalent empty jobs
    eq.add_equality_group(Job(),
                          Job(Circuit()),
                          Job(Circuit([])),
                          Job(Circuit(), ParameterSweep()))

    # Equivalent circuit, different instances
    eq.add_equality_group(Job(Circuit([Moment([ops.Z(q)])])),
                          Job(Circuit([Moment([ops.Z(q)])])))
    # Different Circuit
    c = Circuit([Moment([ops.CZ(q, q2)])])
    eq.add_equality_group(Job(c))

    # Example ParameterSweep
    ps = ParameterSweep()
    ps.repetitions = 2
    ps.sweep.factors.add()
    ps.sweep.factors[0].sweeps.add()
    ps.sweep.factors[0].sweeps[0].parameter_name = "Example"
    ps.sweep.factors[0].sweeps[0].sweep_points.points.append(42.0)

    # Equivalent ParameterSweep arguments
    ps_copy = ParameterSweep()
    ps_copy.CopyFrom(ps)
    eq.add_equality_group(Job(c, ps), Job(c, ps_copy))

    # Different ParameterSweeps
    ps_different1 = ParameterSweep()
    ps_different1.CopyFrom(ps)
    ps_different1.repetitions = 4
    eq.add_equality_group(Job(c, ps_different1))
    ps_different2 = ParameterSweep()
    ps_different2.CopyFrom(ps)
    ps_different2.sweep.factors[0].sweeps[0].sweep_points.points.append(1.4)
    eq.add_equality_group(Job(c, ps_different2))

