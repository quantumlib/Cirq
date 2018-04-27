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

"""Package for handling a full quantum job.

Types and methods related to transforming circuits in preparation of sending
them to Quantum Engine.  Contains classes to help with adding parameter
sweeps and error simulation.
"""

from cirq.contrib.jobs.job import (
    Job,
)
from cirq.contrib.jobs.depolarizer_channel import (
    DepolarizerChannel,
)
