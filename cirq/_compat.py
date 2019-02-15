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

"""Workarounds for differences between version 2 and 3 of python."""

import sys


# Python 3 deprecates `fractions.gcd` in factor of `math.gcd`. Also, `math.gcd`
# never returns a negative result whereas `fractions.gcd` seems to match the
# sign of the second argument.

# coverage: ignore
if sys.version_info < (3,):
    import fractions   # pylint: disable=unused-import

    def gcd(a, b):
        return abs(fractions.gcd(a, b))
else:
    from math import gcd  # pylint: disable=unused-import
