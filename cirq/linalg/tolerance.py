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

"""A type for specifying thresholds for doing approximate equality."""

import numpy as np


class Tolerance:
    """Specifies thresholds for doing approximate equality."""

    ZERO = None
    DEFAULT = None

    def __init__(self,
                 rtol: float = 1e-5,
                 atol: float = 1e-8,
                 equal_nan: bool = False):
        """Initializes a Tolerance instance with the specified parameters.

        Notes:
          Comparisons are done as if by numpy.allclose, which considers x and y
          to be close when abs(x - y) <= atol + rtol * abs(y). See
          numpy.allclose's documentation for more details.

        Args:
          rtol: Relative tolerance.
          atol: Absolute tolerance.
          equal_nan: Whether NaNs are equal to each other.
        """
        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan

    def all_close(self, a, b):
        return np.allclose(
            a, b, rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan)

    def all_near_zero(self, a):
        return self.all_close(a, np.zeros(np.shape(a)))

    def __repr__(self):
        return "Tolerance(rtol={}, atol={}, equal_nan={})".format(
            repr(self.rtol), repr(self.atol), repr(self.equal_nan))


Tolerance.ZERO = Tolerance(rtol=0, atol=0)
Tolerance.DEFAULT = Tolerance()
