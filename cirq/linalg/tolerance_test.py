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

from cirq.linalg.tolerance import Tolerance


def test_init():
    tol = Tolerance(rtol=0.01, atol=0.002, equal_nan=False)
    assert tol.rtol == 0.01
    assert tol.atol == 0.002
    assert not tol.equal_nan

    tol = Tolerance(rtol=3, atol=4, equal_nan=True)
    assert tol.rtol == 3
    assert tol.atol == 4
    assert tol.equal_nan

    tol = Tolerance()
    assert tol.rtol == 1e-5
    assert tol.atol == 1e-8
    assert not tol.equal_nan


def test_all_close():
    no_tol = Tolerance()
    assert no_tol.all_close(1, 1)
    assert not no_tol.all_close(1, 0.5)
    assert not no_tol.all_close(1, 1.5)
    assert not no_tol.all_close(1, 100.5)
    assert no_tol.all_close(100, 100)
    assert not no_tol.all_close(100, 99.5)
    assert not no_tol.all_close(100, 100.5)

    assert no_tol.all_close([1, 2], [1, 2])
    assert not no_tol.all_close([1, 2], [1, 3])

    atol5 = Tolerance(atol=5)
    assert atol5.all_close(1, 1)
    assert atol5.all_close(1, 0.5)
    assert atol5.all_close(1, 1.5)
    assert atol5.all_close(1, 5.5)
    assert atol5.all_close(1, -3.5)
    assert not atol5.all_close(1, 6.5)
    assert not atol5.all_close(1, -4.5)
    assert not atol5.all_close(1, 100.5)
    assert atol5.all_close(100, 100)
    assert atol5.all_close(100, 100.5)
    assert atol5.all_close(100, 104.5)
    assert not atol5.all_close(100, 105.5)

    rtol2 = Tolerance(rtol=2)
    assert rtol2.all_close(100, 100)
    assert rtol2.all_close(299.5, 100)
    assert rtol2.all_close(1, 100)
    assert rtol2.all_close(-99, 100)
    assert not rtol2.all_close(100, 1)  # Doesn't commute.
    assert not rtol2.all_close(300.5, 100)
    assert not rtol2.all_close(-101, 100)

    tol25 = Tolerance(rtol=2, atol=5)
    assert tol25.all_close(100, 100)
    assert tol25.all_close(106, 100)
    assert tol25.all_close(201, 100)
    assert tol25.all_close(304.5, 100)
    assert not tol25.all_close(305.5, 100)


def test_all_zero():
    tol = Tolerance(atol=5, rtol=9999999)
    assert tol.all_near_zero(0)
    assert tol.all_near_zero(4.5)
    assert not tol.all_near_zero(5.5)

    assert tol.all_near_zero([-4.5, 0, 1, 4.5, 3])
    assert not tol.all_near_zero([-4.5, 0, 1, 4.5, 30])


def test_all_zero_mod():
    tol = Tolerance(atol=5, rtol=9999999)
    assert tol.all_near_zero_mod(0, 100)
    assert tol.all_near_zero_mod(4.5, 100)
    assert not tol.all_near_zero_mod(5.5, 100)

    assert tol.all_near_zero_mod(100, 100)
    assert tol.all_near_zero_mod(95.5, 100)
    assert not tol.all_near_zero_mod(94.5, 100)

    assert tol.all_near_zero_mod(-4.5, 100)
    assert not tol.all_near_zero_mod(-5.5, 100)

    assert tol.all_near_zero_mod(104.5, 100)
    assert not tol.all_near_zero_mod(105.5, 100)

    assert tol.all_near_zero_mod([-4.5, 0, 1, 4.5, 3, 95.5, 104.5], 100)
    assert not tol.all_near_zero_mod([-4.5, 0, 1, 4.5, 30], 100)

def test_close():
    no_tol = Tolerance()
    assert no_tol.close(1, 1)
    assert not no_tol.close(1, 0.5)
    assert not no_tol.close(1, 1.5)
    assert not no_tol.close(1, 100.5)
    assert no_tol.close(100, 100)
    assert not no_tol.close(100, 99.5)
    assert not no_tol.close(100, 100.5)

    atol5 = Tolerance(atol=5)
    assert atol5.close(1, 1)
    assert atol5.close(1, 0.5)
    assert atol5.close(1, 1.5)
    assert atol5.close(1, 5.5)
    assert atol5.close(1, -3.5)
    assert not atol5.close(1, 6.5)
    assert not atol5.close(1, -4.5)
    assert not atol5.close(1, 100.5)
    assert atol5.close(100, 100)
    assert atol5.close(100, 100.5)
    assert atol5.close(100, 104.5)
    assert not atol5.close(100, 105.5)

    rtol2 = Tolerance(rtol=2)
    assert rtol2.close(100, 100)
    assert rtol2.close(299.5, 100)
    assert rtol2.close(1, 100)
    assert rtol2.close(-99, 100)
    assert not rtol2.close(100, 1)  # Doesn't commute.
    assert not rtol2.close(300.5, 100)
    assert not rtol2.close(-101, 100)

    tol25 = Tolerance(rtol=2, atol=5)
    assert tol25.close(100, 100)
    assert tol25.close(106, 100)
    assert tol25.close(201, 100)
    assert tol25.close(304.5, 100)
    assert not tol25.close(305.5, 100)

def test_near_zero():
    tol = Tolerance(atol=5, rtol=9999999)
    assert tol.near_zero(0)
    assert tol.near_zero(4.5)
    assert not tol.near_zero(5.5)


def test_near_zero_mod():
    tol = Tolerance(atol=5, rtol=9999999)
    assert tol.near_zero_mod(0, 100)
    assert tol.near_zero_mod(4.5, 100)
    assert not tol.near_zero_mod(5.5, 100)

    assert tol.near_zero_mod(100, 100)
    assert tol.near_zero_mod(95.5, 100)
    assert not tol.near_zero_mod(94.5, 100)

    assert tol.near_zero_mod(-4.5, 100)
    assert not tol.near_zero_mod(-5.5, 100)

    assert tol.near_zero_mod(104.5, 100)
    assert not tol.near_zero_mod(105.5, 100)


def test_repr():
    assert (repr(Tolerance(rtol=2, atol=3, equal_nan=True)) ==
            'Tolerance(rtol=2, atol=3, equal_nan=True)')
    assert (str(Tolerance(rtol=5, atol=6, equal_nan=False)) ==
            'Tolerance(rtol=5, atol=6, equal_nan=False)')
