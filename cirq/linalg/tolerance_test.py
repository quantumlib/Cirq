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

from cirq.linalg.tolerance import all_close, all_near_zero, all_near_zero_mod, \
    near_zero, near_zero_mod


def test_all_close():
    assert all_close(1, 1)
    assert not all_close(1, 0.5)
    assert not all_close(1, 1.5)
    assert not all_close(1, 100.5)
    assert all_close(100, 100)
    assert not all_close(100, 99.5)
    assert not all_close(100, 100.5)

    assert all_close([1, 2], [1, 2])
    assert not all_close([1, 2], [1, 3])

    atol = 5
    assert all_close(1, 1, atol=atol)
    assert all_close(1, 0.5, atol=atol)
    assert all_close(1, 1.5, atol=atol)
    assert all_close(1, 5.5, atol=atol)
    assert all_close(1, -3.5, atol=atol)
    assert not all_close(1, 6.5, atol=atol)
    assert not all_close(1, -4.5, atol=atol)
    assert not all_close(1, 100.5, atol=atol)
    assert all_close(100, 100, atol=atol)
    assert all_close(100, 100.5, atol=atol)
    assert all_close(100, 104.5, atol=atol)
    assert not all_close(100, 105.5, atol=atol)

    rtol = 2
    assert all_close(100, 100, rtol=rtol)
    assert all_close(299.5, 100, rtol=rtol)
    assert all_close(1, 100, rtol=rtol)
    assert all_close(-99, 100, rtol=rtol)
    assert not all_close(100, 1, rtol=rtol)  # Doesn't commute.
    assert not all_close(300.5, 100, rtol=rtol)
    assert not all_close(-101, 100, rtol=rtol)

    assert all_close(100, 100, rtol=rtol, atol=atol)
    assert all_close(106, 100, rtol=rtol, atol=atol)
    assert all_close(201, 100, rtol=rtol, atol=atol)
    assert all_close(304.5, 100, rtol=rtol, atol=atol)
    assert not all_close(305.5, 100, rtol=rtol, atol=atol)


def test_all_zero():
    atol = 5
    rtol = 9999999
    assert all_near_zero(0, atol=atol, rtol=rtol)
    assert all_near_zero(4.5, atol=atol, rtol=rtol)
    assert not all_near_zero(5.5, atol=atol, rtol=rtol)

    assert all_near_zero([-4.5, 0, 1, 4.5, 3], atol=atol, rtol=rtol)
    assert not all_near_zero([-4.5, 0, 1, 4.5, 30], atol=atol, rtol=rtol)


def test_all_zero_mod():
    atol = 5
    rtol = 9999999
    assert all_near_zero_mod(0, 100, atol=atol, rtol=rtol)
    assert all_near_zero_mod(4.5, 100, atol=atol, rtol=rtol)
    assert not all_near_zero_mod(5.5, 100, atol=atol, rtol=rtol)

    assert all_near_zero_mod(100, 100, atol=atol, rtol=rtol)
    assert all_near_zero_mod(95.5, 100, atol=atol, rtol=rtol)
    assert not all_near_zero_mod(94.5, 100, atol=atol, rtol=rtol)

    assert all_near_zero_mod(-4.5, 100, atol=atol, rtol=rtol)
    assert not all_near_zero_mod(-5.5, 100, atol=atol, rtol=rtol)

    assert all_near_zero_mod(104.5, 100, atol=atol, rtol=rtol)
    assert not all_near_zero_mod(105.5, 100, atol=atol, rtol=rtol)

    assert all_near_zero_mod([-4.5, 0, 1, 4.5, 3, 95.5, 104.5], 100, atol=atol,
                             rtol=rtol)
    assert not all_near_zero_mod([-4.5, 0, 1, 4.5, 30], 100, atol=atol,
                                 rtol=rtol)


def test_near_zero():
    atol = 5
    assert near_zero(0, atol=atol)
    assert near_zero(4.5, atol=atol)
    assert not near_zero(5.5, atol=atol)


def test_near_zero_mod():
    atol = 5
    assert near_zero_mod(0, 100, atol=atol)
    assert near_zero_mod(4.5, 100, atol=atol)
    assert not near_zero_mod(5.5, 100, atol=atol)

    assert near_zero_mod(100, 100, atol=atol)
    assert near_zero_mod(95.5, 100, atol=atol)
    assert not near_zero_mod(94.5, 100, atol=atol)

    assert near_zero_mod(-4.5, 100, atol=atol)
    assert not near_zero_mod(-5.5, 100, atol=atol)

    assert near_zero_mod(104.5, 100, atol=atol)
    assert not near_zero_mod(105.5, 100, atol=atol)
