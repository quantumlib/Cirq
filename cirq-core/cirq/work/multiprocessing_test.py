# Copyright 2024 The Cirq Developers
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


import multiprocessing
import concurrent.futures
import pytest

from cirq.work.multiprocessing import execute_with_progress_bar, starmap_with_progress_bar


def _sinle_arg_func(x: int) -> str:
    return f'{x=}'


def _multi_arg_func(x: int, y: int) -> str:
    return f'{x=}-{y=}'


@pytest.mark.parametrize(
    'pool', [None, multiprocessing.Pool(2), concurrent.futures.ThreadPoolExecutor(2)]
)
def test_execute_with_progress_bar(pool):
    desired = set([f'{x=}' for x in range(10)])
    actual = set(execute_with_progress_bar(_sinle_arg_func, range(10), pool=pool))
    assert actual == desired
    if isinstance(pool, multiprocessing.pool.Pool):
        pool.close()
    if isinstance(pool, concurrent.futures.ThreadPoolExecutor):
        pool.shutdown()


@pytest.mark.parametrize(
    'pool', [None, multiprocessing.Pool(2), concurrent.futures.ThreadPoolExecutor(2)]
)
def test_starmap_with_progress_bar(pool):
    desired = set([f'{x=}-{y=}' for x, y in zip(range(10), range(1000, 1000 + 10))])
    actual = set(
        starmap_with_progress_bar(
            _multi_arg_func, zip(range(10), range(1000, 1000 + 10)), pool=pool
        )
    )
    assert actual == desired
    if isinstance(pool, multiprocessing.pool.Pool):
        pool.close()
    if isinstance(pool, concurrent.futures.ThreadPoolExecutor):
        pool.shutdown()


def test_invalid_argument_raises_error():
    with pytest.raises(TypeError):
        _ = execute_with_progress_bar(_sinle_arg_func, range(10), pool=3)

    with pytest.raises(TypeError):
        _ = starmap_with_progress_bar(
            _multi_arg_func, zip(range(10), range(1000, 1000 + 10)), pool=3
        )
