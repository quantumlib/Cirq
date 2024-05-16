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

from typing import Any, Union, List, Callable, TypeVar, Iterable, Optional, ContextManager

import multiprocessing
import concurrent.futures
import tqdm


_OUTPUT_T = TypeVar('_OUTPUT_T')


def execute_with_progress_bar(
    func: Callable[..., _OUTPUT_T],
    inputs: Iterable[Any],
    pool: Optional[
        Union['multiprocessing.pool.Pool', concurrent.futures.ThreadPoolExecutor]
    ] = None,
    progress_bar: Callable[..., ContextManager] = tqdm.tqdm,
    **progress_bar_args,
) -> List[_OUTPUT_T]:
    """Execute a function in parallel with progress bar.

    Args:
        func: The callable to execute, the function takes a single argument.
        inputs: An iterable of the argument to pass to the function.
        pool: An optional multiprocessing or threading pool.
        progress_bar: A callable that creates a progress bar.
        progress_bar_args: Arguments to pass to the progress bar.

    Returns:
        An out-of-order list of the results of the function calls.

    Raises:
        TypeError: If the pool is not a multiprocessing pool or threading pool or None.
    """
    sequential_inputs = [*inputs]
    results: List[_OUTPUT_T] = []
    if pool is None:
        with progress_bar(total=len(sequential_inputs), **progress_bar_args) as progress:
            for args in sequential_inputs:
                results.append(func(args))
                progress.update(1)
        return results
    if isinstance(pool, concurrent.futures.ThreadPoolExecutor):
        futures = [pool.submit(func, args) for args in sequential_inputs]
        with progress_bar(total=len(futures), **progress_bar_args) as progress:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(1)
        return results
    with progress_bar(total=len(sequential_inputs), **progress_bar_args) as progress:
        for res in pool.imap_unordered(func, sequential_inputs):
            results.append(res)
            progress.update(1)
    return results


def starmap_with_progress_bar(
    func: Callable[..., _OUTPUT_T],
    inputs: Iterable[Iterable[Any]],
    pool: Optional[
        Union['multiprocessing.pool.Pool', concurrent.futures.ThreadPoolExecutor]
    ] = None,
    progress_bar: Callable[..., ContextManager] = tqdm.tqdm,
    **progress_bar_args,
) -> List[_OUTPUT_T]:
    """Execute a function in parallel with progress bar.

    Args:
        func: The callable to execute, the function takes one or more arguments.
        inputs: An iterable of the arguments to pass to the function.
        pool: An optional multiprocessing or threading pool.
        progress_bar: A callable that creates a progress bar.
        progress_bar_args: Arguments to pass to the progress bar.

    Returns:
        An out-of-order list of the results of the function calls.
    """
    sequential_inputs = [*inputs]
    results: List[_OUTPUT_T] = []
    if pool is None:
        with progress_bar(total=len(sequential_inputs), **progress_bar_args) as progress:
            for args in sequential_inputs:
                results.append(func(*args))
                progress.update(1)
        return results
    if isinstance(pool, concurrent.futures.ThreadPoolExecutor):
        futures = [pool.submit(func, *args) for args in sequential_inputs]
        with progress_bar(total=len(futures), **progress_bar_args) as progress:
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress.update(1)
        return results
    with progress_bar(total=len(sequential_inputs), **progress_bar_args) as progress:
        for res in pool.starmap(func, sequential_inputs):
            results.append(res)
            progress.update(1)
    return results
