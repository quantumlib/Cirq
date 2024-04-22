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

from typing import (
    cast,
    Any,
    Union,
    Sequence,
    Callable,
    TypeVar,
    Iterable,
    Optional,
    ContextManager,
    TYPE_CHECKING,
)

import itertools
import multiprocessing
import concurrent.futures
import tqdm

import networkx as nx

if TYPE_CHECKING:
    import cirq


def _manhattan_distance(a: 'cirq.GridQubit', b: 'cirq.GridQubit') -> int:
    return abs(a.row - b.row) + abs(a.col - b.col)


def grid_qubits_to_graph(qubits: Sequence['cirq.GridQubit']) -> nx.Graph:
    return nx.Graph(
        pair for pair in itertools.combinations(qubits, 2) if _manhattan_distance(*pair) == 1
    )


_OUTPUT_T = TypeVar('_OUTPUT_T')


def execute_with_progress_par(
    func: Callable[..., _OUTPUT_T],
    inputs: Iterable[Any],
    pool: Optional[Union[multiprocessing.Pool, concurrent.futures.ThreadPoolExecutor]] = None,
    progress_bar: Callable[..., ContextManager] = tqdm.tqdm,
    **progres_bar_args,
) -> Sequence[_OUTPUT_T]:
    if pool is None:
        return [func(args) for args in progress_bar(inputs, **progres_bar_args)]
    if isinstance(pool, concurrent.futures.ThreadPoolExecutor):
        futures = [pool.submit(lambda s: (i, func(s)), args) for i, args in enumerate(inputs)]
        results: Sequence[Optional[_OUTPUT_T]] = [None for _ in range(len(futures))]
        with progress_bar(total=len(results), **progres_bar_args) as progress:
            for future in concurrent.futures.as_completed(futures):
                i, res = future.result()
                results[i] = res
            progress.update(1)
        return cast(_OUTPUT_T, results)
    if isinstance(pool, multiprocessing.pool.Pool):
        raise NotImplementedError
        # def modified_func(args):
        #     return args[0], func(args[1])
        # sequential_inputs = [*inputs]
        # results: Sequence[Optional[_OUTPUT_T]] = [None for _ in range(len(sequential_inputs))]
        # with progress_bar(total=len(sequential_inputs), **progres_bar_args) as progress:
        #     for i, res in pool.imap(modified_func, enumerate(sequential_inputs)):
        #         results[i] = res
        #         progres.update(1)
        # return cast(_OUTPUT_T, results)
    raise TypeError(f'type {type(pool)} of {pool=} is not supported')
