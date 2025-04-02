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

from __future__ import annotations

import multiprocessing
import os
import pathlib
import pickle
from collections.abc import Iterator
from typing import Any, Hashable

import pytest

import cirq
from cirq.protocols.json_serialization_test import MODULE_TEST_SPECS

_EXCLUDE_JSON_FILES = (
    # sympy - related objects
    "cirq/protocols/json_test_data/sympy.Add.json",
    "cirq/protocols/json_test_data/sympy.E.json",
    "cirq/protocols/json_test_data/sympy.Equality.json",
    "cirq/protocols/json_test_data/sympy.EulerGamma.json",
    "cirq/protocols/json_test_data/sympy.Float.json",
    "cirq/protocols/json_test_data/sympy.GreaterThan.json",
    "cirq/protocols/json_test_data/sympy.Integer.json",
    "cirq/protocols/json_test_data/sympy.LessThan.json",
    "cirq/protocols/json_test_data/sympy.Mul.json",
    "cirq/protocols/json_test_data/sympy.Pow.json",
    "cirq/protocols/json_test_data/sympy.Rational.json",
    "cirq/protocols/json_test_data/sympy.StrictGreaterThan.json",
    "cirq/protocols/json_test_data/sympy.StrictLessThan.json",
    "cirq/protocols/json_test_data/sympy.Symbol.json",
    "cirq/protocols/json_test_data/sympy.Unequality.json",
    "cirq/protocols/json_test_data/sympy.And.json",
    "cirq/protocols/json_test_data/sympy.Not.json",
    "cirq/protocols/json_test_data/sympy.Or.json",
    "cirq/protocols/json_test_data/sympy.Xor.json",
    "cirq/protocols/json_test_data/sympy.Indexed.json",
    "cirq/protocols/json_test_data/sympy.IndexedBase.json",
    "cirq/protocols/json_test_data/sympy.pi.json",
    # Cirq-Rigetti is deprecated per #7058
    # Instead of handling deprecation-in-test errors we exclude
    # all cirq_rigetti classes here.
    "cirq_rigetti/json_test_data/AspenQubit.json",
    "cirq_rigetti/json_test_data/OctagonalQubit.json",
    # RigettiQCSAspenDevice does not pickle
    "cirq_rigetti/json_test_data/RigettiQCSAspenDevice.json",
)


def _is_included(json_filename: str) -> bool:
    json_posix_path = pathlib.PurePath(json_filename).as_posix()
    if any(json_posix_path.endswith(t) for t in _EXCLUDE_JSON_FILES):
        return False
    if not os.path.isfile(json_filename):
        return False
    return True


@pytest.fixture(scope='module')
def pool() -> Iterator[multiprocessing.pool.Pool]:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        yield pool


def _read_json(json_filename: str) -> Any:
    obj = cirq.read_json(json_filename)
    obj = obj[0] if isinstance(obj, list) else obj
    # trigger possible caching of the hash value
    if isinstance(obj, Hashable):
        _ = hash(obj)
    return obj


def test_exclude_json_files_has_valid_entries() -> None:
    """Verify _EXCLUDE_JSON_FILES has valid entries."""
    # do not check rigetti files if not installed
    skip_rigetti = all(m.name != "cirq_rigetti" for m in MODULE_TEST_SPECS)
    json_file_validates = lambda f: any(
        m.test_data_path.joinpath(os.path.basename(f)).is_file() for m in MODULE_TEST_SPECS
    ) or (skip_rigetti and f.startswith("cirq_rigetti/"))
    invalid_json_paths = [f for f in _EXCLUDE_JSON_FILES if not json_file_validates(f)]
    assert invalid_json_paths == []


@pytest.mark.parametrize(
    'json_filename',
    [
        f"{abs_path}.json"
        for m in MODULE_TEST_SPECS
        for abs_path in m.all_test_data_keys()
        if _is_included(f"{abs_path}.json")
    ],
)
def test_hash_from_pickle(json_filename: str, pool: multiprocessing.pool.Pool):
    obj_local = _read_json(json_filename)
    if not isinstance(obj_local, Hashable):
        return
    # check if pickling works in the main process for the sake of debugging
    obj_copy = pickle.loads(pickle.dumps(obj_local))
    assert obj_copy == obj_local
    assert hash(obj_copy) == hash(obj_local)
    # Read and hash the object in a separate worker process and then
    # send it back which requires pickling and unpickling.
    obj_worker = pool.apply(_read_json, [json_filename])
    assert obj_worker == obj_local
    assert hash(obj_worker) == hash(obj_local)
