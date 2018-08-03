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

from typing import Any

import os, shutil, tempfile


class TempFilePath:
    """A context manager that provides a temporary file path for use within a
    'with' statement.
    """
    def __enter__(self) -> str:
        self.dir_path = tempfile.mkdtemp(prefix='test-output-')
        file_path = os.path.join(self.dir_path, 'test-file')
        return file_path

    def __exit__(self, err_type: Any, err_args: Any, traceback: Any) -> None:
        shutil.rmtree(self.dir_path)


class TempDirectoryPath:
    """A context manager that provides a temporary directory for use within a
    'with' statement.
    """
    def __enter__(self) -> str:
        self.dir_path = tempfile.mkdtemp(prefix='test-output-')
        return self.dir_path

    def __exit__(self, err_type: Any, err_args: Any, traceback: Any) -> None:
        shutil.rmtree(self.dir_path)
