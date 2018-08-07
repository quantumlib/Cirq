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
import os
import pytest

import cirq


def test_temp_file():
    with cirq.testing.TempFilePath() as file_path:
        with open(file_path, 'w') as f_write:
            f_write.write('Hello!\n')

        with open(file_path, 'r') as f_read:
            contents = f_read.read()

    assert contents == 'Hello!\n'

    with pytest.raises(IOError):
        # Can't open the file because it's deleted
        with open(file_path, 'r'):
            pass

    with pytest.raises(IOError):
        # Can't create a file because the directory is gone
        with open(file_path, 'w'):
            pass


def test_temp_dir():
    with cirq.testing.TempDirectoryPath() as dir_path:
        file_path = os.path.join(dir_path, 'file.txt')
        with open(file_path, 'w') as f_write:
            f_write.write('Hello!\n')

        with open(file_path, 'r') as f_read:
            contents = f_read.read()

    assert contents == 'Hello!\n'

    with pytest.raises(IOError):
        # Can't open the file because it's deleted
        with open(file_path, 'r'):
            pass

    with pytest.raises(IOError):
        # Can't create a file because the directory is gone
        with open(file_path, 'w'):
            pass
