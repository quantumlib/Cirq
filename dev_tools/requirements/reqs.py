# Copyright 2021 The Cirq Developers
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
import sys


def explode(req: str):
    """Returns the exploded dependency list for a requirements file.

    As requirements files can include other requirements files with the -r directive, it can be
    useful to see a flattened version of all the constraints. This method unrolls a requirement file
    and produces a list of strings for each constraint line in the order of inclusion.

    Args:
        req: path to a requirements file.
    Returns:
         list of lines of requirements
    """
    res = []
    d = os.path.dirname(req)
    with open(req) as f:
        for l in f.readlines():
            l = l.rstrip("\n")
            l = l.lstrip(" ")
            if l.startswith("-r"):
                include = l.lstrip(" ").lstrip("-r").lstrip(" ")
                # assuming relative includes always
                res += explode(os.path.join(d, include))
            elif l:
                res += [l]
    return res


if __name__ == '__main__':
    print('\n'.join(explode(sys.argv[1])))
