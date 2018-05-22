#!/usr/bin/env python

# Copyright 2018 Google LLC
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

from dev_tools import shell_tools, all_checks, prepared_env


def main():
    verbose = False
    checks = [
        all_checks.pylint,
        all_checks.typecheck,
        all_checks.pytest,
        all_checks.incremental_coverage,
    ]

    env = prepared_env.PreparedEnv(None, 'HEAD', 'master', '.', None)
    results = []
    for c in checks:
        print()
        print(shell_tools.highlight('Running ' + c.command_line_switch(),
                                    shell_tools.GREEN))
        result = c.context(), c.perform_check(env, verbose=verbose)
        print(shell_tools.highlight('Finished ' + c.command_line_switch(),
                                    shell_tools.GREEN))
        if verbose:
            print(result)
        print()
        results.append(result)

    print()
    print("ALL CHECK RESULTS")
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
