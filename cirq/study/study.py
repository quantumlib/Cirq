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


class Study(object):

    def __init__(self, executor, program, param_resolvers, **executor_kwags):
        self.executor = executor
        self.program = program
        self.param_resolvers = param_resolvers
        self.executor_kwags = executor_kwags

    def run_study(self):
        trial_results = {}
        for param_resolver in self.param_resolvers:
            trial_result = self.executor.run(program=self.program,
                                             param_resolver=param_resolver,
                                             **self.executor_kwags)
            trial_results[trial_result.param_dict] = trial_result.measurements
        return trial_results
