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

"""Functionality for simulating circuits with google cloud TPUs.

To simulate a circuit on a cloud tensor processing unit you:

a) Need the ability to run TPU computations on google cloud.

    See https://cloud.google.com/tpu/docs/

    Make sure you can do this before moving on to the next step! In particular,
    you should be able to run something like the example at
    https://cloud.google.com/tpu/docs/quickstart#run_example


        tpu_computation = tpu.rewrite(axy_computation, inputs)

        tpu_grpc_url = TPUClusterResolver(
            tpu=[os.environ['TPU_NAME']]).get_master()

        with tf.Session(tpu_grpc_url) as sess:
            sess.run(tpu.initialize_system())
            sess.run(tf.global_variables_initializer())
            output = sess.run(tpu_computation)
            print(output)
            sess.run(tpu.shutdown_system())

        print('Done!')


b) Create a `cirq.Circuit`, and convert it into a tensorflow-usable form via the
methods in this module.


    circuit = ...
    compute, feed_dict =
        cirq.contrib.tpu.circuit_to_tensorflow_runnable(
            circuit)

c) Extend the computation to produce the result you want, instead of the entire
gigantic state vector. For example, maybe you only want an expectation value or
a subset of the state vector.

    def extended_compute():
        base = compute()
        return base[:128]  # first 128 values of state vector


    tpu_computation = tpu.rewrite(extended_compute, feed_dict=feed_dict)
    ...

d) If you run into serious trouble getting this setup to work, open an issue

    https://github.com/quantumlib/Cirq/issues

or ask on the quantum computing stack exchange

    https://quantumcomputing.stackexchange.com

and we'll do our best to help and to improve this documentation.
"""

from cirq.contrib.tpu.circuit_to_tensorflow import (
    circuit_to_tensorflow_runnable,
)
