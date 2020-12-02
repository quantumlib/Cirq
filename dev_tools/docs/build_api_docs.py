# Copyright 2020 The Cirq Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tool to generate external api_docs for Cirq (Shameless copy from TFQ)."""

import os
import types

import networkx
from absl import app
from absl import flags
from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import cirq
from cirq import _doc

flags.DEFINE_string("output_dir", "/tmp/cirq_api", "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/quantumlib/cirq/tree/master/cirq",
    "The url prefix for links to code.",
)

flags.DEFINE_bool("search_hints", True, "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "quark/cirq/api_docs/python", "Path prefix in the _toc.yaml")

FLAGS = flags.FLAGS


def filter_unwanted_inherited_methods(path, parent, children):
    """Filter the unwanted inherited methods.

    CircuitDag inherits a lot of methods from `networkx.DiGraph` and `Graph`.
    This filter removes these, as it creates a lot of noise in the API docs.
    """
    if parent.__name__ != "CircuitDag":
        return children

    filtered_children = []
    for name, obj in children:
        if isinstance(obj, types.FunctionType):
            if obj.__module__.startswith('cirq'):
                filtered_children.append((name, obj))
    return filtered_children


def main(unused_argv):
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq",
        py_modules=[("cirq", cirq)],
        base_dir=os.path.dirname(cirq.__file__),
        code_url_prefix=FLAGS.code_url_prefix,
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter, filter_unwanted_inherited_methods],
        private_map={
            # Opt to not build docs for these paths for now since they error.
            "cirq.google.engine.client.quantum.QuantumEngineServiceClient": ["enums"],
            "cirq.google.engine.client.quantum_v1alpha1.QuantumEngineServiceClient": ["enums"],
            "cirq.google.api": ["v1"],
        },
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_controls.decorate_all_class_attributes(
        doc_controls.do_not_doc_inheritable, networkx.DiGraph, skip=[]
    )
    doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
