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
# ==============================================================================
"""Tool to generate external api_docs for Cirq.

In order to publish to our site, devsite runs two jobs for us: stable and nightly.
The stable one downloads the latest cirq release from pypi and uses that to generate the reference
API docs.
The nightly one downloads the latest cirq pre-release (pip install cirq --pre) and uses that to
generate the "nightly diff".

This script needs to cater for both of these cases.
"""

import os
import types

import networkx
from absl import app
from absl import flags
from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import cirq
import cirq_aqt
import cirq_google
import cirq_ionq
import cirq_pasqal
import cirq_rigetti
import cirq_web

from cirq import _doc

flags.DEFINE_string("output_dir", "docs/api_docs", "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/quantumlib/Cirq/blob/master",
    "The url prefix for links to code.",
)

flags.DEFINE_bool("search_hints", True, "Include metadata search hints in the generated files")

flags.DEFINE_string("site_path", "reference/python", "Path prefix in the _toc.yaml")

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
    generate_cirq()
    generate_cirq_google()
    generate_cirq_aqt()
    generate_cirq_ionq()
    generate_cirq_pasqal()
    generate_cirq_rigetti()
    generate_cirq_web()


def generate_cirq():
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq",
        py_modules=[("cirq", cirq)],
        base_dir=os.path.dirname(cirq.__file__),
        code_url_prefix=FLAGS.code_url_prefix + "/cirq-core/cirq",
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter, filter_unwanted_inherited_methods],
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_controls.decorate_all_class_attributes(
        doc_controls.do_not_doc_inheritable, networkx.DiGraph, skip=[]
    )
    doc_generator.build(output_dir=FLAGS.output_dir)


def generate_cirq_aqt():
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq-aqt",
        py_modules=[("cirq_aqt", cirq_aqt)],
        base_dir=os.path.dirname(cirq_aqt.__file__),
        code_url_prefix=FLAGS.code_url_prefix + "/cirq-aqt/cirq_aqt",
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter, filter_unwanted_inherited_methods],
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_controls.decorate_all_class_attributes(
        doc_controls.do_not_doc_inheritable, networkx.DiGraph, skip=[]
    )

    doc_generator.build(output_dir=FLAGS.output_dir)


def generate_cirq_ionq():
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq_ionq",
        py_modules=[("cirq_ionq", cirq_ionq)],
        base_dir=os.path.dirname(cirq_ionq.__file__),
        code_url_prefix=FLAGS.code_url_prefix + "/cirq-ionq/cirq_ionq",
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter, filter_unwanted_inherited_methods],
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_controls.decorate_all_class_attributes(
        doc_controls.do_not_doc_inheritable, networkx.DiGraph, skip=[]
    )

    doc_generator.build(output_dir=FLAGS.output_dir)


def generate_cirq_pasqal():
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq-pasqal",
        py_modules=[("cirq_pasqal", cirq_pasqal)],
        base_dir=os.path.dirname(cirq_pasqal.__file__),
        code_url_prefix=FLAGS.code_url_prefix + "/cirq-pasqal/cirq_pasqal",
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter, filter_unwanted_inherited_methods],
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_controls.decorate_all_class_attributes(
        doc_controls.do_not_doc_inheritable, networkx.DiGraph, skip=[]
    )

    doc_generator.build(output_dir=FLAGS.output_dir)


def generate_cirq_rigetti():
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq_rigetti",
        py_modules=[("cirq_rigetti", cirq_rigetti)],
        base_dir=os.path.dirname(cirq_rigetti.__file__),
        code_url_prefix=FLAGS.code_url_prefix + "/cirq-rigetti/cirq_rigetti",
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter, filter_unwanted_inherited_methods],
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_controls.decorate_all_class_attributes(
        doc_controls.do_not_doc_inheritable, networkx.DiGraph, skip=[]
    )

    doc_generator.build(output_dir=FLAGS.output_dir)


def generate_cirq_google():
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq-google",
        py_modules=[("cirq_google", cirq_google)],
        base_dir=os.path.dirname(cirq_google.__file__),
        code_url_prefix=FLAGS.code_url_prefix + "/cirq-google/cirq_google",
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter, filter_unwanted_inherited_methods],
        private_map={
            # Opt to not build docs for these paths for now since they error.
            "cirq_google.cloud.quantum.QuantumEngineServiceClient": ["enums"],
            "cirq_google.cloud.quantum_v1alpha1.QuantumEngineServiceClient": ["enums"],
            "cirq_google.api": ["v1"],
        },
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_generator.build(output_dir=FLAGS.output_dir)


def generate_cirq_web():
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq_web",
        py_modules=[("cirq_web", cirq_web)],
        base_dir=os.path.dirname(cirq_web.__file__),
        code_url_prefix=FLAGS.code_url_prefix + "/cirq-web/cirq_web",
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[public_api.local_definitions_filter, filter_unwanted_inherited_methods],
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_controls.decorate_all_class_attributes(
        doc_controls.do_not_doc_inheritable, networkx.DiGraph, skip=[]
    )

    doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
