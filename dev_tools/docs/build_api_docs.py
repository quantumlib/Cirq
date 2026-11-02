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
The stable one downloads the latest cirq release from PyPI and uses that to generate the reference
API docs.
The nightly one downloads the latest cirq pre-release (pip install --upgrade cirq~=1.0.dev) and
uses that to generate the "nightly diff".

This script needs to cater for both of these cases.
"""

from __future__ import annotations

import ast
import os
import textwrap
import types

import networkx
from absl import app, flags
from tensorflow_docs.api_generator import doc_controls, generate_lib, public_api, signature

import cirq
import cirq_aqt
import cirq_google
import cirq_ionq
import cirq_pasqal
import cirq_web
from cirq import _doc

flags.DEFINE_string("output_dir", "docs/api_docs", "Where to output the docs")

flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/quantumlib/Cirq/blob/main",
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


def filter_type_checking(path, parent, children):
    filtered_children = []
    for name, obj in children:
        if name != 'TYPE_CHECKING':
            filtered_children.append((name, obj))
    return filtered_children


def _cirq_visit_function_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """Visits a `FunctionDef` or `AsyncFunctionDef` AST node to extract type hints and defaults.

    Works around a limitation in `tensorflow_docs.api_generator.signature` (see
    https://github.com/quantumlib/Cirq/issues/4657) where annotations on `*args`, `**kwargs`,
    positional-only (`/`) parameters, and `async def` functions are ignored.
    """
    if node.returns:
        self.return_annotation = signature._source_from_ast(node.returns)

    pos_args = node.args.posonlyargs + node.args.args
    for arg in (*pos_args, *node.args.kwonlyargs, node.args.vararg, node.args.kwarg):
        if arg and arg.annotation:
            self.annotations[arg.arg] = signature._source_from_ast(arg.annotation)

    for arg, default in zip(reversed(pos_args), reversed(node.args.defaults)):
        self.defaults[arg.arg] = self._preprocess_default(default)

    for arg, kw_default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        if kw_default is not None:
            self.defaults[arg.arg] = self._preprocess_default(kw_default)


def _cirq_signature_str(self) -> str:
    """Formats a `TfSignature` into a string, preserving all parameter annotations and defaults.

    Works around limitations in `tensorflow_docs.api_generator.signature.TfSignature.__str__`
    by formatting all parameters (including positional-only, `*args`, and `**kwargs`) via
    `format_kwargs` to preserve defaults and type annotations, then assembling them in
    canonical parameter-kind order.
    """
    formatter = signature.FormatArguments(parser_config=self.parser_config)
    params = list(self.parameters.values())
    pairs = list(zip(params, formatter.format_kwargs(params)))

    pos_only = [f for p, f in pairs if p.kind == p.POSITIONAL_ONLY]
    pos_or_kw = [f for p, f in pairs if p.kind == p.POSITIONAL_OR_KEYWORD]
    varargs = [f'*{f}' for p, f in pairs if p.kind == p.VAR_POSITIONAL]
    kw_only = [f for p, f in pairs if p.kind == p.KEYWORD_ONLY]
    varkw = [f'**{f}' for p, f in pairs if p.kind == p.VAR_KEYWORD]

    all_args_list = [
        *([*pos_only, '/'] if pos_only else []),
        *pos_or_kw,
        *(varargs or (['*'] if kw_only else [])),
        *kw_only,
        *varkw,
    ]

    arguments_signature = ''
    if all_args_list:
        str_signature = ',\n'.join(all_args_list)
        # An expectation of 80 characters is built into the DevSite formatter.
        if len(str_signature) + 4 < 80:
            str_signature = textwrap.fill(str_signature, width=80)
        arguments_signature = '\n' + textwrap.indent(str_signature, prefix='    ') + '\n'

    if (
        self.return_annotation is not signature.EMPTY
        and signature.EMPTY not in self.return_annotation
    ):
        return f'({arguments_signature}) -> {formatter.format_return(self.return_annotation)}'
    return f'({arguments_signature})'


def _patch_tf_docs_signature_handlers() -> None:
    """Patches `tensorflow_docs.api_generator.signature` to preserve `*args`/`**kwargs` types."""
    signature._ArgDefaultAndAnnotationExtractor.visit_FunctionDef = _cirq_visit_function_def
    signature._ArgDefaultAndAnnotationExtractor.visit_AsyncFunctionDef = _cirq_visit_function_def
    signature.TfSignature.__str__ = _cirq_signature_str


def main(unused_argv):
    _patch_tf_docs_signature_handlers()
    generate_cirq()
    generate_cirq_google()
    generate_cirq_aqt()
    generate_cirq_ionq()
    generate_cirq_pasqal()
    generate_cirq_web()


def generate_cirq():
    doc_generator = generate_lib.DocGenerator(
        root_title="Cirq",
        py_modules=[("cirq", cirq)],
        base_dir=os.path.dirname(cirq.__file__),
        code_url_prefix=FLAGS.code_url_prefix + "/cirq-core/cirq",
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[
            public_api.local_definitions_filter,
            filter_unwanted_inherited_methods,
            filter_type_checking,
        ],
        extra_docs=_doc.RECORDED_CONST_DOCS,
        # Submodules that we do not wish to document.
        private_map={
            'cirq': [
                'circuits',
                'devices',
                'experiments',
                'interop',
                'linalg',
                'ops',
                'optimizers',
                'transformers',
                'qis',
                'sim',
                'study',
                'value',
                'protocols',
                'ion',
                'neutral_atoms',
                'vis',
                'work',
                'json_resolver_cache',
            ]
        },
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
        callbacks=[
            public_api.local_definitions_filter,
            filter_unwanted_inherited_methods,
            filter_type_checking,
        ],
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
        callbacks=[
            public_api.local_definitions_filter,
            filter_unwanted_inherited_methods,
            filter_type_checking,
        ],
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
        callbacks=[
            public_api.local_definitions_filter,
            filter_unwanted_inherited_methods,
            filter_type_checking,
        ],
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
        callbacks=[
            public_api.local_definitions_filter,
            filter_unwanted_inherited_methods,
            filter_type_checking,
        ],
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
        callbacks=[
            public_api.local_definitions_filter,
            filter_unwanted_inherited_methods,
            filter_type_checking,
        ],
        extra_docs=_doc.RECORDED_CONST_DOCS,
    )
    doc_controls.decorate_all_class_attributes(
        doc_controls.do_not_doc_inheritable, networkx.DiGraph, skip=[]
    )

    doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
