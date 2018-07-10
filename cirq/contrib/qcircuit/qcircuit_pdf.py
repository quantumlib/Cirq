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

import errno
import os

from pylatex import Document, NoEscape, Package

from cirq.contrib.qcircuit.qcircuit_diagram import (
        circuit_to_latex_using_qcircuit) # coverage: ignore

# coverage: ignore
def circuit_to_pdf_using_qcircuit_via_tex(circuit, filepath, 
                                          pdf_kwargs=None,
                                          qcircuit_kwargs=None,
                                          clean_ext=('dvi', 'ps'),
                                          documentclass='article'):
    """Compiles the QCircuit-based latex diagram of the given circuit.

    Args:
        circuit: The circuit to produce a pdf of.
        pdf_kwargs: The arguments to pass to generate_pdf.
        qcircuit_kwargs: The arguments to pass to 
            circuit_to_latex_using_qcircuit. 
        clean_ext: The file extensions to clean up after compilation. By
            default, latexmk is used with the '-pdfps' flag, which produces
            intermediary dvi and ps files.
        documentclass: The documentclass of the latex file.
    """
    pdf_kwargs = {'compiler': 'latexmk', 'compiler_args': ['-pdfps'],
                  **({} if pdf_kwargs is None else pdf_kwargs)}
    tex = circuit_to_latex_using_qcircuit(circuit, **qcircuit_kwargs)
    doc = Document(documentclass=documentclass, document_options='dvips')
    doc.packages.append(Package('amsmath'))
    doc.packages.append(Package('qcircuit'))
    doc.append(NoEscape(tex))
    doc.generate_pdf(filepath, **pdf_kwargs)
    for ext in clean_ext:
        try:
            os.remove(filepath + '.' + ext)
        except (OSError, IOError) as e:
            if e.errno != errno.ENOENT:
                raise
