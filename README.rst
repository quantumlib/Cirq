.. ╭─────────────────────── Notice ── Notice ── Notice ────────────────────────╮
.. │ Some valid RST constructs in this file are nevertheless rejected by PyPI. │
.. │ They're delineated by comment lines like "start github-only" and removed  │
.. │ by code in setup.py when creating Python distributions.                   │
.. ╰─────────────────────── Notice ── Notice ── Notice ────────────────────────╯

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-3c60b1.svg?logo=opensourceinitiative&logoColor=white&style=flat-square
   :alt: Licensed under the Apache 2.0 license
   :target: https://github.com/quantumlib/Cirq/blob/main/LICENSE

.. |python| image:: https://img.shields.io/badge/Python-3.10+-fcbc2c.svg?style=flat-square&logo=python&logoColor=white
   :alt: Compatible with Python versions 3.10 and higher
   :target: https://www.python.org/downloads/

.. |contributors| image:: https://img.shields.io/github/contributors/quantumlib/cirq?label=Contributors&logo=github&style=flat-square&color=lightgray
   :alt: GitHub contributors
   :target: https://github.com/quantumlib/Cirq/graphs/contributors

.. |stars| image:: https://img.shields.io/github/stars/quantumlib/cirq?style=flat-square&logo=github&label=Stars&color=lightgray
   :alt: GitHub stars
   :target: https://github.com/quantumlib/cirq

.. |zenodo| image:: https://img.shields.io/badge/10.5281%2Fzenodo.4062499-gray.svg?label=DOI&style=flat-square&colorA=gray&colorB=3c60b1
   :alt: Archived in Zenodo
   :target: https://doi.org/10.5281/zenodo.4062499

.. |ci| image:: https://img.shields.io/github/actions/workflow/status/quantumlib/cirq/ci.yml?event=schedule&style=flat-square&logo=GitHub&label=CI
   :alt: Cirq continuous integration status
   :target: https://github.com/quantumlib/Cirq/actions/workflows/ci-daily.yml

.. |codecov| image:: https://img.shields.io/codecov/c/github/quantumlib/cirq?style=flat-square&logo=codecov&logoColor=white&label=Codecov
   :alt: Code coverage report
   :target: https://codecov.io/gh/quantumlib/Cirq

.. |pypi| image:: https://img.shields.io/pypi/v/cirq.svg?logo=python&logoColor=white&label=PyPI&style=flat-square&color=fcbc2c
   :alt: Cirq project on PyPI
   :target: https://pypi.org/project/cirq

.. |colab| image:: https://img.shields.io/badge/Google%20Colab-gray.svg?style=flat-square&logo=googlecolab&logoColor=fd6c0c
   :alt: Google Colab
   :target: https://colab.google/

.. |qsim| replace:: qsim
.. _qsim: https://github.com/quantumlib/qsim

.. |stim| replace:: Stim
.. _stim: https://github.com/quantumlib/stim

.. |qualtran| replace:: Qualtran
.. _qualtran: https://github.com/quantumlib/qualtran

.. |openfermion| replace:: OpenFermion
.. _openfermion: https://github.com/quantumlib/openfermion

.. |openfermioncirq| replace:: OpenFermion-Cirq
.. _openfermioncirq: https://github.com/quantumlib/openfermion_cirq

.. |recirq| replace:: ReCirq
.. _recirq: https://github.com/quantumlib/recirq

.. |tfq| replace:: TensorFlow Quantum
.. _tfq: https://github.com/tensorflow/quantum


.. ▶︎─── start github-only ───
.. PyPI supports RST's ".. class::", but GitHub does not. PyPI respects
.. ":align: center" for images, but GitHub does not. The only way to center
.. text or images in GitHub is to use raw HTML -- which PyPI rejects (!!!).
.. To satisfy both, this file contains fences around code that is removed
.. when creating distributions for PyPI.

.. raw:: html

   <div align="center">
.. ▶︎─── end github-only ───

.. image:: https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/Cirq_logo_color.svg
   :alt: Cirq logo
   :width: 280px
   :height: 135px
   :align: center

.. class:: centered

Python package for writing, manipulating, and running
`quantum circuits <https://en.wikipedia.org/wiki/Quantum_circuit>`__
on quantum computers and simulators.

.. class:: margin-top
.. class:: centered

| |license| |python| |colab| |pypi| |zenodo|
| |contributors| |stars| |ci| |codecov|

.. class:: centered

.. ▶︎─── start github-only ───
.. Note #1: the following addition of <p> is deliberate; it improves spacing.
.. Note #2: we're not done with the <div> -- it gets closed later, not here.
.. raw:: html

   <p>
.. ▶︎─── end github-only ───

`Features <#features>`__ –
`Installation <#installation>`__ –
`Quick Start <#quick-start-hello-qubit-example>`__ –
`Documentation <#cirq-documentation>`__ –
`Integrations <#integrations>`__ –
`Community <#community>`__ –
`Citing Cirq <#citing-cirq>`__ –
`Contact <#contact>`__

.. ▶︎─── start github-only ───
.. raw:: html

   </p></div>
.. ▶︎─── end github-only ───


Features
--------

.. image:: https://raw.githubusercontent.com/quantumlib/Cirq/refs/heads/main/docs/images/quantum-computer-200px.svg
   :alt: Google's quantum computer
   :align: right
   :width: 200px
   :height: 267px

Cirq provides useful abstractions for dealing with today’s `noisy
intermediate-scale quantum <https://arxiv.org/abs/1801.00862>`__ (NISQ)
computers, where the details of quantum hardware are vital to achieving
state-of-the-art results. Some of its features include:

* Flexible gate definitions and custom gates
* Parametrized circuits with symbolic variables
* Circuit transformation, compilation and optimization
* Hardware device modeling
* Noise modeling
* Multiple built-in quantum circuit simulators
* Integration with |qsim|_ for
  high-performance simulation
* Interoperability with `NumPy <https://numpy.org>`__ and
  `SciPy <https://scipy.org>`__
* Cross-platform compatibility


Installation
------------

Cirq supports Python version 3.10 and later, and can be used on Linux, MacOS,
and Windows, as well as `Google Colab <https://colab.google/>`__. For complete
installation instructions, please refer to the `Install
<https://quantumai.google/cirq/start/install>`__ section of the Cirq
documentation.


Quick Start – “Hello Qubit” Example
-----------------------------------

Here is a simple example to get you up and running with Cirq after you have
installed it. Start a Python interpreter, and then type the following:

.. code-block:: python

  import cirq

  # Pick a qubit.
  qubit = cirq.GridQubit(0, 0)

  # Create a circuit.
  circuit = cirq.Circuit(
      cirq.X(qubit)**0.5,  # Square root of NOT.
      cirq.measure(qubit, key='m')  # Measurement.
  )
  print("Circuit:")
  print(circuit)

  # Simulate the circuit several times.
  simulator = cirq.Simulator()
  result = simulator.run(circuit, repetitions=20)
  print("Results:")
  print(result)

You should see the following output printed by Python:

.. code-block:: text

  Circuit:
  (0, 0): ───X^0.5───M('m')───
  Results:
  m=11000111111011001000

Congratulations! You have run your first quantum simulation in Cirq. You can
continue learning more by exploring the `many Cirq tutorials
<#tutorials>`__ described below.


Cirq Documentation
------------------

Documentation for Cirq is available in a variety of formats.


Tutorials
.........

* `Video tutorials
  <https://www.youtube.com/playlist?list=PLpO2pyKisOjLVt_tDJ2K6ZTapZtHXPLB4>`__
  on YouTube are an engaging way to learn Cirq.

* `Jupyter notebook-based tutorials
  <https://colab.research.google.com/github/quantumlib/Cirq>`__ let you learn
  Cirq from your browser – no installation needed.

* `Text-based tutorials <https://quantumai.google/cirq/start/basics>`__ are
  great when combined with a local installation of Cirq on your computer.


Reference Documentation
.......................

* Docs for the `current stable release
  <https://quantumai.google/reference/python/cirq/all_symbols>`__ correspond
  to what you get with ``pip install cirq``.

* Docs for the `pre-release
  <https://quantumai.google/reference/python/cirq/all_symbols?version=nightly>`__
  correspond to what you get with ``pip install cirq~=1.0.dev``.


Examples
........

.. |eg-dir| replace:: ``examples`` subdirectory of the Cirq GitHub repo
.. _eg-dir: ./examples/

.. |exp-page| replace:: *Experiments* page on the Cirq documentation site
.. _exp-page: https://quantumai.google/cirq/experiments/qcqmc/high_level

* The |eg-dir|_ has many programs illustrating the application of Cirq to
  everything from common textbook algorithms to more advanced methods.

* The |exp-page|_ has yet more examples, from simple to advanced.


Change log
..........

* The `Cirq releases <https://github.com/quantumlib/cirq/releases>`__ page on
  GitHub lists the changes in each release.


Integrations
------------

Google Quantum AI has a suite of open-source software thats let you do more
with Cirq. From high-performance simulators, to novel tools for expressing and
analyzing fault-tolerant quantum algorithms, our software stack lets you
develop quantum programs for a variety of applications.

.. ▶︎─── start github-only ───
.. raw:: html

   <div align="center">
.. ▶︎─── end github-only ───
.. class:: centered

+-------------------------------------------------+----------------------+
| Your interests                                  | Software to explore  |
+=================================================+======================+
| Large circuits and/or a lot of simulations?     | * |qsim|_            |
|                                                 | * |stim|_            |
+-------------------------------------------------+----------------------+
| | Quantum algorithms?                           | * |qualtran|_        |
| | Fault-tolerant quantum computing (FTQC)?      |                      |
+-------------------------------------------------+----------------------+
| Quantum error correction (QEC)?                 | * |stim|_            |
+-------------------------------------------------+----------------------+
| Chemistry and/or material science?              | * |openfermion|_     |
|                                                 | * |openfermioncirq|_ |
+-------------------------------------------------+----------------------+
| Quantum machine learning (QML)?                 | * |tfq|_             |
+-------------------------------------------------+----------------------+
| Real experiments using Cirq?                    | * |recirq|_          |
+-------------------------------------------------+----------------------+

.. ▶︎─── start github-only ───
.. raw:: html

   </div>
.. ▶︎─── end github-only ───


Community
---------

Cirq has benefited from open-source contributions by over 200 people and
counting. We are dedicated to cultivating an open and inclusive community to
build software for quantum computers, and have a `code of conduct
<https://github.com/quantumlib/cirq/blob/main/CODE_OF_CONDUCT.md>`__ for our
community.


Announcements
.............

Stay on top of Cirq developments using the approach that best suits your needs:

* For releases and major announcements: sign up to the low-volume mailing list
  `cirq-announce <https://groups.google.com/forum/#!forum/cirq-announce>`__.

* For releases only:

  * Via GitHub notifications: configure `GitHub repository notifications
    <https://docs.github.com/en/account-and-profile/managing-subscriptions-and-notifications-on-github/setting-up-notifications/configuring-notifications#configuring-your-watch-settings-for-an-individual-repository>`__
    for the Cirq repo.

  * Via Atom/RSS from GitHub: subscribe to the GitHub `Cirq releases Atom feed
    <https://github.com/quantumlib/Cirq/releases.atom>`__.

  * Via RSS from PyPI: subscribe to the PyPI `Cirq releases RSS feed
    <https://pypi.org/rss/project/cirq/releases.xml>`__.

Cirq releases take place approximately every quarter.


Questions and Discussions
.........................

.. |cirq| replace:: ``cirq``
.. _cirq: https://quantumcomputing.stackexchange.com/questions/tagged/cirq

* Do you have questions about using Cirq? Post them to the `Quantum Computing
  Stack Exchange <https://quantumcomputing.stackexchange.com/>`__ and tag them
  with the |cirq|_ tag.

* Would you like to get more involved? *Cirq Cynq* is our biweekly virtual
  meeting of contributors to discuss everything from issues to ongoing
  efforts, as well as to ask questions. Join the `cirq-dev Google Group
  <https://groups.google.com/forum/#!forum/cirq-dev>`__ to get an automatic
  meeting invitation.


Issues and Pull Requests
........................

* Do you have a feature request or want to report a bug? `Open an issue on
  GitHub <https://github.com/quantumlib/Cirq/issues/new/choose>`__ to report it!

* Do you have a code contribution? Read our `contribution guidelines
  <https://github.com/quantumlib/cirq/blob/main/CONTRIBUTING.md>`__, then open
  a `pull request <https://help.github.com/articles/about-pull-requests/>`__!


Citing Cirq
-----------

When publishing articles or otherwise writing about Cirq, please cite the Cirq
version you use – it will help others reproduce your results. We use Zenodo to
preserve releases. The following links let you download the bibliographic
record for the latest stable release of Cirq in various popular formats:

.. |bibtex| image:: https://img.shields.io/badge/Download%20record-eeeeee.svg?style=flat-square&logo=LaTeX&label=BibTeX&labelColor=106f6e
   :alt: Download BibTeX bibliography record for latest Cirq release
   :target: https://zenodo.org/records/8161252/export/bibtex

.. |marcxml| image:: https://img.shields.io/badge/Download%20record-eeeeee.svg?style=flat-square&label=MARCXML&labelColor=2f00cd&logo=data:image/gif;base64,R0lGODlhPwBAAPIAADQBzFEm03JP2/r5/tvS9bWi641x4QAAACH5BAQAAAAALAAAAAA/AEAAAAP/CLrczkGYQgmhxojwuv9gEBjWYJ6oeRkc6L5LUBBpbassrDsyXVsXinBIbO1gIx+qoBEdny+B0lTYQK+vQqpgjAkkmbB4TMYqZCiCACKY3d427jUwzZ3d8Pyvi1SqY3g/QIOEhYNyOnQnBEYGP0xOZk9aKi1oS1aSV46VZ1OIEBFfo6SlpqR8DwKLLVKLazETJXq0fx6KJrC4A6A9tL9pH5QDBp4niJfAyibFDwGsCsMFC65pTKfY2agdw7CcvK1bsJovuNMAuIwK1QO2MSLw8fLz9JEM37DDzbu9sob/AAk1W+CDwJljxpgBWqZMnYJvxZK1+HYOADuGv8ZRUmdu/92iBd+oMCFDsmTJcQC0VHSlsd04iqnIPWDCoE0XCg2kDJT5JCYPnkCDQvEJoQPRM/A+2OsghcC+GaAAkLhQUyCEGUBiNrWA8iGHVSY8tuvyzcizdj500XA6oWu0DGstNeWAL+UJs7PGndWg0K7DW9FMWNKiDqaScTppVNwLds3ZiiHW1hyb8kI7R5w0MiJsiQqNZmAhC8s1WR0FsFw/E0Tb92xasaKZUslJ+YI5RedWjZSMzjNlRX89cKbNiM60zLijOaTkmBkn0FTgaXDAicsXamN1K6AhIi4dh9UtRl/rbRHXBrtUP8wl6gysUSKur4MvyrEGvWCmozdlhJTQ//4AqsIDH6kcJZQjUZHQhVNVGShTGwgZgwhYAznylwxEZKjhhkNMQUwLSjTTDToFdYYRMO5QdFB04jFjxDAnvmEbSNDYRZqNwW0V0I6EnGbWMLaAxcuKQwaoSh0JDZDPCW4BOAKMtiQD3WzOaGOlKRkEokJLVEokW4x6MHhHjSoKB6aMdqAD5Hs1GuVhjCtk8pAfajHpQj145nlkGpYoEZuREEywxWDHOBhKnoiKImgKYpLYJgxT8fiPjKCwE5wOMJ55GVl8YhESQxZAktMnhn5wpTZLrZPpn4AqRQKjTZJjT6pIReAPo2kG1YOkby4SlZOvLrNCqTJFoOUNw5qRAAA7
   :alt: Download MARCXML bibliography record for latest Cirq release
   :target: https://zenodo.org/records/8161252/export/marcxml

.. |csl| image:: https://img.shields.io/badge/Download%20record-eeeeee.svg?style=flat-square&label=CSL&labelColor=2d98e0&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAUCAMAAADbT899AAAACXBIWXMAAAnXAAAJ1wGxbhe3AAABGlBMVEUtmOBZreZhsuhSquVNp+RfsOcsmOD///92vOs3neHx+P2n1PLY7PrS6fmi0vE7nuJ8v+z+//9Tq+VJpuT8/v+Fw+3t9/2r1vPH4/fd7vovmeCf0PH0+v5uuOlesOfV6/nO5/ik0vLy+f41nOFbr+fL5vhVq+Y6nuLI5Pd6vutotelntOg/oOL2+/6YzfDb7vrQ6Pim1PLw+P04neHK5fcumOCUy+/4/P5CouNst+k7n+Lz+f6e0PFQqeX9/v9+wOwzm+Ht9v2w2PPB4fbg8PuPyO/7/f9HpeSt1/NLp+SDwu3p9fy33PV5vetVrOZSq+av2POczvA+oOJ1u+tFpOOMx+7q9fyy2fS+3/bi8fsxmuCWzPBBouNaruaHgOQWAAAAXnRSTlP////////+///////////////////////////////////////////////////////////////////////////////////////////////+/v//////////////////3JzDywAAAQRJREFUeJyNkddWwkAURS9VgyBFDYIFdewNlAQrVcVewQr+/2+Yc3gylwfuy5619pk5mYnIKBMIehOSMBD5r6L0Y5hxKwZMDAnEIRIyCcSSuiAFkc5MTYMz2ls2RFZmgVzeZ72KOYj55AI/ZNG/3QsUIJZkGVgxq2uY9Y1NYIuBbW7d2d0D9qUIlOQAOCzzFAfrolsBjo6DzJ+cngHnvGa1hnW90QRacgFc5q+A9jXf4QbrpuF1a9VbXveucQ88sGNQX3FZ70iW9eaRXU9D/gwDtjUIPGufSUOkDN/vJaoDCYi4vPL93rTv8ORAtw28a28+ID7lC/j+EVUR6mH6FvErOqDmD0UfH1PhwykNAAAAAElFTkSuQmCC
   :alt: Download CSL JSON bibliography record for latest Cirq release
   :target: https://zenodo.org/records/8161252/export/csl

.. ▶︎─── start github-only ───
.. raw:: html

   <div align="center">
   <p>
.. ▶︎─── end github-only ───
.. class:: centered
.. Important: the spaces between items below are no-break spaces (U+00A0).

|bibtex|    |marcxml|    |csl|

.. ▶︎─── start github-only ───
.. raw:: html

   </p>
   </div>
.. ▶︎─── end github-only ───

For formatted citations and records in other formats, as well as records for
all releases of Cirq past and present, visit the `Cirq page on Zenodo
<https://doi.org/10.5281/zenodo.4062499>`__.


Contact
-------

For any questions or concerns not addressed here, please email
quantum-oss-maintainers@google.com.


Disclaimer
----------

Cirq is not an official Google product. Copyright 2019 The Cirq Developers
