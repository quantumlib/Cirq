# Cirq Performance Benchmarks

This directory contains Cirq performance benchmarks established using the
[pytest-benchmark] plugin for pytest.

## Overview

The benchmarks are defined by the `*_perf.py` files provided in this
`benchmarks` package and its sub-folders.  The benchmark definitions are
very similar to common pytest test functions, but they use an extra features
from the pytest-benchmark plugin to collect code execution times and statistics.
For more information on how to write new benchmarks, please refer to
existing benchmark files and to the [pytest-benchmark] documentation

## Usage

To run all benchmarks, navigate to the root Cirq directory in
a shell and execute the following command:

```bash
pytest -p no:randomly --override-ini="python_files=*_perf.py" \
    --benchmark-enable ./benchmarks
```

This will run the entire benchmark suite which takes approximately
30 minutes.  Note that it is important to pass the `--benchmark-enable`
option as otherwise the code would run as a standard one-shot
pytest and would not collect timing statistics (this may be preferable
for benchmark development).  Some of the benchmarks are labeled with
the `slow` marker and are by default deselected in a standard benchmark
session.  The `slow` marker is applied for larger sizes of parametrized
benchmarks, which are also covered at smaller computational scales, and
are thus not critical for assessing performance trends.
That said, to execute all benchmarks including the `slow` ones, use

```bash
pytest -p no:randomly --override-ini="python_files=*_perf.py" \
    --benchmark-enable --enable-slow-tests ./benchmarks
```

Finally, to run a single specific benchmark and save its results
for later comparison, use the `--benchmark-autosave` option together
with the pytest identifier of the benchmark, for example,

```bash
pytest -p no:randomly --benchmark-enable \
    --benchmark-enable --benchmark-autosave \
    "benchmarks/linalg_decompositions_perf.py::test_kak_decomposition[CNOT]"
```

Please refer to the [pytest-benchmark] documentation for further instructions
on comparing and visualizing benchmark results.

## Results Database

TODO: b/393456969 - provide pointers to the internal results database

[pytest-benchmark]: https://pytest-benchmark.readthedocs.io/en/latest
