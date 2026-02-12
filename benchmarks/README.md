# Cirq Performance Benchmarks

This directory concerns doing Cirq performance benchmarks with [Airspeed
Velocity](https://asv.readthedocs.io/en/stable/index.html).

## Overview

The benchmark files (`bench_*.py`) stored in the current package
(`benchmarks/*`) are used by ASV to run benchmark tests for Cirq. For more
information on how to write new benchmarks, please refer [Writing benchmarks
guide by ASV](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)

## Usage

To run all benchmarks, navigate to the root Cirq directory at the command line
and execute the following command:

```bash
./check/asv_run
```

You can also pass arguments to the `asv_run` script, which will be forwarded to
the `asv run` command. Here's an example:

```bash
./check/asv_run --quick --bench bench_examples --python 3.11
```

Please refer to [Running Benchmarks guide by
ASV](https://asv.readthedocs.io/en/stable/using.html#running-benchmarks) for
more information.

## Results Database

TODO([#3838](https://github.com/quantumlib/Cirq/issues/3838)): Add details
regarding GCP setup.
