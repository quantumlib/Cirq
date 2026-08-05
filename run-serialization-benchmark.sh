#!/bin/sh

pytest -p no:randomly \
    --benchmark-enable \
    --benchmark-min-rounds=9 \
    --benchmark-autosave \
    "benchmarks/serialization_perf.py::test_json_serialization[100-1000]"
