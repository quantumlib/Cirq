# Calibration metrics

Quantum processors periodically undergo calibrations to maintain the
quality of the programs that can be run on them.
During this calibration, metrics about the performance of the device
are collected.  This calibration data is stored by Quantum Engine and users can
then query for the current or previous state of the calibration.
Calibrations are also available for past jobs.

A calibration object is a dictionary from metric name (see below) to the value
of the metric.  Note that the value of the metric is also usually a dictionary
(for instance, from qubit or qubit pair to a float value).

## Retrieving calibration metrics

Calibration metrics can be retrieved through Quantum Engine's Python API.

<!---test_substitution
engine = cg.get_engine\(.*
\g<0>
engine = mock.create_autospec(cirq_google.Engine, instance=True)
mock_engine_processor = mock.create_autospec(cirq_google.EngineProcessor, instance=True)
engine.configure_mock(**{"get_processor.return_value": mock_engine_processor})
--->
<!---test_substitution
PROJECT_ID|PROGRAM_ID|PROCESSOR_ID|JOB_ID|SNAPSHOT_ID
'placeholder'
--->
```python
import cirq_google as cg

# Create an Engine object to use.
engine = cg.get_engine(PROJECT_ID) # Replace with your your Cloud project id.
processor = engine.get_processor(processor_id=PROCESSOR_ID) # replace this
config = processor.get_config() # or get a non-default config

# Get the latest calibration metrics.
calibration = config.calibration

# You can see which metrics are available by looking at the keys:
print(list(calibration.keys()))
```

We typically report Pauli errors, although note that the
[spec 
sheet](https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.p
df)
reports average errors. Please see Table 1 on page 11 of the
[Supplementary Information](https://arxiv.org/abs/1910.11333)
document for a description and comparison between average error, Pauli error,
and depolarization error.

## Individual Metrics

Each metric can be referenced by its key in the calibration object, e.g.
```calibration['sq_rb_pauli_error']```. Keys with names ending in `_ERR`
indicate the statistical uncertainty of the corresponding metric.

**Note that the metric names below are subject to change without notice.**

### Parallel readout error
*   Metric key: zero_error
*   Metric key: one_error

`zero_error` is the probability that a qubit prepared in |0⟩ is measured in
|1⟩, and similarly for `one_error`. These readout error rates are benchmarked
by preparing the qubits simultaneously in random bitstring states, measuring,
and checking what fraction of the time the wrong outcome was measured,
conditioned on the prepared initial state. This is also implemented in
[`cirq.estimate_parallel_single_qubit_readout_errors`](https://quantumai.google/
reference/python/cirq/estimate_parallel_single_qubit_readout_errors).

The plotting functions `calibration.plot()` and `calibration.plot_histograms`
can take `readout` as an input, in which case they will plot the qubit-wise
average of `zero_error` and `one_error`.


### Single qubit randomized benchmark error:
*   Metric key: sq_rb_pauli_error

Single qubit gate error is estimated using randomized benchmarking by taking
sequences of varying length of the 24 gates within the single-qubit Clifford 
group
(those that preserve the Pauli group under conjugation) then
applying the inverse of the unitary equivalent to the executed gate sequence.
The result of the total sequence should always be the identity (|0⟩ state).
The final error is measured and compared against this state to produce the
total error. The decay constant as the length of the sequence is varied gives
the error rate. These single-qubit errors are measured in parallel across the
qubits.

More information about randomized benchmarking can be found in section 6.3
(page 120) of this
[thesis](https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf).

### Two-qubit XEB error
*   Metric key: cz_inferred_gate_error_pauli

Two qubit error is primarily characterized by applying cross-entropy 
benchmarking (XEB).  This procedure repeatedly performs a "cycle" of a random 
one-qubit gate on each qubit followed by the two qubit entangling gate. The 
resulting distribution is analyzed and compared to the expected distribution 
using cross entropy.  See 
[this](https://quantumai.google/cirq/noise/qcvv/xeb_theory) page for more 
details. The decay constant as the legnth of the sequence is increased gives 
the cycle error rate, which includes contributions from both single- and 
two-qubit gates. The contribution from single-qubit gates (characterized using 
randomized benchmarking) is subtracted off to give the inferred two-qubit error 
rate.

When we measure XEB, we typically reoptimize the single-qubit phases (`zeta`, 
`chi`, and `gamma` in a 
[`cirq.PhasedFSimGate`](https://quantumai.google/reference/python/cirq/PhasedFSimGate))
in the simulated version of the gate in order to maximize the XEB 
fidelity with the experimental data. This procedure reoptimizes these 
single-qubit phases, which are then cancelled using z rotations (which are done 
virtually for CZ and cphase gates). Therefore, XEB is actually a calibration in 
addition to a benchmark. These phases drift, and running XEB periodically is 
important to correct for that.

Since there are many different possible layouts of parallel two-qubit gates and 
each layout may have different cross-talk effects, users may want to perform 
this experiment on their own if they have a specific layout commonly used in 
their experiment.

## Plotting
Several tools exist for plotting error metrics and comparing against those 
reported in the
[spec sheet](https://quantumai.google/static/site-assets/downloads/willow-spec-sheet.pdf)
(after converting them to Pauli errors for consistency). For example, one 
can plot an individual metric:
```python
calibration.plot('sq_rb_pauli_error')
```

or one can plot several metrics together in a histogram:
```python
calibration.plot_histograms(['sq_rb_pauli_error', 'cz_inferred_gate_error_pauli', 'readout'])
```

## Historical calibration metrics
Historical metrics can be retrieved by loading the appropriate config. For 
example, to find the metrics corresponding to a job that you ran, you can do:
```python
job = engine.get_program(PROGRAM_ID).get_job(JOB_ID)
config = job.get_config()
calibration = config.calibration
```
Every `config` also has a snapshot ID, `config.snapshot_id`, which shows when 
the calibration was performed (in UTC) and uniquely specifies that calibration. 
A `config` can be loaded from a snapshot ID using
```python
processor = engine.get_processor(PROCESSOR_ID)
config = processor.get_config(device_config_revision = cg.Snapshot(SNAPSHOT_ID))
```
A given snapshot ID may have multiple configs (multiple choices of calibration 
parameters, possibly corresponding to different gatesets. One is set as the 
default, which is what is retrieved above. To list them, you can do
```python
processor.list_configs()
```
and you can load non-default configs by specifying the config name in 
`processor.get_config(...)`.