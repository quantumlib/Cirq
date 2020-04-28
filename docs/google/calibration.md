# Calibration Metrics

Quantum processors periodically undergo calibrations to maintain the
quality of the programs that can be run on them.
During this calibration metrics about the performance of the quantum computer
are collected.  This calibration data is stored by Quantum Engine and users can
then query for the current or previous state of the calibration.
Calibrations are also available for past jobs.

A calibration object is a dictionary from metric name (see below) to the value
of the metric.  Note that the value of the metric is also usually a dictionary
(for instance, from qubit or qubit pair to a float value).

## Retrieving calibration metrics

Calibration metrics can be retrieved using an engine instance or with a job.

```
import cirq.google as cg

# Create an Engine object to use.
# Replace YOUR_PROJECT_ID with the id from your cloud project.
engine = cg.Engine(project_id=YOUR_PROJECT_ID, proto_version=cg.ProtoVersion.V2)
processor = engine.get_processor(processor_id=PROCESSOR_ID)

# Get the latest calibration metrics.
latest_calibration = processor.get_current_calibration()

# If you know the timestamp of a previous calibration, you can retrieve the
# calibration using the timestamp in epoch seconds.
previous_calibration = processor.get_calibration(CALIBRATION_SECONDS)

# If you would like to find a calibration from a time-frame, use this.
calibration_list = processor.list_calibration(START_SECONDS, END_SECONDS)

# If you know the job-id, you can retrieve the calibration that the job used.
job = engine.get_job("projects/" + PROJECT_ID
                   + "/programs/"+PROGRAM_ID
                   + "/jobs/" + JOB_ID)
job_calibration = cg.EngineJob(cg.JobConfig(), job, engine).get_calibration()

# The calibration can be iterated through using something like the following.
for metric_name in latest_calibration:
  print(metric_name)
  print('------')
  for qubit_or_pair in latest_calibration[metric_name]:
     # Note that although the value is often singular,
     # the metric_value is of the type list and can have multiple values.
     metric_value = latest_calibration[metric_name][qubit_or_pair]
     print(f'{qubit_or_pair} = {metric_value}')
```

Calibration metrics will also soon be available from the
[Google Cloud Platform Console](https://console.cloud.google.com).

## Average, Pauli and Incoherent Error

Several metrics below define average error, Pauli error and incoherent error.
This section explains the difference between each of these metrics.

The average error is equal to one minus fidelity averaged over all possible
input states.

Pauli error defines decoherence of a single qubit in one of the Pauli channels
X, Y, or Z.  If the errors are distributed in the uniform distribution over all
three axes, the probability of applying an erroneous Pauli gate X, Y, or Z will
be the Pauli error divided by three.  The Pauli error and average error are
related by a multiplicative factor dependent on the number of qubits.

See Table 1 on page 11 of the
[Supplementry Information](https://arxiv.org/abs/1910.11333)
document for a description and comparison between average error, Pauli error,
and depolarization error.

The incoherent error is the "unitarity" of the gate or cycle.
This is defined as the decay rate per gate (or cycle) of the
[Purity](https://en.wikipedia.org/wiki/Purity_(quantum_mechanics))
when fit to an exponential curve.  This rate has been scaled to match the
average error per Clifford gate (or per 2-qubit cycle).
For more about purity benchmarking, see Section 6.3 of this
[thesis](https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf).

The purity error can be interpreted as a measure of the incoherent error,
such as those caused by stochastic processes such as relaxation.  The average
error can be interpreted as containing both this incoherent error as well as
the coherent error resulting from improper control or calibration of the device.

Note that, due to statistical fluctuations, it is possible that the purity
error can exceed the average error by small amounts.

## Individual Metrics

Each metric can be referenced by its key in the calibration object, e.g.
```latest_calibration['single_qubit_idle_t1_micros']```.

**Note that the metric names below are subject to change without notice.**

### P_00 readout error
*   Metric key: single_qubit_p00_error
*   Metric key: parallel_p00_error

The p_00 is defined as the probability that the state is measured as |0⟩ after
being prepared in the |0⟩ state.  The p_00 error is defined as one minus this
result.

There are several sources of error in this model.  This error is primarily
composed of error in measurement (readout) of the qubit while in the ground
state.  However, this probability also contains the error than the qubit was not
reset into the |0⟩ ground state properly.  This is often called the SPAM (state
preparation and measurement) error.

The single qubit error is when the readout is measured in isolation (only one
qubit is measured at a time), while the parallel error is taken for all qubits
at the same time.

### P_11 readout error
*   Metric key: single_qubit_p11_error
*   Metric key: parallel_p11_error

The p_11 is defined as the probability that the state is measured as |1⟩ after
being prepared in the |1⟩ state.  The p_11 error is defined as one minus this
result.

This is dominated by the error in measurement (readout) of the qubit, but it
implicitly contains several different types of error.  Also possible is that the
excited state |1⟩ was not prepared correctly or that the state decayed before
measurement.  This error is generally expected to be higher than the P_00 error.

The single qubit error is when the readout is measured in isolation (only one
qubit is measured at a time), while the parallel error is taken for all qubits
at the same time.

### Readout separation error
*   Metric key: single_qubit_readout_separation_error

When measured by the system, the |0⟩ and |1⟩ states manifest as outgoing analog
signals.  These signals must be interpreted as signifying one state or the
other.  Since these analog signals are continuous distributions, there will be
some statistical overlap in the two distributions that would be theoretically
impossible to distinguish.  This is classified as the separation error, and is
calculated by fitting Gaussian distributions to the signals prepared in the
|0⟩ state and |1⟩ state and calculating the overlap between the two
distributions.  Note that this is a component of both the p_00 and p_11 errors
and is included within those metrics.

### Isolated 1 qubit randomized benchmark error: 
*   Metric key: single_qubit_rb_average_error_per_gate
*   Metric key: single_qubit_rb_pauli_error_per_gate
*   Metric key: single_qubit_rb_incoherent_error_per_gate

Single qubit gate error is estimated using randomized benchmarking by taking
sequences of varying length of the 24 gates within the Clifford group
(those that preserve the Pauli group under conjugation) then
applying the inverse of the unitary equivalent to the executed gate sequence.
The result of the total sequence should always be the identity (|0⟩ state).
The final error is measured and compared against this state to produce the
total error.  This error is calculated for one qubit at a time while all
other qubits on the device are idle (isolated).  See the above section for
descriptions of total versus purity error.

More information about randomized benchmarking can be found in section 6.3
(page 120) of this
[thesis](https://web.physics.ucsb.edu/~martinisgroup/theses/Chen2018.pdf).

### T1 
*   Metric key: single_qubit_idle_t1_micros

The T1 of a qubit represents the time constant of the exponential decay of a
qubit in the excited |1⟩ state into the ground |0⟩ state.  This is calculated
by preparing the excited state with a microwave pulse (a.k.a. an X gate),
measured after a variety of decay times.

An exponential curve is then fit to the resulting data to determine the T1 time,
which is reported in microseconds.

### 2-qubit Isolated XEB error
*   Metric key: two_qubit_sqrt_iswap_gate_xeb_cycle_average_error_per_cycle
*   Metric key: two_qubit_sqrt_iswap_gate_xeb_cycle_pauli_error_per_cycle
*   Metric key: two_qubit_sqrt_iswap_gate_xeb_cycle_incoherent_error_per_cycle
*   Metric key: two_qubit_sycamore_gate_xeb_cycle_average_error_per_cycle
*   Metric key: two_qubit_sycamore_gate_xeb_cycle_pauli_error_per_cycle
*   Metric key: two_qubit_sycamore_gate_xeb_cycle_incoherent_error_per_cycle

Two qubit error is primarily characterized by applying cross-entropy
benchmarking (XEB).  This procedure repeatedly performs a "cycle" of a
random one-qubit gate on each qubit followed by the two qubit entangling gate.
The resulting distribution is analyzed and compared to the expected distribution
using cross entropy.  The value reported is the error rate per cycle (both
the 1 qubit gates as well as the 2 qubit gate).

See the above section for descriptions of average, Pauli, and incoherent error.

These errors are isolated, meaning that, during the metric measurement, only the
pair of qubits being considered is active.  All other qubits are idle.

### 2-qubit Parallel XEB error
*   Metric key:
    two_qubit_parallel_sqrt_iswap_gate_xeb_cycle_average_error_per_cycle
*   Metric key:
    two_qubit_parallel_sqrt_iswap_gate_xeb_cycle_pauli_error_per_cycle
*   Metric key:
    two_qubit_parallel_sqrt_iswap_gate_xeb_cycle_incoherent_error_per_cycle
*   Metric key:
    two_qubit_parallel_sycamore_gate_xeb_cycle_average_error_per_cycle
*   Metric key:
    two_qubit_parallel_sycamore_gate_xeb_cycle_pauli_error_per_cycle
*   Metric key:
    two_qubit_parallel_sycamore_gate_xeb_cycle_incoherent_error_per_cycle

These metrics are calculated the same way as the 2-qubit isolated XEB error
metrics.  However, this metric quantifies the error of multiple parallel 2-qubit
cycles at a time.  Four different discrete patterns of 2-qubits are used,
with each pair of qubits in only one pattern.

Since there are many different possible layouts of parallel two-qubit gates
and each layout may have different cross-talk effects, users may want to perform
this experiment on their own if they have a specific layout commonly used in
their experiment.
