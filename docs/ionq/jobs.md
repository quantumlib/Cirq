# Running IonQ API Jobs

Here we detail how to run jobs on the IonQ API against the IonQ QPU and the
IonQ simulator.

In this section we assume a `cirq.ionq.Service` object has been instantiated and is
called `service` and `cirq` and `cirq.ionq` have been imported:
```python
import cirq
import cirq.ionq as ionq
service = ionq.Service()
```
See [IonQ API Service](service.md) for how to set the service up.

## Running programs

The IonQ API is a service that allows you to send a quantum circuit as a *job*
to a scheduler server.  This means that a user can submit a job to the API, and
then this job is held in a queue before being scheduled to run on the appropriate
hardware (QPU) or simulator.  Once a job is created (but not necessarily yet run)
on the scheduler, the job is assigned an id and then one can query this
job via the API. The job has a status on it which tells what state the job is in
`running`, `completed`, `failed`, etc.  From a users perspective this is abstracted
mostly away in Cirq, depending on whether one wants to make blocking calls (which
wait for the job to complete), or non-blocking calls (where one creates a job and
only later does one query for the results.)

Here we describe these different methods.

### Via Run

The first method for running is to do so via the `run` method on `cirq.ionq.Service`.

```python
qubit = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,            # Square root of NOT.
    cirq.measure(qubit, key='x')   # Measurement store in key 'x'
)

result = service.run(circuit=circuit, repetitions=100, target='qpu')
print(result)
```
Which results in
```
x=0000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111
```
Looking at these results you should notice something strange. What are the odds
that the x measurements were all 0s followed by all 1s?  The reason for this
sorting is that the IonQAPI only returns statistics about the results, i.e. what
count of results were 0 and what count were 1 (or if you are measuring
multiple qubits the counts of the different outcome bit string outcomes).  In
order to make this compatible with Cirq's notition of `cirq.Result`, these
are then converted into raw results with the exactly correct number of
results (in lexical order). In other words, the measurement results are not
in an order corresponding to the temporal order of the measurements.

When calling run, one will need to include the number of `repetitions` or shots
for the given circuit.  In addition, if there is no `default_target` set on the
service, then a `target` needs to be specified.  Currently the supported targets
are `qpu` and `simulator`.

### Via a sampler

Another method to get results from the IonQ API is to use a sampler.  A sampler
is specifically design to be a lightweight interface for obtaining results
in a [pandas](https://pandas.pydata.org/) dataframe and is the interface
used by other classes in Cirq for objects that process data.  Here is a
simple example showing how to get a sampler and use it.

```python
qubit = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,            # Square root of NOT.
    cirq.measure(qubit, key='x')   # Measurement store in key 'x'
)
sampler = service.sampler(target='qpu')
result = sampler.run(program=circuit, repetitions=100)
print(result)
```

### Via create job

The above two methods, using run and the sampler, both block waiting for
results.  This can be problematic when the queueing time for the service
is long.  Instead it is recommended that one use the job api directly.
In this pattern one first create the job with the quantum circuit one
wishes to run, and the service immediately returns an object that has
the id of the job.  This job id can be recorded and at any time in
the future one can query for the results of this job.

```python
qubit = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,            # Square root of NOT.
    cirq.measure(qubit, key='x')   # Measurement store in key 'x'
)
job = service.create_job(circuit=circuit, target='qpu', repetitions=100)
print(job)
```
which shows that the returned object is a `cirq.ionq.Job`:
```
cirq.ionq.Job(job_id=93d111c1-0898-48b8-babe-80d182f8ad66)
```

One difference between this approach and the run and sampler methods
is that the returned job objects results are more directly related to the
return data from the IonQ API.  They are of types `ionq.QPUResult` or
`ionq.SimulatorResult`.  If one wishes to massage these into the
`cirq.Result` format, one can use `to_cirq_result` on both of these.

Another useful feature of working with jobs directly is that one can
directly cancel or delete jobs.  In particular the `ionq.Job` object
returned by `create_job` has `cancel` and `delete` methods.

## Next steps

Learn how to get information about the performance of the hardware

[IonQ calibrations](calibrations.md)
