# Quantum Engine API

Google's Quantum Computing Service provides the Quantum Engine API to execute
circuits on Google's quantum processor or simulator backends and
to access or manage the jobs, programs, reservations and calibrations. As of Cirq is
the only supported client for this API, using the `cirq_google.Engine` class.
For other use cases (e.g. from a different language), contact
[cirq-maintainers@googlegroups.com](mailto:cirq-maintainers@googlegroups.com)
with a short proposal or submit an [RFC](/cirq/dev/rfc_process.md).

Note: the Quantum Engine API is not yet open for public access.

## Authenticating to Google Cloud

Before you begin, you will need to create a Google Cloud project with the API
enabled and billing enabled.  You will then to create credentials in order to
access the API.

You can create application default credentials from the command line using the
gcloud client:

`gcloud auth application-default login`

From a colab, you can execute:

```
from google.colab import auth
auth.authenticate_user(clear_output=False)
```

More information on creating application default credentials can be found on the
[Google cloud](https://cloud.google.com/docs/authentication/production) website.

## Engine class

The `Engine` class is the entry point to communicate with the API.

It can be initialized using your project id (found within your
[Google Cloud Platform Console](https://console.cloud.google.com)).
You can use this instance to run quantum circuits or sweeps (parameterized
variants of a general circuit).

<!---test_substitution
# Add each circuit to the batch.*
class MockEngine:\n  def run_batch(self, *args, **kwargs):\n    pass
--->
<!---test_substitution
results = job.results.*
results = None
--->
<!---test_substitution
print.results.idx.*
print()
--->
<!---test_substitution
engine = cirq_google.Engine(.*)
engine = MockEngine()
--->
<!---test_substitution
cg.Engine(.*)
cirq.Simulator()
--->
<!---test_substitution
sampler = .*
sampler = engine
--->
```python
import cirq
import cirq_google as cg

# A simple sample circuit
qubit = cirq.GridQubit(5, 2)
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,                 # Square root of NOT.
    cirq.measure(qubit, key='result')   # Measurement.
)

# Create an Engine object.
# Replace YOUR_PROJECT_ID with the id from your cloud project.
engine = cg.Engine(project_id=YOUR_PROJECT_ID)

# Create a sampler from the engine
sampler = engine.sampler(processor_id='PROCESSOR_ID', gate_set=cg.SYC_GATESET)

# This will run the circuit and return the results in a 'Result'
results = sampler.run(circuit, repetitions=1000)

# Sampler results can be accessed several ways

# For instance, to see the histogram of results
print(results.histogram(key='result'))

# Or the data itself
print(results.data)
```

## Device Specification

Several public devices have been released and can be found in the `cirq_google`
package.  These are documented further on the [Google Device](devices.md) page.

However, you can also retrieve the device using the `get_device_specification` of an
`Engine` object.  This is a [protocol buffer](https://developers.google.com/protocol-buffers)
message that contains information about the qubits on the device, the
connectivity, and the supported gates.

This proto can be queried directly to get information about the device or can be transformed
into a `cirq.Device` by using `cirq_google.GridDevice.from_proto()` that will
enforce constraints imposed by the hardware.

See the [Device Specification](specification.md) page for more information on
device specifications.


## Calibration Metrics

Metrics from the current status of the device can be retrieved using the\
`get_current_calibration` method of an `EngineProcessor` object.
`EngineProcessor` objects can be retrieved from `Engine` using `get_processor`.
This will return a Python dictionary where each key is the metric name.  The
value of the dictionary will be the value of the metric, which can also be
a dictionary.

For example, the key may refer to a two-qubit gate error, and the value may
be a dictionary from 2-tuples of `cirq.GridQubits` to an error rate represented
as a float value.

See the [Calibration Metrics](calibration.md) page for more information.

## Running circuits in batch

Circuits can be batched together for improved performance.  The engine object
has a method `run_batch()` that functions similar to `run()` but accepts a
list of circuits and parameter sweeps.  Each circuit must have a corresponding
parameter sweep.  If the circuit does not use a sweep, pass in `None`.

There are some restrictions on the circuits that can be batched together:

*   **Same qubits**: All circuits in the same batch must measure the same
set of qubits.
*   **Same repetitions**: All circuits in the same batch must have the same
number of repetitions.

Batching circuits together that do not follow these restrictions may not
cause an error, but your performance will not be significantly improved.

Results can be retrieved in two different forms:

*    `EngineJob.results()` will return a single `List` object,
with all the sweeps of the first circuit in the batch
followed by all the sweeps in the second circuit, and so on.
*     EngineJob.batched_results()` will return a `List` of `List`s.
The first index will refer to the circuit run, and the second index
will refer to the sweep result in that circuit.

If the circuits are not parameterized, there will only be one `Result`
per circuit using either variant.

The following code shows an example of batching together parameterized
circuits, each of which is a sweep.

```python
import sympy
import cirq

q = cirq.GridQubit(5, 2)

# Create a list of example circuits
circuit_list = []
param_list = []

# Create a list of 5 circuits with 10 sweeps each
num_circuits_in_batch = 5
num_sweeps_in_circuit = 10

# Add each circuit to the batch
for circuit_num in range(num_circuits_in_batch):
  # Example circuit
  circuit = cirq.Circuit(
      cirq.YPowGate(exponent=circuit_num / 10.0)(q),
      cirq.XPowGate(exponent=sympy.Symbol('t'))(q),
      cirq.measure(q, key='m', invert_mask=(True,)))
  # add a sweep for each circuit
  param_sweep = cirq.Linspace('t', start=0, stop=1, length=num_sweeps_in_circuit)
  # Add the circuit/sweep pair to the list
  circuit_list.append(circuit)
  param_list.append(param_sweep)

# Create an Engine object.
# Replace YOUR_PROJECT_ID with the id from your cloud project.
engine = cirq_google.Engine(project_id='YOUR_PROJECT_ID')

# Create a sampler from the engine
job = engine.run_batch(circuit_list,
                       processor_ids=['PROCESSOR_ID'],
                       gate_set=cirq_google.FSIM_GATESET,
                       repetitions=1000,
                       params_list=param_list)
results = job.results()

# The results will be flattened into one list
# You will need to iterate through each circuit and each sweep value
idx = 0
for b in range(num_circuits_in_batch):
  for s in range(num_sweeps_in_circuit):
     print(f'Batch #{b}, Sweep #{s}')
     print(results[idx].histogram(key='m'))
     idx+=1

# Alternative way of getting results.
# Results will be nested in Lists
batch_results = job.batched_results()
for batch_idx, batch in enumerate(batch_results):
  for sweep_idx, result in enumerate(batch):
     print(f'Batch #{batch_idx}, Sweep #{sweep_idx}')
     print(result.histogram(key='m'))
```

## Downloading historical results

Results from  previous computations are archived and can be downloaded later
by those in the same cloud project.  You must use the same project id to
access historical results or your request will be denied.

Each time that you run a circuit or sweep, the `Engine` class will generate
a program id and job id for you.  (You can also specify the program and job id
yourself when running the program).  Both the program and job id will need to be
unique within the project.  In order to retrieve previous results,
you will need both this program id as well as the job id.
If these were generated by the `Engine`, they can be retrieved from the
job object when you run a sweep.
Currently, getting the program and job ids can only be done through the
`Engine` interface and not through the sampler interface.
You can then use `get_program` and `get_job` to retrieve the results.
See below for an example:

```python
# Initialize the engine object
engine = cirq_google.Engine(project_id='YOUR_PROJECT_ID')

# Create an example circuit
qubit = cirq.GridQubit(5, 2)
circuit = cirq.Circuit(
    cirq.X(qubit)**sympy.Symbol('t'),
    cirq.measure(qubit, key='result')
)
param_sweep = cirq.Linspace('t', start=0, stop=1, length=10)

# Run the circuit
job = e.run_sweep(program=circuit,
                  params=param_sweep,
                  repetitions=1000,
                  processor_ids=[PROCESSOR_ID],
                  gate_set=GATE_SET)

# Save the program and jo id for later
program_id = job.program_id
job_id = job.job_id

# Retrieve the results
results = job.results()

# ...
# Some time later, the results can be retrieved
# ...

# Recreate the job object
historical_job = engine.get_program(program_id=program_id).get_job(job_id=job_id)

# Retrieve the results
historical_results = historical_job.results()

```

If you did not save the ids, you can still find them from your
job using the [Cloud Console](https://console.cloud.google.com/quantum/jobs) or
by using our list methods.


### Listing jobs

To list the executions of your circuit, i.e. the jobs, you can use `cirq_google.Engine.list_jobs()`.
You can search in all the jobs within your project using filtering criteria on creation time, execution state and labels.

```python
from cirq_google.engine.client.quantum import enums

# Initialize the engine object
engine = cirq_google.Engine(project_id='YOUR_PROJECT_ID')

# List all the jobs on the project since 2020/09/20 that succeeded:
jobs = engine.list_jobs(created_after=datetime.date(2020,9,20),
                        execution_states=[enums.ExecutionStatus.State.SUCCESS])
for j in jobs:
   print(j.job_id, j.status(), j.create_time())
```

### Listing programs

To list the different instances of your circuits uploaded, i.e. the programs, you can use `cirq_google.Engine.list_programs()`.
Similar to jobs, filtering makes it possible to list programs by creation time and labels.
With an existing `cirq_google.EngineProgram` object, you can list any jobs that were run using that program.

```python
from cirq_google.engine.client.quantum import enums

# Initialize the engine object
engine = cirq_google.Engine(project_id='YOUR_PROJECT_ID')

# List all the programs on the project since 2020/09/20 that have
# the "variational" label with any value and the "experiment" label
# with value "vqe001":
programs = engine.list_programs(
                created_after=datetime.date(2020,9,20),
                has_labels={"variational":"*", "experiment":"vqe001"}
           )
for p in programs:
   print(p.program_id, p.create_time())
   # the same filtering parametrization is available as in engine.list_jobs()
   # for example here we list the jobs under the programs that failed
   for j in p.list_jobs(execution_states=[enums.ExecutionStatus.State.FAILURE]):
     print(j.job_id, j.status())
```

