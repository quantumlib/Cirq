# Quantum Engine

The Quantum Engine, via the `cirq.google.Engine` class, executes programs and jobs using the
Quantum Engine API.

Note that the Quantum Engine API is not yet open for public access.

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
engine = cirq.google.Engine(.*)
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
import cirq.google as cg

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

# This will run the circuit and return the results in a 'TrialResult'
results = sampler.run(circuit, repetitions=1000)

# Sampler results can be accessed several ways

# For instance, to see the histogram of results
print(results.histogram(key='result'))

# Or the data itself
print(results.data)
```

## Device Specification

Several public devices have been released and can be found in the `cirq.google`
package.  These are documented further on the [Google Device](devices.md) page. 

However, you can also retrieve the device using the `get_device_specification` of an
`Engine` object.  This is a [protocol buffer](https://developers.google.com/protocol-buffers)
message that contains information about the qubits on the device, the
connectivity, and the supported gates.

This proto can be queried directly to get information about the device or can be transformed
into a `cirq.Device` by using `cirq.google.SerializableDevice.from_proto()` that will
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
engine = cirq.google.Engine(project_id='YOUR_PROJECT_ID')

# Create a sampler from the engine
job = engine.run_batch(circuit_list,
                       processor_ids=['PROCESSOR_ID'],
                       gate_set=cirq.google.FSIM_GATESET,
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
```

