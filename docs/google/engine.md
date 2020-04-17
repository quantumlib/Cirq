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

# Create an Engine object to use.
# Replace YOUR_PROJECT_ID with the id from your cloud project.
engine = cg.Engine(project_id=YOUR_PROJECT_ID, proto_version=cg.ProtoVersion.V2)

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
`get_latest_calibration` method of the `Engine` object.  This will return a
Python dictionary where each key is the metric name.  The value of the
dictionary will be the value of the metric, which can also be a dictionary.

For example, the key may refer to a two-qubit gate error, and the value may
be a dictionary from 2-tuples of `cirq.GridQubits` to an error rate represented
as a float value.

See the [Calibration Metrics](calibration.md) page for more information.
