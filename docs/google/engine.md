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

```
import cirq

# A simple sample circuit
qubit = cirq.GridQubit(5, 2)
circuit = cirq.Circuit(
    cirq.X(qubit)**0.5,                 # Square root of NOT.
    cirq.measure(qubit, key='result')   # Measurement.
)

# Create an Engine object to use.
# Replace the project id with the id from your cloud project.
engine = cirq.google.Engine(project_id='your_program_id',
                            proto_version=cirq.google.ProtoVersion.V2)

# Create a unique name for the program.
name = 'example-%s' % ''.join(random.choice(
    string.ascii_uppercase + string.digits) for _ in range(10))

# Upload the program and submit jobs to run in one call.
# Replace PROCESSOR_ID with the processor that you are allowed to run on.
job = engine.run_sweep(
    program=circuit,
    program_id=name,
    repetitions=10000,
    processor_ids=['PROCESSOR_ID'],
    gate_set=cirq.google.SYC_GATESET)

# At this time, the job will be scheduled and pending execution.

# Print out the results. This blocks until the results are returned.
results = [str(int(b)) for b in job.results()[0].measurements['result'][:, 0]]
print("Measurement results:\n")
print(''.join(results))
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


## Calibration Metrics

Metrics from the current status of the device can be retrieved using the\
`get_latest_calibration` method of the `Engine` object.  This will return a
Python dictionary where each key is the metric name.  The value of the
dictionary will be the value of the metric, which can also be a dictionary.

For example, the key may refer to a two-qubit gate error, and the value may
be a dictionary from 2-tuples of `cirq.GridQubits` to an error rate represented
as a float value.

Information about specific metrics will be released at a later date.
