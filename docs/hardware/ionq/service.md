# IonQ API Service

IonQ's API provides a way to execute quantum circuits on [IonQ's](https://ionq.com)
trapped ion quantum computers and cloud based simulators. An IonQ account is
required to use this service. To learn more, or sign up, see
[ionq.com/get-started](https://ionq.com/get-started).

## Service class

The main entrance for accessing IonQ's API are instances of the `cirq_ionq.Service` class.
These objects need to be initialized with an API key, see
[Access and Authentication](access.md) for details on obtaining one.

The basic steps for running a quantum circuit in a blocking manner are:

1. Create a circuit to run.
2. Create a `cirq_ionq.Service` with proper authentication and endpoints.
3. Submit this circuit to run on the service and await the results of this call.
(Or alternatively use asynchronous jobs and processing)
4. Transform the results in a form that is most useful for your analysis.

Here is a simple example of this flow:

```python
import cirq
import cirq_ionq as ionq

# A circuit that applies a square root of NOT and then a measurement.
qubit = cirq.LineQubit(0)
circuit = cirq.Circuit(
  cirq.X(qubit)**0.5,      # Square root of NOT.
  cirq.measure(qubit, key='x')  # Measurement store in key 'x'
)

# Create a ionq.Service object.
# Replace API_KEY with your API key.
# Alternatively, if you have the IONQ_API_KEY environment
# variable set, you can omit specifying this api_key parameters.
service = ionq.Service(api_key=API_KEY)

# Run a program against the service. This method will block execution until
# the result is returned (determined by periodically polling the IonQ API).
result = service.run(circuit=circuit, repetitions=100, target='qpu')

# The return object of run is a cirq.Result object.
# From this object, you can get a histogram of results.
histogram = result.histogram(key='x')
print(f'Histogram: {histogram}')

# You can also get the data as a pandas frame.
print(f'Data:\n{result.data}')
```

This produces the following output: (will vary due to quantum randomness!)

```
Histogram: Counter({0: 53, 1: 47})
Data:
  x
0  0
1  0
2  0
3  0
4  0
.. ..
95 1
96 1
97 1
98 1
99 1

[100 rows x 1 columns]
```

## Service parameters

In addition to the `api_key`, there are some other parameters which are
useful for configuring the service. These are passed as arguments
when creating a `cirq_ionq.Service` object.

* `remote_host`: The location of the API in the form of a URL. If this is None,
then this instance will use the environment variable `IONQ_REMOTE_HOST`. If that
variable is not set, then this uses `https://api.ionq.co/{api_version}`.
* `default_target`: this is a string of either `simulator` or `qpu`. By setting this, you do not have to specify a target every time you run a job using `run`, `create_job` or via the `sampler` interface. A helpful pattern is to create two services with defaults for the simulator and for the QPU separately.
* `api_version`: Version of the API to be used. Defaults to 'v0.3'.
* `max_retry_seconds`: The API will poll with exponential backoff for completed jobs. By specifying this, you can change the number of seconds before this retry gives up. It is common to set this to a very small number when, for example, wanting to fail fast, or to be set very high for long running jobs.

## Run parameters

When running a job, there are several parameters that can be provided:

* `circuit`: The `cirq.Circuit` to run.
* `repetitions`: The number of times to run the circuit.
* `name`: An name for the created job (optional.)
* `target`: Where to run the job. Can be 'qpu' or 'simulator'.
* `param_resolver`: A `cirq.ParamResolver` to resolve parameters in `circuit`.
* `seed`: For `simulation` jobs, specify the seed for simulating results. If None, this will be `np.random`, if an int, will be `np.random.RandomState(int)`, otherwise must be a modulate similar to `np.random`.
* `error_mitigation`: A dictionary of error mitigation settings. Valid keys include:
  - 'debias': Set to true to turn on [debiasing](https://ionq.com/resources/debiasing-and-sharpening), which can improve circuit performance by removing qubit-specific noise. _Debiasing is on by default for Aria-class systems._
* `sharpen`: Get sharpened results from debiasing. Off by default. Will generally return more accurate results if your expected output distribution has peaks.
* `extra_query_params`: A dictionary that specifies additional parameters to be provided in the request. (Currently unused)

Here is an example of using error mitigation and sharpening options:

```python
# Run a program against the service with error mitigation and sharpening
result = service.run(
  circuit=circuit,
  repetitions=100,
  target='qpu',
  error_mitigation={'debias': True},
  sharpen=True
)
```

The run method will return a `cirq.Result` object from which you can get a histogram of results. Refer to the first example in this doc for how to process the results.

## Next steps

[Learn how to build circuits for the API](circuits.md)

[How to use the service API](jobs.md)

[Get information about QPUs from IonQ calibrations](calibrations.md)
