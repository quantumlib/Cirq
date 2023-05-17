# Access and Authentication

IonQ's API gives access to IonQ's trapped ion quantum computers as well as a cloud simulator.
IonQ direct access is currently restricted to those with access to an IonQ API key.
As of January 2021, this access is currently restricted to partners. More information
about partnerships can be found at [ionq.com/get-started](https://ionq.com/get-started).

## Authentication

An API key is required to access IonQ via Cirq. You will pass this key to an 
instance of a `cirq_ionq.Service`, which can then be used to interact
with IonQ computers.

Here is an example of this pattern:

```python
import cirq_ionq as ionq

service = ionq.Service(api_key='tomyheart')
```

Alternatively, you can use environment variables to set this value. This 
environment variable for the API key is `IONQ_API_KEY`. Details of how to set 
environment variables vary by operating system. For example, in `bash`:

```bash
export IONQ_API_KEY=tomyheart
```

Once this variable is set, the `ionq.Service()` will look for it automatically
in the environment:

```python
import cirq_ionq as ionq

service = ionq.Service()
```

The advantage of doing things this way is that you do not have to store the API key in
source code, which might accidentally get uploaded to a version control system, and hence
leak the API key.

## Next steps

[Learn how to run jobs against the service](service.md)

[Learn how to build circuits for the API](circuits.md)

[How to use the service API](jobs.md)

[Get information about QPUs from IonQ calibrations](calibrations.md)
