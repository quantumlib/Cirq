# Access and Authentication

IonQ's API gives access to IonQ's trapped ion quantum computers as well as a cloud simulator.
IonQ direct access is currently restricted to those with access to an IonQ API key.
As of January 2021, this access is currently restricted to partners. More information
about partnerships can be found at [ionq.com/get-started](https://ionq.com/get-started).

## Authentication

To use Cirq with the IonQ API, two pieces of information are required.  The first is the
url of the API frontend.  This should be specified as something like `https://example.com/`
(the actual url will be provided to partners). The second is the API key, which is just a
random looking string.

Given that you have the remote host and the API key, there are two ways to use these to
get an object in python to access the API. The object that you construct to access
the API are instances of the `cirq.ionq.Service` class. You can directly use the remote host
and API key in constructing this instances of this class. Here is an example of this pattern:
```python
import cirq.ionq as ionq

service = ionq.Service(remote_host='http://example.com/', api_key='tomyheart')
```

Alternatively, you can use environment variables for these values. These environment variables
are `IONQ_REMOTE_HOST` and `IONQ_API_KEY`.  Details of how to set environment variables vary
by operating system.  For example in bash, you would do
```bash
export IONQ_REMOTE_HOST=http://example.com/v1
export IONQ_API_KEY=tomyheart
```
In the case that you have set these environment variables, you can just perform
```python
import cirq.ionq as ionq

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
