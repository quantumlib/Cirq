# Access and Authentication

IonQ's API gives access to IonQ's trapped ion quantum computers as well as a cloud simulator.
IonQ direct access is currently restricted to those with access to an IonQ API key.
As of January 2021, this access is currently restricted to partners. More information
about partnerships can be found at [ionq.com/get-started](https://ionq.com/get-started).

## Authentication

To use Cirq with the IonQ API, one needs an API key.  This is a random looking string.

Given that you have the API key, there are two ways to use these to
get an object in python to access the API. The object that you construct to access
the API are instances of the `cirq_ionq.Service` class. You can directly use the API key in constructing this instances of this class. Here is an example of this pattern:
```python
import cirq_ionq as ionq

service = ionq.Service(api_key='tomyheart')
```

Alternatively, you can use environment variables to set this value. These environment variable for the api key is `IONQ_API_KEY`.  Details of how to set environment variables vary by operating system.  For example in bash, you would do
```bash
export IONQ_API_KEY=tomyheart
```
In the case that you have set these environment variables, you can just perform
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
