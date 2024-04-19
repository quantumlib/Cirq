# Access and Authentication

AQT offers access to several quantum computing devices, called quantum resources,
ranging from real-hardware ion traps with various number of ions to
quantum computing simulators including different noise models.
To get an overview of available resources and information on how to get 
access to them, visit [www.aqt.eu](https://www.aqt.eu/qc-systems/){:.external}.

## Tokens

The AQT API to access quantum resources uses token-based authentication. In order to be
able to submit quantum circuits you need to supply your token. You can request a 
token from AQT and once you have it, use it in your quantum programs 
or Jupyter notebook tutorials.

## Workspaces and Resources

To submit circuits to an AQT backend you need to specify a workspace and resource.
E.g. to send a circuit to one of the hosted AQT simulators, which are capable of 
running ideal simulations (without a noise model) and real simulations (with a 
noise model) of a quantum circuit, you might use the workspace `aqt-simulators` 
and the resource `simulator_noise`.

Which workspaces and resources you have access to, can be retrieved using your access 
token. The resource type helps distinguishing between
- device (real hardware)
- simulator (hosted simulators)
- offline_simulator (offline simulators)

## Offline Simulators

The Cirq simulator with AQT specific noise model can be used to simulate circuits 
even without a token on your machine. 

## REST API

It is also possible to access the documentation of the underlying REST API at 
[AQT Public API](https://arnica.aqt.eu/api/v1/docs){:.external}.

## Next Steps

You can now try out our [Getting Started Guide](./getting_started.ipynb).
