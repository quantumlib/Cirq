# TODO
# Access and Authentication

AQT offers access to several quantum computing devices, called backends,
ranging from real-hardware ion traps with various number of ions to
quantum computing simulators including different noise models.
To get an overview of available devices visit
[www.aqt.eu](https://www.aqt.eu){:.external} ~~and get direct access to the devices via the
[AQT gateway portal](https://gateway-portal.aqt.eu){:.external}~~.

## Tokens

The AQT API to access backends uses token-based authentication. In order to be
able to submit quantum circuits via quantum programming software development
kits, you need to supply your token. ~~Once you have successfully subscribed
to an AQT backend, you can retrieve the token on the
[AQT gateway portal](https://gateway-portal.aqt.eu){:.external}
and~~ use it in your quantum programs or Jupyter notebook tutorials.

## Accessing Workspaces and Resources

Accessing the AQT resources is done by providing a workspace and resource ID for each backend.
E.g. the AQT simulators which are capable of running ideal simulations
(without a noise model) and real simulations (with a noise model) of a
quantum circuit have different URLs. For running a simulation without noise model use:

```python
workspace = "aqt_simulators"
resource = "simulator_noise"
```

whereas for a simulation with noise model use:

```python
workspace = "aqt_simulators"
resource = "simulator_no_noise"
```

Real-hardware backends have similar IDs which can be retrieved using your access token on the
[AQT Public API](https://arnica.aqt.eu){:.external}.

## Next Steps

You can now try out our
[Getting Started Guide](./getting_started.ipynb).
