# Access and Authentication

AQT offers access to several quantum computing devices, called backends,
ranging from real-hardware ion traps with various number of ions to
quantum computing simulators including different noise models.
To get an overview of available devices visit
[www.aqt.eu](https://www.aqt.eu){:.external} and get direct access to the devices via the
[AQT gateway portal](https://gateway-portal.aqt.eu){:.external}.

## Tokens

The AQT API to access backends uses token-based authentication. In order to be
able to submit quantum circuits via quantum programming software development
kits, you need to supply these tokens. Once you have successfully subscribed
to an AQT backend, you can retrieve the token on the
[AQT gateway portal](https://gateway-portal.aqt.eu){:.external}
and use it in your quantum programs or Jupyter notebook tutorials.

## Backend URLs

Accessing the AQT backends is done using a URL for each backend.
E.g. the AQT simulators which are capable of running ideal simulations
(without a noise model) and real simulations (with a noise model) of a
quantum circuit have different URLs. For running a simulation without noise model use:

```python
url = 'https://gateway.aqt.eu/marmot/sim/'
```

whereas for a simulation with noise model use:

```python
url = 'https://gateway.aqt.eu/marmot/sim/noise-model-1'
```

Real-hardware backends have similar URLs which can be retrieved together
with the token on the
[AQT gateway portal](https://gateway-portal.aqt.eu){:.external}.

## Next Steps

At this point, you should now have access to the AQT service.
You can now try out our
[Getting Started Guide](./getting_started.ipynb).
