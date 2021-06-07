## Cirq Visualizations

*This directory contains the code and instructions for calling Typescript visualization in Cirq using Python. While necessary, this is only half of the code needed to run the `cirq-web` project. For information on how to integrate projects here with Python and the wider Cirq package, see the `cirq_ts` directory.*

The `cirq_web` package runs separately from the rest of Cirq, and can be used on an opt-in basis with the rest of the project. 

### Module build structure

A good example for the build structure of a module is the Bloch sphere. Reference the `bloch_sphere/` directory to see the code. In a general sense, modules should:
 - Abide by Cirq convention in terms of testing, styling, and initialization files.
 - Contain a "root" folder labeled according to the title of the module. In the case of the Bloch sphere, this is `bloch_sphere/`. 
 - Contain a main class that contains the code for the visualization. All supporting files should be imported into this class, and the methods  In the case of the Bloch sphere, this is `cirq_bloch_sphere.py`.
 - Make sure that any additional modules and files are in separate subdirectories labeled accordingly.

### Developing modules

 
