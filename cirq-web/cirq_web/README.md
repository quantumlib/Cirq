## Cirq Visualizations

*This directory contains the code and instructions for calling Typescript visualization in Cirq using Python. While necessary, this is only half of the code needed to run the `cirq-web` project. For information on how to integrate projects here with Python and the wider Cirq package, see the `cirq_ts` directory.*

The `cirq_web` package runs separately from the rest of Cirq, and can be used on an opt-in basis with the rest of the project. 

### Module build structure

A reference for the build structure of a module is the Bloch sphere. Reference the `bloch_sphere/` directory to see the code. Modules should:
 - Abide by Cirq convention in terms of testing, styling, and initialization files.
 - Contain a "root" folder labeled according to the title of the module. In the case of the Bloch sphere, this is `bloch_sphere/`. 
 - Contain a main class that contains the code for the visualization. All supporting files should be imported into this class. In the case of the Bloch sphere, this is `bloch_sphere.py`.
 - Make sure that any additional modules and files are in separate subdirectories labeled accordingly.

### Developing Python modules for visualization

In order to actually get visualization output from our Python calls, we return strings of HTML and Javascript. In order to keep things organized, we include a parent class `Widget` (`widget.py`) which handles the configuration behind locating and reading files so that only code specific to each visualization lives in its main class. 

The main class for all visualizations should inherit from the `Widget` class located in this directory. Upon creating a new Widget, you should include a call to initialize the parent like so:
```python
class MyWidget(widget.Widget):
    def __init__(self,...):
        ...
        super().__init__()
        ...
```
This ensures that your widget has the standard functionality of all Cirq visualization widgets, including:
 - A unique id for each instance of your visualization.
 - Magic method so that your visualization can be displayed in a Colab/Jupyter notebook.
 - The ability to generate a standalone HTML file with your visualization.

`Widget` is an abstract class with methods `get_client_code()` and `get_widget_bundle_name()` that need to be implemented in your visualization as well. Failure to implement these will lead to a `NotImplementedError` at runtime. Instructions on how to properly implement these methods are in the next section.

### Handling HTML output from Python

In your individual visualizations class, you only need to handle two things:
 1. The client code that's unique to your visualization.
 2. The name of the bundle file.

```python
from cirq_web import widget

class MyWidget(widget.Widget):
    ...
    def get_client_code(self) -> str:
        return f"""
            <script>
            YOUR_CLIENT_CODE
            </script>
        """
    
    def get_widget_bundle_name(self) -> str:
        return 'YOUR_BUNDLE_FILE.bundle.js'
```

`Widget` will take this information and organize it so that it can be properly displayed.

#### Viewing a visualization in a notebook setting
We capitalize on IPython's `_repr_html_` magic method to help display visualizations in the notebook. This will allow your visualization to be displayed in a notebook cell with:
```python
    widget = MyWidget()
    display(widget)
```

#### Generating a standalone HTML file from a visualization
You can generate a standalone HTML file of your visualization like so:
```python
    widget = MyWidget()

    output_directory = './'
    file_name = 'YOUR_VIZ.html'
    open_in_browser = False

    widget.generate_html_file(output_directory, file_name, open_in_browser)
```

This code above writes a file named `YOUR_VIZ.html` to the specified output directory, returning the path of the file as a string. If the `open_in_browser` flag is used, Python will automatically open your visualization in a new tab using the default browser on your computer. 

The important thing about generating standalone HTML files is that they can be sent and viewed anywhere, regardless of whether the recipient has Cirq installed on their computer or not.
