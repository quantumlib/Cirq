## Cirq Visualizations

*This directory contains the code and instructions for calling Typescript visualization in Cirq using Python. While necessary, this is only half of the code needed to run the `cirq-web` project. For information on how to integrate projects here with Python and the wider Cirq package, see the `cirq_ts` directory.*

The `cirq_web` package runs separately from the rest of Cirq, and can be used on an opt-in basis with the rest of the project. 

### Module build structure

A reference for the build structure of a module is the Bloch sphere. Reference the `bloch_sphere/` directory to see the code. Modules should:
 - Abide by Cirq convention in terms of testing, styling, and initialization files.
 - Contain a "root" folder labeled according to the title of the module. In the case of the Bloch sphere, this is `bloch_sphere/`. 
 - Contain a main class that contains the code for the visualization. All supporting files should be imported into this class. In the case of the Bloch sphere, this is `cirq_bloch_sphere.py`.
 - Make sure that any additional modules and files are in separate subdirectories labeled accordingly.

### Developing Python modules for visualization

In order to actually get visualization output from our Python calls, we return strings of HTML and Javascript. In order to keep things organized, we include a parent class `Widget` (`widget.py`) which handles the configuration behind locating and reading files so that only code specific to each visualization lives in its main class. 

The main class for all visualizations should inherit from the `Widget` class located in this directory. Upon creating a new Widget, you should include a call to initialize the parent like so:
```python
class MyWidget(widget.Widget):
    def __init__(self,...):
        ...
        super().__init__('cirq_ts/dist/YOUR_BUNDLE_FILE.js')
        ...
```
This will allow you to easily access the absolute path to your bundle file, which should always be used when referencing it, with `self.bundle_file_path`. It also allows you to call `super().get_bundle_script()` to easily 
get full script content of your widget as a string. These methods should be used when generating your HTML outputs for boththe web browser and notebooks.

The `widget.py` file also contains methods like `write_output_file()` that should be used for all visualizations, as well as some other helpful methods. 

### Handling HTML output from Python
#### Viewing a visualization in a notebook setting
We capitalize on IPython's `_repr_html_` magic method to help display visualizations in the notebook. In the main class of your visualization, include a method like so:
```python
from cirq_web import widget
class MyWidget(widget.Widget):
    ...
    def _repr_html_(self):
        bundle_script = super().get_bundle_script()
        return f"""
            <meta charset="UTF-8">
            <div id="container"></div>
            {bundle_script}
            <script>YOUR_BUNDLE_FUNCTION<script>
        """
``` 
This will allow your visualization to be displayed in a notebook cell with:
```python
    widget = MyWidget()
    display(widget)
```

#### Generating a standalone HTML file from a visualization
Generating a standalone HTML file is a bit more involved than in a notebook, but not difficult. In the main class of your visualiztion, include a method like so:
```python
import webbrowser
from cirq_web import widget
class MyWidget(widget.Widget):
    ...
    def generate_html_file(self, output_directory='./', file_name="YOUR_VIZ.html", open_in_browser=False):
        bundle_script = super().get_bundle_script()
        contents = f"""
            <meta charset="UTF-8">
            <div id="container"></div>
            {bundle_script}
            <script>YOUR_BUNDLE_FUNCTION<script>
        """
        path_of_html_file = widget.write_output_file(output_directory, file_name, contents)
        
        if open_in_browser:
            webbrowser.open(str(path_of_html_file), new=2)
        
        return path_of_html_file
```
This code above writes a file named `YOUR_VIZ.html` to the specified output directory, returning the path of the file as a `Path` object. To get the path as string, use `str()`. If the `open_in_browser` flag is used, Python will automatically open your visualization in a new tab using the default browser on your computer. 

The important thing about generating standalone HTML files is that they can be sent and viewed anywhere, regardless of if the recipient has Cirq installed on their computer or not.


