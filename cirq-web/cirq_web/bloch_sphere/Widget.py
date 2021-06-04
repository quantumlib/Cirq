import os
import IPython
from enum import Enum

class Env(Enum):
    JUPYTER = 1
    COLAB = 2
    OTHER = 3

class Widget:
    """Parent class for all widgets, containing standard methods to help
    print the output to a widget's respective shell.
    """

    def determine_env(self):
        """Determines if a Widget is being run in a Jupyter notebook"""
        env = IPython.get_ipython().__class__.__name__
        if env == 'ZMQInteractiveShell':
            return Env.JUPYTER
        else:
            return Env.OTHER

    def determine_repr_path(self):
        """
        Determines the correct path for each widget's 
        _repr_html() method.

        If running from the command line, this function will 
        return nothing, since HTML output isn't supported. Use
        generate_HTML_file() instead. 

        Access files in Jupyter notebook by currently
        going through localhost:PORT/tree/[directory_path]
        """
        env = self.determine_env()
        if env == Env.JUPYTER:
            # Jupyter notebook starts out in localhost:PORT/examples directory
            return '../tree'
        elif env == Env.OTHER:
            return None


    def unpack_bundle_src(self, path):
        """Dumps the contents of a particular bundle file into a script tag.

        Args:
            path: the path to the bundle file
        """
        try:
            bundle_file_path = path
            bundle_file = open(bundle_file_path, 'r')
            bundle_file_contents = bundle_file.read()
            bundle_file.close()
            bundle_html = f'<script>{bundle_file_contents}</script>'
        except:
            print('Failed')

        return bundle_html

    def write_output_file(self, output_directory, file_name, contents):
        """Writes the output file and returns its absolute path.

        Args:
            output_directory: the directory in which the output file will be
            generated.

            file_name: the name of the output file. Default is 'bloch_sphere'

            contents: the contents of the file
        """
        try:
            file_to_write_in = open(output_directory + file_name, 'w')
            file_to_write_in.write(contents)
            file_to_write_in.close()
        except:
            print('Failed')

        absolute_path = os.path.abspath(output_directory + file_name)
        print(f'File can be found at: {absolute_path}')
        return absolute_path
