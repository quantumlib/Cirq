# Cirq modules 

Cirq has a modular architecture and is organized in a monorepo, all of the modules are in the same folder structure.
Each module is structured as follows: 

```
cirq-<module-name>
   <top level package 1> 
   <top level package 2>
   ...
   setup.py
   setup.cfg
   requirements.txt
   LICENSE
   README.rst 
...
setup.py # metapackage 
```

Note that typically there is only a single top level package, but there might be exceptions. 

The highest level module, `cirq` is an exception, as it is a metapackage, kind of a "parent" module, that only contains the set of default submodules as requirements. 
This enables `pip install cirq` to have all the included submodules to be installed for our users.

All submodules should depend on `cirq-core`, which is the central, core library for Cirq.    

## Packaging 

Each package gets published to PyPi as a separate package. To build all the wheel files locally, use

```bash
dev_tools/packaging/produce-package.sh ./dist `./dev_tools/packaging/generate-dev-version-id.sh`
```
 
Packages are versioned together, share the exact same version, and released together. 

## Setting up a new module

To setup a new module follow these steps: 

1. Create the folder structure above, copy the files based on an existing module
    1. LICENSE should be the same
    2. README.rst will be the documentation that appears in PyPi
    3. setup.py should specify an `install_requires` configuration that has `cirq-core=={module.version}` at the minimum
2. Setup JSON serialization for each top level python package 


### Setting up JSON serialization

1. Add the `<top_level_package>/json_resolver_cache.py` file 
    ```python
    @functools.lru_cache(maxsize=1)  # coverage: ignore
    def _class_resolver_dictionary() -> Dict[str, ObjectFactory]:  # coverage: ignore
        return {}
    ```
2. Register the resolver cache - at _the end_ of the `<top_level_package>/__init__.py`:
    ```python
    def _register_resolver() -> None:
        """Registers the cirq_mynewtoplevelpackage's public classes for JSON serialization."""
        from cirq.protocols.json_serialization import _internal_register_resolver
        from cirq_mynewtoplevelpackage.json_resolver_cache import _class_resolver_dictionary
    
        _internal_register_resolver(_class_resolver_dictionary)
    
    
    _register_resolver()
    ``` 
3. Add the `<top_level_package>/json_test_data` folder with the following content: 
   1. `__init__.py` should export `TestSpec` from `spec.py`
   2. `spec.py` contains the core test specification for JSON testing, that plugs into the central framework. It should have the minimal setup:    
       ```python
       import pathlib
       import cirq_mynewtoplevelpackage
       from cirq_mynewtoplevelpackage.json_resolver_cache import _class_resolver_dictionary
       
       from cirq.testing.json import ModuleJsonTestSpec
       
       TestSpec = ModuleJsonTestSpec(
           name="cirq_mynewtoplevelpackage",
           packages=[cirq_mynewtoplevelpackage],
           test_data_path=pathlib.Path(__file__).parent,
           not_yet_serializable=[],
           should_not_be_serialized=[],
           resolver_cache=_class_resolver_dictionary(),
           deprecated={},
        )
       ```
   3. in `cirq/protocols/json_serialization_test.py` add `'cirq_mynewtoplevelpackage':None` to the `TESTED_MODULES` variable
 
You can run `check/pytest-changed-files` and that should execute the json_serialization_test.py as well. 

That's it! Now, you can follow the [Serialization guide](./serialization.md) for adding and removing serializable objects.

# Utilities 

## List modules 

To iterate through modules, you can list them by invoking `dev_tools/modules.py`. 
 
```bash
python dev_tools/modules.py --list 
```

There are different modes of listing (e.g the folder, package-path, top level package), 
you can refer to `python dev_tools/modules.py --list --help` for the most up to date features. 
