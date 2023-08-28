# Cirq modules

Cirq has a modular architecture and is organized in a monorepo, all of the modules follow the same folder structure.
Each module is structured as follows. Let's take as example a module named `cirq-example`:

```
cirq-example
├── cirq_example
│   ├── __init__.py
│   ├── _version.py
│   ├── json_resolver_cache.py
│   └── json_test_data
│       ├── __init__.py
│       └── spec.py
├── LICENSE
├── README.rst
├── requirements.txt
├── setup.cfg
└── setup.py
```

Note that typically there is only a single top level package, `cirq_example` - but there might be exceptions.

Additionally, there is a metapackage "cirq" that's a completely different beast and just depends on the modules.
This enables `pip install cirq` to have all the included modules to be installed for our users.

All modules should depend on `cirq-core`, which is the central, core library for Cirq.

## Packaging

Each package gets published to PyPi as a separate package. To build all the wheel files locally, use

```bash
dev_tools/packaging/produce-package.sh ./dist `./dev_tools/packaging/generate-dev-version-id.sh`
```

Packages are versioned together, share the same version number, and are released together.

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
    @functools.lru_cache()  # pragma: no cover
    def _class_resolver_dictionary() -> Dict[str, ObjectFactory]:  # pragma: no cover
        return {}
    ```
2. Register the resolver cache - at _the end_ of the `<top_level_package>/__init__.py`:
    ```python

    # Registers cirq_example's public classes for JSON serialization.
    from cirq.protocols.json_serialization import _register_resolver
    from cirq_example.json_resolver_cache import _class_resolver_dictionary
    _register_resolver(_class_resolver_dictionary)

    ```
3. Add the `<top_level_package>/json_test_data` folder with the following content:
   1. `spec.py` contains the core test specification for JSON testing, that plugs into the central framework:
       ```python
       import pathlib
       import cirq_example
       from cirq_example.json_resolver_cache import _class_resolver_dictionary

       from cirq.testing.json import ModuleJsonTestSpec

       TestSpec = ModuleJsonTestSpec(
           name="cirq_example",
           packages=[cirq_example],
           test_data_path=pathlib.Path(__file__).parent,
           not_yet_serializable=[],
           should_not_be_serialized=[],
           resolver_cache=_class_resolver_dictionary(),
           deprecated={},
        )
       ```
   2. `__init__.py` should import `TestSpec` from `spec.py`
   3. in `cirq/protocols/json_serialization_test.py` add `'cirq_example':None` to the `TESTED_MODULES` variable. `TESTED_MODULES` is also used to prepare the test framework for deprecation warnings.
      With new modules, we use`None` as there is no deprecation setup.

You can run `check/pytest-changed-files` and that should execute the json_serialization_test.py as well.

That's it! Now, you can follow the [Serialization guide](./serialization.md) for adding and removing serializable objects.

# Utilities

## List modules

To iterate through modules, you can list them by invoking `dev_tools/modules.py`.

```bash
python dev_tools/modules.py list
```

There are different modes of listing (e.g the folder, package-path, top level package),
you can refer to `python dev_tools/modules.py list --help` for the most up to date features.
