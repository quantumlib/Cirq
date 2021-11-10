# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, cast, List, Optional
from types import ModuleType
from importlib.machinery import ModuleSpec
from importlib.abc import Loader

from contextlib import contextmanager
import importlib
from importlib import abc
import sys


class InstrumentedFinder(abc.MetaPathFinder):
    """A module finder used to hook the python import statement."""

    def __init__(
        self,
        finder: Any,
        module_name: str,
        wrap_module: Callable[[ModuleType], Optional[ModuleType]],
        after_exec: Callable[[ModuleType], None],
    ):
        """A module finder that uses an existing module finder to find a python
        module spec and intercept the execution of matching modules.

        Replace finders in `sys.meta_path` with instances of this class to
        instrument import statements.

        Args:
            finder: The original module finder to wrap.
            module_name: The fully qualified module name to instrument e.g.
                `'pkg.submodule'`.  Submodules of this are also instrumented.
            wrap_module: A callback function that takes a module object before
                it is run and either modifies or replaces it before it is run.
                The module returned by this function will be executed.  If None
                is returned the module is not executed and may be executed
                later.
            after_exec: A callback function that is called with the return value
                of `wrap_module` after that module was executed if `wrap_module`
                didn't return None.
        """

        self.finder = finder
        self.module_name = module_name
        self.match_components: List[str] = []
        if self.module_name:
            self.match_components = self.module_name.split('.')
        self.wrap_module = wrap_module
        self.after_exec = after_exec

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        components = fullname.split('.')
        spec = self.finder.find_spec(fullname, path=path, target=target)
        if spec is None:
            return None
        if components[: len(self.match_components)] == self.match_components:
            spec = self.wrap_spec(spec)
        return spec

    def wrap_spec(self, spec: Any) -> Any:
        spec.loader = InstrumentedLoader(spec.loader, self.wrap_module, self.after_exec)
        return spec


class InstrumentedLoader(abc.Loader):
    """A module loader used to hook the python import statement."""

    def __init__(
        self,
        loader: Any,
        wrap_module: Callable[[ModuleType], Optional[ModuleType]],
        after_exec: Callable[[ModuleType], None],
    ):
        """A module loader that uses an existing module loader and intercepts
        the execution of a module.

        Use `InstrumentedFinder` to instrument modules with instances of this
        class.

        Args:
            loader: The original module loader to wrap.
            module_name: The fully qualified module name to instrument e.g.
                `'pkg.submodule'`.  Submodules of this are also instrumented.
            wrap_module: A callback function that takes a module object before
                it is run and either modifies or replaces it before it is run.
                The module returned by this function will be executed.  If None
                is returned the module is not executed and may be executed
                later.
            after_exec: A callback function that is called with the return value
                of `wrap_module` after that module was executed if `wrap_module`
                didn't return None.
        """
        self.loader = loader
        self.wrap_module = wrap_module
        self.after_exec = after_exec

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return self.loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        wrapped_module = self.wrap_module(module)
        if wrapped_module is not None:
            self.loader.exec_module(module)
            self.after_exec(module)


@contextmanager
def wrap_module_executions(
    module_name: str,
    wrap_func: Callable[[ModuleType], Optional[ModuleType]],
    after_exec: Callable[[ModuleType], None] = lambda m: None,
    assert_meta_path_unchanged: bool = True,
):
    """A context manager that hooks python's import machinery within the
    context.

    `wrap_func` is called before executing the module called `module_name` and
    any of its submodules.  The module returned by `wrap_func` will be executed.
    """

    def wrap(finder: Any) -> Any:
        if not hasattr(finder, 'find_spec'):
            return finder
        return InstrumentedFinder(finder, module_name, wrap_func, after_exec)

    new_meta_path = [wrap(finder) for finder in sys.meta_path]

    try:
        orig_meta_path, sys.meta_path = sys.meta_path, new_meta_path
        yield
    finally:
        if assert_meta_path_unchanged:
            assert sys.meta_path == new_meta_path
        sys.meta_path = orig_meta_path


@contextmanager
def delay_import(module_name: str):
    """A context manager that allows the module or submodule named `module_name`
    to be imported without the contents of the module executing until the
    context manager exits.
    """
    delay = True
    execute_list = []

    def wrap_func(module: ModuleType) -> Optional[ModuleType]:
        if delay:
            execute_list.append(module)
            return None  # Don't allow the module to be executed yet
        return module  # Now allow the module to be executed

    with wrap_module_executions(module_name, wrap_func):
        importlib.import_module(module_name)

    yield  # Run the body of the context

    delay = False
    for module in execute_list:
        if module.__loader__ is not None and hasattr(module.__loader__, 'exec_module'):
            cast(Loader, module.__loader__).exec_module(module)  # Calls back into wrap_func


class LazyLoader(ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.

    This class is a modified version of a similar class in TensorFlow.

    To use, instead of importing the module normally
        ```
        import heavy_module
        ```
    define the module
        ```
        heavy_module = LazyLoader("heavy_module", globals(), "mypackage.heavy_module")
        ```
    """

    def __init__(self, local_name, parent_module_globals, name):
        """Create the LazyLoader module.

        Args:
            local_name: The local name that the module will be refered to as.
            parent_module_globals: The globals of the module where this should be imported.
                Typically this will be globals().
            name: The full qualified name of the module.
        """
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        self._module = None
        super().__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals."""
        # Import the target module and insert it into the parent's namespace
        if self._module:
            return self._module
        self._module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = self._module

        # Update this object's dict so that if someone keeps a reference to the LazyLoader,
        # lookups are efficient (__getattr__ is only called on lookups that fail).
        self.__dict__.update(self._module.__dict__)

        return self._module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)
