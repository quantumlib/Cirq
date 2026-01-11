# Copyright 2018 The Cirq Developers
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

import duet

def duet_sync(async_func):
    """Wraps an async method to run synchronously via duet. This wraps the duet.sync() call to
    ensure that the executor utilizes duet futures.

    Usage example:
        create_program = duet_sync(create_program_async)
    """

    # Use '_self' as the first arg name because duet uses 'self' as a hack to support inheritance,
    # at the expense of unwrapping the wrapper.
    async def wrapper(_self, *args, **kwargs):
        executor = None

        if hasattr(_self, '_executor'):
            executor = _self._executor
        elif hasattr(_self, 'context') and hasattr(_self.context, 'client'):
            executor = _self.context.client._executor

        if executor:
            with executor.duet_futures():
                return await async_func(_self, *args, **kwargs)
        else:
            return await async_func(_self, *args, **kwargs)

    sync_func = duet.sync(wrapper)
    sync_func.__name__ = async_func.__name__
    sync_func.__doc__ = async_func.__doc__

    return sync_func
