# Copyright 2023 Amirkeivan Mohtashami, Martin Jaggi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import cache, mem_cache, kv_cache, kv_cache_train

CACHES = {
    "none": cache.LMCache,
    "mem": mem_cache.MemLMCache,
    "kv": kv_cache.KVLMCache,
    "kv_train": kv_cache_train.KVLMCache
}


def get_cache(cache_name):
    return CACHES[cache_name]


def registered_caches():
    return CACHES.keys()
