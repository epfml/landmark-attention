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

from torch import nn

class LMCacheStorage(nn.Module):

    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        #self._layer = [layer]

    #@property
    #def layer(self):
    #    return self._layer[0]

    def store_in_cache(self, keys, values_dict):
        pass

    def retrieve_for_query(self, q, cache_context, pos_emb_closure, start_index):
        return None, {}

    def clear_state(self):
        pass


class LMCacheContext(object):
    pass


class LMCache(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_storages_map = dict()
        self.layer_storages = nn.ModuleList()
        self.cache_storage = self.get_cache_storage()
        self.context_class = self.get_context_class()

    def get_cache_storage(self):
        return LMCacheStorage

    def get_context_class(self):
        return LMCacheContext

    def forward(self, x):
        return x, 0, self.get_context_class()

    def get_final_logits(self, logits):
        return logits

    def get_storage_for_layer(self, l):
        if l not in self.layer_storages_map:
            self.layer_storages_map[l] = len(self.layer_storages)
            self.layer_storages.append(self.cache_storage(self.config, l))
        return self.layer_storages[self.layer_storages_map[l]]

    def clear_state(self):
        for storage in self.layer_storages:
            storage.clear_state()
