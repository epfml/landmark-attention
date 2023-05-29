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

import math

import torch

from .cache import LMCache, LMCacheStorage

class MemLMCacheStorage(LMCacheStorage):

    def __init__(self, config, layer):
        super().__init__(config, layer)
        n_embd_per_head = config.n_embd // config.n_head
        self.register_buffer("cache_mem_k", torch.empty((config.batch_size, config.mem_cache_size, config.mem_cache_freq + 1, config.n_head, n_embd_per_head)), persistent=False)
        self.register_buffer("cache_mem_v", torch.empty((config.batch_size, config.mem_cache_size, config.mem_cache_freq + 1, config.n_head, n_embd_per_head)), persistent=False)
        self.register_buffer("last_incomplete_k", torch.empty((config.batch_size, config.n_head, config.mem_cache_freq + 1, n_embd_per_head)), persistent=False)
        self.register_buffer("last_incomplete_v", torch.empty((config.batch_size, config.n_head, config.mem_cache_freq + 1, n_embd_per_head)), persistent=False)
        self.register_buffer("last_incomplete_ismem", torch.empty((config.batch_size, config.mem_cache_freq + 1), dtype=torch.bool), persistent=False)
        self.last_incomplete_len = 0
        self.cache_iter = 0
        self.cache_size = 0
        self.clear_state()
        
    def clear_state(self):
        self.last_incomplete_len = 0
        self.cache_iter = 0
        self.cache_size = 0

    def retrieve_for_query(self, q, cache_context, pos_emb_closure, start_index):
        B, nh, T, hs = q.size() 
        last_incomplete_k = pos_emb_closure.adapt_keys(self.last_incomplete_k[:B, :, :self.last_incomplete_len], start_index=start_index - self.last_incomplete_len)
        att_incomplete = (q @ last_incomplete_k.transpose(-2, -1)) * (1.0 / math.sqrt(last_incomplete_k.size(-1)))
        last_incomplete_v = self.last_incomplete_v[:B, :, :self.last_incomplete_len].unsqueeze(2).expand(B, nh, T, self.last_incomplete_len, hs)
        last_incomplete_mem = self.last_incomplete_ismem[:B, :self.last_incomplete_len]

        if self.cache_size == 0 or self.config.cache_topk == 0:
            return att_incomplete, {'v': last_incomplete_v.clone(), 'is_mem': last_incomplete_mem.clone()}
        
        top_k = self.config.cache_topk
        k_with_cached_mem = self.cache_mem_k[:B, :self.cache_size, -1].view(B, -1, nh, hs).transpose(1, 2) # (B, nh, T, hs)
        mem_indices = torch.cat((
            torch.arange(k_with_cached_mem.shape[2] - self.cache_iter, k_with_cached_mem.shape[2], device=q.device),
            torch.arange(0, k_with_cached_mem.shape[2] - self.cache_iter, device=q.device),
        )).unsqueeze(0).expand(B, -1) 

        mem_indices = torch.where(mem_indices >= (k_with_cached_mem.shape[2] - top_k), mem_indices - (k_with_cached_mem.shape[2] - top_k) + 1, 0)
        mem_token_indices = mem_indices * self.cache_mem_k.shape[2] + self.cache_mem_k.shape[2] - 1
        k_with_cached_mem = pos_emb_closure.adapt_keys(k_with_cached_mem, indices=mem_token_indices.unsqueeze(1).expand(B, nh, -1))
        mem_att = (q @ k_with_cached_mem.transpose(-2, -1)) * (1.0 / math.sqrt(k_with_cached_mem.size(-1)))
        mem_att = torch.nn.functional.softmax(mem_att, dim=-1) # (B, nh, T, mem_count)
        if self.config.cache_selection_method == "max_over_heads":
            mem_att = mem_att.amax(dim=1).unsqueeze(1).expand(B, nh, T, -1)
        elif self.config.cache_selection_method == "per_token_and_head":
            pass
        elif self.config.cache_selection_method == "max_over_tokens":
            mem_att = mem_att.amax(dim=2, keepdim=True).expand(B, nh, T, -1)
        else:
            raise NotImplementedError

        mem_att = mem_att.topk(dim=-1,k=top_k)[1]
        if top_k > 1:
            # sort
            mem_att = torch.where(
                mem_att < self.cache_iter, 
                (k_with_cached_mem.shape[2] - self.cache_iter) + mem_att, 
                mem_att - self.cache_iter)
            mem_att = mem_att.sort(dim=-1)[0] # (B, nh, T, top_k)
            mem_att = torch.where(
                mem_att >= (k_with_cached_mem.shape[2] - self.cache_iter), 
                mem_att - (k_with_cached_mem.shape[2] - self.cache_iter), 
                mem_att + self.cache_iter) # (B, nh, T, top_k)
        
        mem_att_or = mem_att
        mem_att = mem_att.permute(0, 2, 3, 1).view(B, T, top_k, 1, nh, 1).expand(B, T, top_k, self.cache_mem_k.shape[2], nh, hs)
        mem_k = self.cache_mem_k[:B].unsqueeze(1).expand(B, T, -1, -1, -1, -1).take_along_dim(mem_att, dim=2).view(B, T, -1, nh, hs)
        mem_k = mem_k.permute((0, 3, 1, 2, 4))
        mem_v = self.cache_mem_v[:B].unsqueeze(1).expand(B, T, -1, -1, -1, -1).take_along_dim(mem_att, dim=2).view(B, T, -1, nh, hs) # (B, T, top_k * mem_block, nh, hs)
        mem_v = mem_v.permute((0, 3, 1, 2, 4)) # (B, nh, T, mem_block, hs)

        k_indices = torch.arange(0, self.cache_mem_k.shape[2] * top_k, device=q.device)
        chosen_indices = mem_indices.view(B, 1, 1, -1).expand(B, nh, T, -1).take_along_dim(mem_att_or, dim=-1)
        k_indices = (((chosen_indices > 0) * self.cache_mem_k.shape[2]).unsqueeze(-1) + k_indices.view(1, 1, 1, top_k, -1)).view(B, nh, T, -1) # (B, nh, T, top_k * mem_block)
       
        is_mem = torch.cat((q.new_zeros((B, top_k, self.cache_mem_k.shape[2] - 1), dtype=torch.bool), q.new_ones((B, top_k, 1), dtype=torch.bool)), dim=-1).view(B, -1)
       
        mem_k = pos_emb_closure.adapt_keys(mem_k.reshape(B, nh, -1, hs), indices=k_indices.reshape(B, nh, -1)).view(B, nh, T, -1, hs)
        att_k = (q.unsqueeze(3) @ mem_k.transpose(-2, -1)).squeeze(3) * (1.0 / math.sqrt(mem_k.size(-1)))
        att_k = pos_emb_closure.adapt_attention_before_softmax(att_k, start_query_index=start_index, k_indices=k_indices)

        
        att_prefix = torch.cat((att_k, att_incomplete), dim=-1)

        v_prefix = torch.cat((mem_v, last_incomplete_v), dim=-2)

        is_mem_prefix = torch.cat((is_mem, last_incomplete_mem), dim=-1)

        return att_prefix, {'v': v_prefix, 'is_mem': is_mem_prefix}

    def store_in_cache(self, keys, values_dict):

        B, nh, T, hs = keys.size()
        k = torch.cat((self.last_incomplete_k[:B, :, :self.last_incomplete_len], keys), dim=-2)
        v = torch.cat((self.last_incomplete_v[:B, :, :self.last_incomplete_len], values_dict['v']), dim=-2)
        is_mem = torch.cat((self.last_incomplete_ismem[:B, :self.last_incomplete_len], values_dict['is_mem']), dim=-1)
        B, nh, T, hs = k.size()

        incomplete_len = T % self.cache_mem_k.shape[2]
        full_len = T - incomplete_len
        k, incomplete_k = torch.split(k, (full_len, incomplete_len), dim=-2)
        v, incomplete_v = torch.split(v, (full_len, incomplete_len), dim=-2)
        is_mem, incomplete_ismem = torch.split(is_mem, (full_len, incomplete_len), dim=-1)
        self.last_incomplete_k[:B, :, :incomplete_len].copy_(incomplete_k)
        self.last_incomplete_v[:B, :, :incomplete_len].copy_(incomplete_v)
        self.last_incomplete_ismem[:B, :incomplete_len].copy_(incomplete_ismem)
        self.last_incomplete_len = incomplete_len
        T = full_len
        assert T % self.cache_mem_k.shape[2] == 0
        is_mem_for_cache = is_mem.view(B, -1, self.cache_mem_k.shape[2])

        assert is_mem_for_cache[..., -1].all()
        assert not is_mem_for_cache[..., :-1].any()
        added_size = is_mem_for_cache.shape[1]
        k_for_cache = k.transpose(1, 2).view(B, added_size, self.cache_mem_k.shape[2], nh, hs)
        v_for_cache = v.transpose(1, 2).view(B, added_size, self.cache_mem_v.shape[2], nh, hs)
        is_mem_for_cache = is_mem_for_cache[:, -self.cache_mem_k.shape[1]:]
        k_for_cache = k_for_cache[:, -self.cache_mem_k.shape[1]:]
        v_for_cache = v_for_cache[:, -self.cache_mem_k.shape[1]:]
        self.cache_iter = (self.cache_iter + added_size - is_mem_for_cache.shape[1]) % self.cache_mem_k.shape[1]
        self.cache_size += added_size - is_mem_for_cache.shape[1]
        added_size = is_mem_for_cache.shape[1]
        # torch._assert(added_size <= self.cache_mem_k.shape[1], "Should fit. Sanity check")

        if self.cache_iter + added_size >= self.cache_mem_k.shape[1]:
            next_iter = (self.cache_iter + added_size) - self.cache_mem_k.shape[1]
            rem = (self.cache_mem_k.shape[1] - self.cache_iter)
            self.cache_mem_k[:B, :next_iter].copy_(k_for_cache[:, rem:])
            self.cache_mem_k[:B, self.cache_iter:].copy_(k_for_cache[:, :rem])
            self.cache_mem_v[:B, :next_iter].copy_(v_for_cache[:, rem:])
            self.cache_mem_v[:B, self.cache_iter:].copy_(v_for_cache[:, :rem])
        else:
            next_iter = self.cache_iter + added_size
            self.cache_mem_k[:B, self.cache_iter:next_iter].copy_(k_for_cache)
            self.cache_mem_v[:B, self.cache_iter:next_iter].copy_(v_for_cache)
        
        self.cache_iter = next_iter
        self.cache_size += added_size

class MemLMCacheContext(object):
    def __init__(self):
        self.group_prob = None


class MemLMCache(LMCache):

    def __init__(self, config):
        super().__init__(config)
        self.last_incomplete_len = 0
        self.total_len = 0

    def get_cache_storage(self):
        return MemLMCacheStorage

    def get_context_class(self):
        return MemLMCacheContext

    def forward(self, x):
        
        previous_incomplete_len = self.last_incomplete_len
        #print("Concatenating with {}".format(previous_incomplete_len))
        #print("Being forward", x.shape)
        B, T = x.size()
        
        incomplete_placeholder = x.new_full((B, previous_incomplete_len), -1)
        x = torch.cat((incomplete_placeholder, x), dim=-1)
        B, T = x.size()
        incomplete_len = T % self.config.mem_cache_freq
        full_len = T - incomplete_len
        mem_x, incomplete_x = torch.split(x, (full_len, incomplete_len), dim=-1)
        mem_x = mem_x.view(B, -1, self.config.mem_cache_freq)
        mem_x = torch.cat((mem_x, mem_x.new_full((mem_x.shape[0], mem_x.shape[1], 1), self.config.landmark_id)), dim=-1)
        x = torch.cat((mem_x.view(B, -1), incomplete_x), dim=-1)[:, previous_incomplete_len:]
        self.last_incomplete_len = incomplete_len
        #print("End forward", x.shape)
        #print(x)
        prev_total_len = self.total_len
        self.total_len += x.shape[1]
        start_index = min(prev_total_len // (self.config.mem_cache_freq + 1), (self.config.cache_topk + 1)) * (self.config.mem_cache_freq + 1) + previous_incomplete_len
        return x, start_index, self.context_class()

    def get_final_logits(self, x):
        B, T, C = x.size()
        incomplete_len = self.last_incomplete_len
        T_with_mem = T - incomplete_len
        if T_with_mem <= 0: 
            incomplete_len = T
            T_with_mem = 0
            x, incomplete = torch.split(x, (0, T), dim=1)
            previous_incomplete_len = -T_with_mem
        else:
            x, incomplete = torch.split(x, (T_with_mem, incomplete_len), dim=1)
            previous_incomplete_len = (self.config.mem_cache_freq + 1 - T_with_mem % (self.config.mem_cache_freq + 1)) % (self.config.mem_cache_freq + 1)
        incomplete_placeholder = x.new_full((B, previous_incomplete_len, C), -1)
        x = torch.cat((incomplete_placeholder, x), dim=1).view(B, -1, self.config.mem_cache_freq + 1, C)
        x = x[:, :, :-1].reshape(B, -1, C)[:, previous_incomplete_len:]
        return torch.cat((x, incomplete), dim=1)

    def clear_state(self):
        super().clear_state()
        self.last_incomplete_len = 0
        self.total_len = 0
        
