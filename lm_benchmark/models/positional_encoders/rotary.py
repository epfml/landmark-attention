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

import torch
from torch import nn

from .encoder import PositionalEncoder, PositionalEncoderClosure
from .rotary_utils import apply_rotary_emb


class RotaryPositionalEncoderClosure(PositionalEncoderClosure):

    def adapt_vector_for_indices(self, v, indices):
        #changer = torch.zeros_like(indices)
        #changer[50::51] = 1
        #indices -= torch.cumsum(changer, dim=-1)

        *other_dims, T, hs = v.shape
        if T == 0:
            return v
        other_dims_prefix = other_dims[:len(other_dims) - len(indices.shape) + 1]
        freqs = (indices.unsqueeze(-1) * self.encoder.freqs.view(1, -1)).unsqueeze(-1).expand(*indices.shape, -1, 2).reshape(*indices.shape, hs)
        freqs = freqs.view([1] * len(other_dims_prefix) + list(indices.shape) + [hs]).expand(*v.shape)
        v = apply_rotary_emb(freqs, v)
        return v

    def _adapt_keys_for_indices(self, k, indices):
        return self.adapt_vector_for_indices(k, indices)

    def adapt_queries(self, q, start_index):
        T = q.shape[-2]
        indices = torch.arange(start_index, T + start_index, device=q.device)
        return self.adapt_vector_for_indices(q, indices)


class RotaryPositionalEncoder(PositionalEncoder):

    def __init__(self, config):
        super().__init__(config)
        self.max_pos_log = 4
        self.max_pos_base = 10  
        n_embd_per_head = config.n_embd // config.n_head
        freqs =  (self.max_pos_base ** (-self.max_pos_log * torch.arange(0, n_embd_per_head, 2)[:(n_embd_per_head // 2)].float() / n_embd_per_head))
        self.register_buffer("freqs", freqs)

    closure_model = RotaryPositionalEncoderClosure
