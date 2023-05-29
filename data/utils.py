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

import numpy as np
import torch
import multiprocessing
import itertools
import functools


def apply_add_mem_tokens(mem_id, tokens_filename, freq, start_idx, end_idx):
    tokens = np.memmap(tokens_filename, dtype=np.uint16, mode='r')
    print(f"Processing {start_idx}-{end_idx}")
    tokens_with_mem = []
    for t_idx in range(start_idx, end_idx):
        t =  tokens[t_idx]
        tokens_with_mem.append(t)
        if freq is not None and t_idx % freq == freq - 1:
            tokens_with_mem.append(mem_id)
    return tokens_with_mem

def add_mem_tokens(mem_id, tokens, freq, n_workers=32):
    print(len(tokens))
    with multiprocessing.Pool(n_workers) as pool:
        ids = list(range(0, len(tokens), 10 * 1000 * 1000))
        pair_ids = zip(ids, ids[1:] + [len(tokens)])
        apply = functools.partial(apply_add_mem_tokens, mem_id, tokens.filename, freq)
        return list(itertools.chain.from_iterable(pool.starmap(apply, pair_ids)))
