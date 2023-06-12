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

import dataclasses

@dataclasses.dataclass
class Setting(object):
    exp_dir: str
    eval_len: int
    topk: int
    mem_size: int
    mid_length: int
    use_cache: bool = True
    selection_method: str = "per_token_and_head"
    mem_cache_freq: int = 50
    eval_sample_size: int = 4000000
    lm_cache: str = "mem"
    

exp_dirs = {
    "arxiv_landmark": "./exps/arxiv_landmark",
    "arxiv_baseline": "./exps/arxiv_baseline",

    "pg19_landmark": "./exps/pg19_landmark",
    "pg19_baseline": "./exps/pg19_baseline",
    "pg19_xl": "./exps/pg19_xl",
}
settings = [
    dict(exp_dir=exp_dirs["pg19_baseline"], eval_len=360, mid_length=360, 
         lm_cache="none", mem_cache_freq=None, mem_size=None, topk=None, use_cache=False,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_baseline"], eval_len=512, mid_length=512, 
         lm_cache="none", mem_cache_freq=None, mem_size=None, topk=None, use_cache=False,eval_sample_size=None),
        
    dict(exp_dir=exp_dirs["pg19_xl"], eval_len=2048, mid_length=256, 
         lm_cache="kv", mem_cache_freq=None, mem_size=256, topk=None,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_xl"], eval_len=4096, mid_length=256, 
         lm_cache="kv", mem_cache_freq=None, mem_size=256, topk=None,eval_sample_size=None),

    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=512, mid_length=250, mem_size=10, topk=2,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=2,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=350, mem_size=40, topk=2,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=300, mem_size=40, topk=3,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=20, topk=4,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=4,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=4096, mid_length=250, mem_size=40, topk=4,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=4096, mid_length=250, mem_size=80, topk=2,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=4096, mid_length=250, mem_size=80, topk=4,eval_sample_size=None),

    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=2,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=4,eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=4096, mid_length=250, mem_size=40, topk=4,eval_sample_size=None),

    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=2,
         selection_method="max_over_heads",eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=4,
         selection_method="max_over_heads",eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=4096, mid_length=250, mem_size=80, topk=4,
         selection_method="max_over_heads",eval_sample_size=None),

    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=2,
         selection_method="max_over_tokens",eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=4,
         selection_method="max_over_tokens",eval_sample_size=None),
    dict(exp_dir=exp_dirs["pg19_landmark"], eval_len=4096, mid_length=250, mem_size=80, topk=4,
         selection_method="max_over_tokens",eval_sample_size=None),

    dict(exp_dir=exp_dirs["arxiv_baseline"], eval_len=360, mid_length=360, 
         lm_cache=None, mem_cache_freq=None, mem_size=None, topk=None, use_cache=False),
    dict(exp_dir=exp_dirs["arxiv_baseline"], eval_len=512, mid_length=512, 
         lm_cache=None, mem_cache_freq=None, mem_size=None, topk=None, use_cache=False),

    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=512, mid_length=250, mem_size=10, topk=2),
    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=2),
    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=2048, mid_length=350, mem_size=40, topk=2),
    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=2048, mid_length=300, mem_size=40, topk=3),
    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=2048, mid_length=250, mem_size=20, topk=4),
    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=2048, mid_length=250, mem_size=40, topk=4),
    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=4096, mid_length=250, mem_size=40, topk=4),
    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=4096, mid_length=250, mem_size=80, topk=2),
    dict(exp_dir=exp_dirs["arxiv_landmark"], eval_len=4096, mid_length=250, mem_size=80, topk=4),
]

import itertools
def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

flat_settings = []
for setting in settings:
    flat_settings.extend(product_dict(**{x: y if isinstance(y, list) else [y] for x, y in setting.items()}))

settings = [Setting(**d) for d in flat_settings]
last_exp_dir = None

print ("#!/bin/bash")
for setting in settings:
     s_lines = []
     if last_exp_dir != setting.exp_dir:
          s_lines.append("""EXP_DIR="{exp_dir}";""".format(**dataclasses.asdict(setting)))
     last_exp_dir = setting.exp_dir
     use_cache_str = "--use_cache" if setting.use_cache else ""
     mem_size_flag = ""
     s_lines += ["""
     filename="$EXP_DIR/eval-{eval_len}-{selection_method}-{topk}-memsize{mem_size}-midlength{mid_length}-memcachefreq{mem_cache_freq}"; 
     grep val_acc $filename /dev/null; 
     if [[ $? -ne 0 ]]; then 
          script -c \\
          "python eval.py \\
               --checkpoint  $EXP_DIR \\
               --distributed_backend None  \\
               --lm_cache {lm_cache} \\""","""
               --mem_cache_size {mem_size} \\""" if setting.mem_size is not None else "","""
               --mem_cache_freq {mem_cache_freq} \\""" if setting.mem_cache_freq is not None else "", """
               --mem_freq None \\
               --eval_seq_length {eval_len} \\
               --cache_selection_method {selection_method} \\""","""
               --cache_topk {topk} \\""" if setting.topk is not None else "", """
               --no_compile \\
               --batch_size 16 \\
               --mid_length {mid_length} \\
               --positional_encoder rotary \\
               --pos_jump_on_mem 0   \\
               {use_cache_str} \\""", """
               --eval_sample_size {eval_sample_size}""" if setting.eval_sample_size is not None else "", """
               " $filename; 
     fi;"""]
     print ("".join(s_lines).format(**dataclasses.asdict(setting), use_cache_str=use_cache_str))
