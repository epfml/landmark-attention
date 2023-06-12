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

import os
import zipfile
import urllib
import numpy as np
import tiktoken
import torch
import regex
import multiprocessing
import itertools
import functools

from .utils import add_mem_tokens

ARXIVMATH_ORIGINAL_PATH = "./data/proof-pile/"


def get_path(config):
    dataset_name = f"arxiv_mem={config.mem_freq}"
    return os.path.join(os.path.dirname(__file__), f"datasets/{dataset_name}/")
    
def prepare_arxivmath_data(config):
    DATA_PATH = get_path(config)
    print(DATA_PATH)
    os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(os.path.join(DATA_PATH, 'train.bin')):
        train_data = np.memmap(os.path.join(ARXIVMATH_ORIGINAL_PATH, 'train.bin'), dtype=np.uint16, mode='r')
        raw_tokenized_train = add_mem_tokens(config.landmark_id, train_data, config.mem_freq)
        train_tokenized = np.array(raw_tokenized_train, dtype=np.uint16) 
        train_tokenized.tofile(os.path.join(DATA_PATH, 'train.bin'))
    
    if not os.path.exists(os.path.join(DATA_PATH, 'val.bin')):
        val_data = np.memmap(os.path.join(ARXIVMATH_ORIGINAL_PATH, 'validation.bin'), dtype=np.uint16, mode='r')
        raw_tokenized_eval = add_mem_tokens(config.landmark_id, val_data, config.mem_freq)
        eval_tokenized = np.array(raw_tokenized_eval, dtype=np.uint16)
        eval_tokenized.tofile(os.path.join(DATA_PATH, 'val.bin'))
    print("completed the tokenization process!")


def get_arxivmath_data(config):
    DATA_PATH = get_path(config)
    
    train_data = np.memmap(os.path.join(DATA_PATH, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_PATH, 'val.bin'), dtype=np.uint16, mode='r')

    return {'train': train_data, 'val': val_data}
