# Copyright 2023 Together Computer
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

# Lint as: python3
"""RedPajama: An Open-Source, Clean-Room 1.2 Trillion Token Dataset."""


import json

import datasets
import traceback
import numpy as np
import math

logger = datasets.logging.get_logger(__name__)


_DESCRIPTION = """\
RedPajama is a clean-room, fully open-source implementation of the LLaMa dataset.
"""

_URL_LISTS = {
    "arxiv": "urls/arxiv.txt",
    "book": "urls/book.txt",
    "c4": "urls/c4.txt",
    "common_crawl": "urls/common_crawl.txt",
    "github": "urls/github.txt",
    "stackexchange": "urls/stackexchange.txt",
    "wikipedia": "urls/wikipedia.txt",
}


class RedPajama1TConfig(datasets.BuilderConfig):
    """BuilderConfig for RedPajama sample."""

    def __init__(self, *args, subsets, p_sample=None, **kwargs):
        """BuilderConfig for RedPajama.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(RedPajama1TConfig, self).__init__(**kwargs)

        self.subsets = subsets
        self.p_sample = p_sample


class RedPajama1T(datasets.GeneratorBasedBuilder):
    """RedPajama: Reproducing the LLaMA training dataset of over 1.2 trillion tokens. Version 1.0.0."""
    BUILDER_CONFIG_CLASS = RedPajama1TConfig
    BUILDER_CONFIGS = [
        RedPajama1TConfig(
            subsets = list(_URL_LISTS.keys()),
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
        RedPajama1TConfig(
            subsets = list(_URL_LISTS.keys()),
            name="plain_text_tenpercent",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
            p_sample=0.1
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "meta": datasets.Value("string"),
                    "red_pajama_subset": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        url_lists = dl_manager.download_and_extract({
            subset: _URL_LISTS[subset] for subset in self.config.subsets
        })

        urls = {}
        rng = np.random.default_rng(seed=2)

        for subset, url_list in url_lists.items():
            with open(url_list, encoding="utf-8") as f:
                urls[subset] = [line.strip() for line in f]
            if self.config.p_sample is not None:
                urls[subset] = rng.choice(
                    urls[subset], 
                    size=int(math.ceil(len(urls[subset]) * self.config.p_sample)), replace=False).tolist()

        downloaded_files = dl_manager.download(urls)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs = {
                    "files": {
                        subset: downloaded_files[subset]
                        for subset in self.config.subsets
                    }
                }
            )
        ]

    def _generate_examples(self, files):
        """This function returns the examples in the raw (text) form."""
        key = 0
        for subset in files:
            if subset == "common_crawl":
                import zstandard as zstd

                for path in files[subset]:
                    with zstd.open(open(path, "rb"), "rt", encoding="utf-8") as f:
                        for i, row in enumerate(f):
                            try:
                                data = json.loads(row)
                                text = data["text"]
                                del data["text"]
                                yield key, {
                                    "text": text,
                                    "meta": json.dumps(data),
                                    "red_pajama_subset": subset,
                                }
                                key += 1
                            except Exception as e:
                                print(f'Subset: {subset}')
                                print(f'Path: {path}')
                                print(f'Row: {row}')
                                traceback.print_exc()

                                raise e
            else:
                for path in files[subset]:
                    with open(path, encoding="utf-8") as f:
                        for i, row in enumerate(f):
                            try:
                                data = json.loads(row)
                                if "meta" not in data:
                                    text = data["text"]
                                    del data["text"]
                                    yield key, {
                                        "text": text,
                                        "meta": json.dumps(data),
                                        "red_pajama_subset": subset,
                                    }
                                else:
                                    yield key, {
                                        "text": data["text"],
                                        "meta": data["meta"],
                                        "red_pajama_subset": subset,
                                    }
                                key += 1
                            except Exception as e:
                                print(f'Subset: {subset}')
                                print(f'Path: {path}')
                                print(f'Row: {row}')
                                traceback.print_exc()

                                raise e
