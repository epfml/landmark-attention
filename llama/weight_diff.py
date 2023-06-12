#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file has been changed by Amirkeivan Mohtashami
# to take into account the new token in the embedding layer

import os
from typing import Optional

import fire
import torch
import tqdm
import transformers
from train import smart_tokenizer_and_embedding_resize
import llama_mem

@torch.inference_mode()
def make_diff(
    path_raw: str, path_tuned: str, path_diff: str, device="cpu",  # "cuda" or "cpu"
):
    """Make the weight diff.

    This function is given to present full transparency of how the weight diff was created.

    Run:
        python weight_diff.py make_diff --path_raw <your_path_raw> --path_tuned <your_path_tuned> --path_diff <your_path_diff>
    """
    model_tuned: transformers.PreTrainedModel = llama_mem.LlamaForCausalLM.from_pretrained(
        path_tuned,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_tuned: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_tuned
    )
    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]", additional_special_tokens=["<landmark>"]),
        model=model_raw,
        tokenizer=tokenizer_raw,
    )



    state_dict_tuned = model_tuned.state_dict()
    state_dict_raw = model_raw.state_dict()
    with open(os.path.join(path_diff, "checksum_psum.txt"), "w") as f:
        f.write(str(sum(state_dict_tuned[key].sum().item() for key in state_dict_tuned)))

    for key in tqdm.tqdm(state_dict_tuned):
        state_dict_tuned[key].add_(-state_dict_raw[key])

    model_tuned.save_pretrained(path_diff)
    tokenizer_tuned.save_pretrained(path_diff)


@torch.inference_mode()
def recover(
    path_raw,
    path_diff,
    path_tuned: Optional[str] = None,
    device="cpu",
    test_inference=True,
    check_integrity_naively=True,
):
    """Recover the original weights from the released weight diff.

    This function is given for you to run.

    Things to do before running this:
        1. Convert Meta's released weights into huggingface format. Follow this guide:
            https://huggingface.co/docs/transformers/main/model_doc/llama
        2. Make sure you cloned the released weight diff into your local machine. The weight diff is located at:
            https://huggingface.co/tatsu-lab/alpaca-7b/tree/main
        3. Run this function with the correct paths. E.g.,
            python weight_diff.py recover --path_raw <path_to_step_1_dir> --path_diff <path_to_step_2_dir>

    Additional notes:
        - If things run too slowly, and you have an 80G GPU lying around, let GPU go brrr by setting `--device "cuda"`.
        - If you want to save the recovered weights, set `--path_tuned <your_path_tuned>`.
            Next time you can load the recovered weights directly from `<your_path_tuned>`.
    """
    model_raw: transformers.PreTrainedModel = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model_recovered: transformers.PreTrainedModel = llama_mem.LlamaForCausalLM.from_pretrained(
        path_diff,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )

    tokenizer_raw: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_raw
    )
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]", additional_special_tokens=["<landmark>"]),
        model=model_raw,
        tokenizer=tokenizer_raw,
    )
    tokenizer_recovered: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        path_diff
    )

    state_dict_recovered = model_recovered.state_dict()
    state_dict_raw = model_raw.state_dict()
    for key in tqdm.tqdm(state_dict_recovered):
        state_dict_recovered[key].add_(state_dict_raw[key])

    if check_integrity_naively:
        # This is not a rigorous, cryptographically strong integrity check :)
        allsum = sum(state_dict_recovered[key].sum() for key in state_dict_recovered)
        if os.path.exists(os.path.join(path_diff, "checksum_psum.txt")):
            with open(os.path.join(path_diff, "checksum_psum.txt")) as f:
                expected_sum = float(f.read())
        else:
            expected_sum = 49798.7656 # backward compatibility with the first released weights
        assert torch.allclose(
            allsum, torch.full_like(allsum, fill_value=expected_sum), atol=1e-2, rtol=0
        ), "Naive integrity check failed. This could imply that some of the checkpoint files are corrupted."

    if path_tuned is not None:
        model_recovered.save_pretrained(path_tuned)
        tokenizer_recovered.save_pretrained(path_tuned)

    return model_recovered, tokenizer_recovered


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
