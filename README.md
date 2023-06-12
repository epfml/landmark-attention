# Landmark Attention

This repository contains the implementation of landmark attention as described in our paper:

**Landmark Attention: Random-Access Infinite Context Length for Transformers**<br>
Amirkeivan Mohtashami, Martin Jaggi<br>
Preprint: https://arxiv.org/abs/2305.16300

## Repository Structure

The repository contains three code bases under the following directories:

1. `lm_benchmark`: This directory contains the code used for performing language modeling over PG19 and arXiv Math datasets.
2. `llama_legacy`: This directory contains the code used to obtain the results of fine-tuning LLaMA as reported in the paper. The code in this directory is frozen to allow reproduction of the results. Thus, except when trying to exactly replicate our results, we suggest using the code under `llama` directory.
3. `llama`: This directory contains the current implementation of landmark attention. The directory includes both a high-level implementation and a Triton implementation of landmark attention combined with Flash Attention. As an example, the directory contains the code for applying the implementation to LLaMA models. 


Note: During the development of this project, we made the decision to update the names of certain components. However, as this decision was made later in the project timeline, you may encounter references to the old names within the code (e.g. `mem` instead of `landmark`). We are working to address this issue.


## Language Modeling Benchmarks
### Training
For training, the landmark tokens are added during data preparation. The following command is an example of training a model on PG19 with landmark tokens added every 50 tokens:
```
python main.py \
    --config_format rotary \
    --model landmark \
    --n_embd 1024 \
    --n_head 8 \
    --n_layer 12 \
    --batch_size 16 \
    --sequence_length 512 \
    --acc_steps 8 \
    --wandb_project memory-llm \
    --dataset pg19 \
    --iterations 240000 \
    --dropout 0.0 \
    --positional_encoder rotary \
    --softmax_func mem_opt \
    --mem_freq 50 \
    --wandb \
    --save_checkpoint_freq 20000
```

To run on multi-GPUs use torchrun (e.g. `torchrun --nproc_per_node=4`) and pass `--distributed_backend nccl` to `main.py` script. We suggest first running the script until the training starts on a single GPU before switching to multi-GPU settings. This is because the first node will have to perform the initialization of the data which can take a long time leading to a timeout on the synchronization in multi-GPU settings. However, once the initialization is performed once, the result is stored on the disk so the next runs will be quick.    

You will need to initialize the dataset before running the training script. For instructions, use the `prepare.py` script in the corresponding dataset folder located inside `data/`. 

### Inference
The code supports inference in various settings. To perform standard evaluation, disable cache and use the same chunk size (specified using `--mid_length` flag) as the evaluation length (specified by `--eval_seq_length`). Using landmarks is possible when using `mem_cache`. The script `eval_cmd_generator.py` can be used to generate a bash script containining commands to perform evaluations corresponding to Tables 1 and 2 of the paper. The path of the output models need to be updated inside the script.

## LLaMA fine-tuning
The code for fine-tuning LLaMA and testing the final model is available as a standalone project in the sub-directory "llama".  An example for running the fine tuning (from inside the sub-directory) is:

```
torchrun --nproc_per_node=8  train.py  \
    --model_name_or_path /llama_weights/7B_hf/ \
    --bf16 True \
    --output_dir /llama-redpajama-mem-15000-with-mem/  \
    --cache_dir /hf-cache/ \
    --num_train_epochs 1  \
    --per_device_train_batch_size 2     \
    --per_device_eval_batch_size 2     \
    --gradient_accumulation_steps 8     \
    --evaluation_strategy "no"     \
    --save_strategy "steps"     \
    --save_steps 2000     \
    --save_total_limit 2     \
    --learning_rate 2e-5     \
    --weight_decay 0.1     \
    --warmup_ratio 0.03     \
    --lr_scheduler_type "cosine"     \
    --logging_steps 1     \
    --fsdp "full_shard auto_wrap"     \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'     \
    --tf32 True \
    --max_steps 15000
```

In the above example, LLaMA wieghts (converted to huggingface format) should be in `/llama_weights/7B_hf/`.

### Fine-tuned Weights
We have released the weight diff between the original LLaMA 7B and the same model fine-tuned for 15000 steps on [RedPajama](https://github.com/togethercomputer/RedPajama-Data) dataset with landmark attention [here](https://huggingface.co/epfml/landmark-attention-llama7b-wdiff). You may use the `weight_diff.py` script to recover the weights:
```
python weight_diff.py recover --path_raw <path_to_original_llama7b_weights> --path_diff <path_to_weight_diff> --path_tuned <path_to_store_recovered_weights>
```
For an example of how to perform inference using landmarks, look at `run_test.py`.

### Triton Implementation

We have added a Triton implementation of the combination of our method and Flash Attention which significantly reduces memory usage and also increases performance. Using this implementation, we trained LLaMA 7B with 2048 context length (instead of 512). Also, adding landmark attention to any model can be done by applying the following changes: 

1. Adding landmark tokens to the input at regular intervals of block size.
2. (Optional) Creating a boolean mask of which tokens are landmarks. The mask can be passed to the landmark attention function to ensure the landmarks are placed correctly. This step can be skipped to obtain the highest speed.
3. Replacing `torch.nn.functional.scaled_dot_product_attention` with `fused_landmark_attention`. 

Note that the implemnetation relies on the latest version of Triton which causes a conflict with latest version of PyTorch. Therefore, a special `install_deps.sh` script is provided to install the dependencies.

Finally, note that the current implementation makes the following assumptions:

1. The implementation assumes the landmark blocks have the same size as blocks used for computing the attention in Flash Attention. This limits the maximum size of the block as the whole landmark block should fit into GPU's local memory. However, using bfloat16 it should be possible to use block sizes as large as 64 or 128 which should be enough for landmark blocks.
2. The implementation assumes the difference between number of keys and queries is a multiple of the block size. Therefore, normal attention must be applied in the auto-regressive part of the generation when the tokens are generated one by one. The implemnetation can still be used to go over the input before reaching the generation. 
Note that this is not a big limitation since when generating tokens one by one, the attention matrix has only a single row, limiting the benefits of Flash Attention. 
3. While the high level implementation allows the landmark tokens to be placed anywhere, the fused implementation assumes the landmark tokens are placed regularly at the end of each block. Since we always use this pattern at inference, this should not be noticed.
 
