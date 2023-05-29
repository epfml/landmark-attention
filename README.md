# Landmark Attention

This repository contains the implementation of landmark attention as described in our paper:

**Landmark Attention: Random-Access Infinite Context Length for Transformers**<br>
Amirkeivan Mohtashami, Martin Jaggi<br>
Preprint: https://arxiv.org/abs/2305.16300

## Training
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

## Inference
The code supports inference in various settings. To perform standard evaluation, disable cache and use the same chunk size (specified using `--mid_length` flag) as the evaluation length (specified by `--eval_seq_length`). Using landmarks is possible when using `mem_cache`. The script `eval_cmd_generator.py` can be used to generate a bash script containining commands to perform evaluations corresponding to Tables 1 and 2 of the paper. The path of the output models need to be updated inside the script.

## LLaMA fine-tuning
The code for fine-tuning LLaMA and testing the final model is available as a standalone project in the sub-directory "llama".  An example for running the fine tuning is:

```torchrun --nproc_per_node=8  train.py  \
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

## Naming
During the development of this project, we made the decision to update the names of certain components. However, as this decision was made later in the project timeline, you may encounter references to the old names within the code (e.g. `mem` instead of `landmark`). We are working to address this issue.
