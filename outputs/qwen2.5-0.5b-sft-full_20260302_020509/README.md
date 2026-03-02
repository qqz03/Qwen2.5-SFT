---
library_name: transformers
license: other
base_model: /home/qianqz/Model/Qwen2.5-0.5B
tags:
- llama-factory
- full
- generated_from_trainer
model-index:
- name: qwen2.5-0.5b-sft-full_20260302_020509
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen2.5-0.5b-sft-full_20260302_020509

This model is a fine-tuned version of [/home/qianqz/Model/Qwen2.5-0.5B](https://huggingface.co//home/qianqz/Model/Qwen2.5-0.5B) on the openorca_sft_300k dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 469
- num_epochs: 2.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- Transformers 5.2.0
- Pytorch 2.5.1
- Datasets 4.0.0
- Tokenizers 0.22.2
