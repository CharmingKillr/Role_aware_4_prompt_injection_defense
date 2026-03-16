# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
import os
os.environ["WANDB_DISABLED"] = "true"
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import transformers
import trl
from struq_my import SupervisedDataset

from role_modeling import LlamaForCausalLMWithRole, MistralForCausalLMWithRole
from role_utils import ROLE_SYSTEM
from config import IGNORE_INDEX, DEFAULT_TOKENS, SPECIAL_DELM_TOKENS, TEXTUAL_DELM_TOKENS, SPECIAL_DELM_TOKENS_W, TEXTUAL_DELM_TOKENS_W

@dataclass
class ModelArguments: 
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    window_size: int = field(default=0, metadata={"help": "Window size for the sliding window attention."})
    padding_side: str = field(default="right", metadata={"help": "Padding side for tokenization."})

@dataclass
class DataArguments: data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class AttackArguments: 
    attack: str = field(default='TextTextText_None', metadata={"help": "Attack type for SFT/Align"})
    alignment: str = field(default='none', metadata={"help": "Alignment type."})

@dataclass
class TrainingArguments(trl.ORPOConfig):#transformers.TrainingArguments): # 
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    fsdp_state_dict_type: Optional[str] = field(
        default="FULL_STATE_DICT",
        metadata={"help": "Make FSDP save a consolidated full state dict on rank 0."}
    )
    downsample: Optional[bool] = field(default=True)
    lr_scale: Optional[bool] = field(default=True)
    beta: float = field(default=0.1)
    ref_model_init_kwargs: Optional[str] = field(default=None)
    precompute_ref_log_probs: Optional[bool] = field(default=False)
    desirable_weight: Optional[float] = field(default=1)
    undesirable_weight: Optional[float] = field(default=1)

    #
    num_roles: int = field(default=4, metadata={"help": "Number of role types for role embeddings."})
    role_embedding_scale: float = field(
        default=1.0,
        metadata={"help": "Scale factor for role embeddings added to token embeddings."},
    )

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning (role-aware)."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, role_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "role_ids")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        role_ids = torch.nn.utils.rnn.pad_sequence(
            role_ids, batch_first=True, padding_value=ROLE_SYSTEM
        )

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            role_ids=role_ids,
            attention_mask=attention_mask,
        )

def smart_tokenizer_and_embedding_resize(model, tokenizer):
    """
    Enlarge the vocabulary for the model and tokenizer, with new special tokens for StruQ delimiter tokens.
    The special delimiters are denoted by SPECIAL_DELM_TOKENS in config.py
    The textual delimiters (used for special delimiter initialization) are denoted by TEXTUAL_DELM_TOKENS in config.py
    The model/tokenizer is not deepcopied, so no need to return
    """
    assert len(SPECIAL_DELM_TOKENS_W) == len(TEXTUAL_DELM_TOKENS_W)
    num_new_tokens = tokenizer.add_special_tokens({
        'pad_token': DEFAULT_TOKENS['pad_token'],
        'additional_special_tokens': SPECIAL_DELM_TOKENS_W
        })
    model.resize_token_embeddings(len(tokenizer))
    delimiter_init_embed_index_from_text = [tokenizer.encode(v, add_special_tokens=False)[0] for v in TEXTUAL_DELM_TOKENS_W]
    assert num_new_tokens == len(SPECIAL_DELM_TOKENS_W) + 1

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    # Initialize the [PAD] token with the mean of all embeddings
    input_embeddings[-num_new_tokens] = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings[-num_new_tokens] = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

    # Initialize the 5 StruQ delimiters with the embeddings of the corresponding textual delimiters
    for i in range(len(SPECIAL_DELM_TOKENS_W)):
        index = -num_new_tokens+i+1
        print('Initialize special delimiter token', tokenizer.decode(len(tokenizer) + index), 'from the embedding of', tokenizer.decode(delimiter_init_embed_index_from_text[i]))
        input_embeddings[index] = input_embeddings[delimiter_init_embed_index_from_text[i]]
        output_embeddings[index] = output_embeddings[delimiter_init_embed_index_from_text[i]]


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, downsample=True) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, attack=data_args.attack, downsample=downsample)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, AttackArguments))
    model_args, data_args, training_args, attack_args = parser.parse_args_into_dataclasses()
    data_args.attack = attack_args.attack

    training_args.gradient_checkpointing = True

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"npu:{local_rank}")
    torch_npu.npu.set_device(device)
    print(f"[Rank {os.environ.get('RANK', '0')}] Using device: {device}")

    if 'Instruct' in model_args.model_name_or_path: assert 'SpclSpclSpcl' not in data_args.attack
    print('\n\n' + training_args.output_dir + '\n\n')

    # 用带 role embedding 的模型；否则保持原样
    
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.num_roles = training_args.num_roles
    config.role_embedding_scale = training_args.role_embedding_scale
    
    if config.model_type == "llama":
        model = LlamaForCausalLMWithRole.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir, config=config
        )
    elif config.model_type == "mistral":
        model = MistralForCausalLMWithRole.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir, config=config
        )
    else:
        raise NotImplementedError(f"role model not implemented for model_type={config.model_type}")

    # 
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if model_args.window_size > 0:
        model.config.window = model_args.window_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
    )

    if 'Instruct' not in model_args.model_name_or_path: smart_tokenizer_and_embedding_resize(model, tokenizer)
    else: tokenizer.pad_token = tokenizer.eos_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, downsample=training_args.downsample)
    if not training_args.downsample and training_args.lr_scale:
        training_args.learning_rate /= data_module["train_dataset"].data_copy_count

    trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    #
    trainer.accelerator.wait_for_everyone()
    if trainer.is_fsdp_enabled:
        fsdp_model = trainer.model                      # 这是 FSDP wrapper
        unwrapped = trainer.accelerator.unwrap_model(fsdp_model)  # 真正的 HF PreTrainedModel

        full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        # 重要：所有 rank 都要跑到这里参与 gather
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_cfg):
            state_dict = fsdp_model.state_dict()

        # 只有 rank0 落盘
        if trainer.args.should_save:
            unwrapped.save_pretrained(trainer.args.output_dir,
                                    state_dict=state_dict,
                                    safe_serialization=True)
            tokenizer.save_pretrained(trainer.args.output_dir)

        trainer.accelerator.wait_for_everyone()
    else:
        trainer.save_model(output_dir=training_args.output_dir)
    #trainer.save_state()
    #trainer.save_model(output_dir=training_args.output_dir)

    
if __name__ == "__main__":
    train()