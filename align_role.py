# align_role.py (FINAL: role-aware DPO on SecAlign preference data, prompt-compatible)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
os.environ["WANDB_DISABLED"] = "true"
import time
import json
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Tuple, List

import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import torch.nn.functional as F
import torch.nn as nn
import transformers
from transformers import Trainer
from datasets import load_dataset

from peft import get_peft_model, LoraConfig, TaskType

from config import PROMPT_FORMAT, DELIMITERS, IGNORE_INDEX
from struq import jload, format_with_other_delimiters
from train import ModelArguments, DataArguments, AttackArguments, TrainingArguments  # keep same CLI

from role_modeling import LlamaForCausalLMWithRole, MistralForCausalLMWithRole
from role_utils import ROLE_SYSTEM, ROLE_INSTRUCTION, ROLE_INPUT, ROLE_ASSISTANT


# -------------------------- small utils --------------------------
def assert_only_lora_and_role_trainable(model, allow_keywords=("lora_", "role_embeddings")):
    bad = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in allow_keywords):
            continue
        bad.append(n)

    if bad:
        msg = "\n".join(bad[:50])
        raise RuntimeError(
            "Unexpected trainable parameters (ref will drift and DPO becomes invalid).\n"
            "Only LoRA and role_embeddings should be trainable.\n"
            f"Found {len(bad)} unexpected trainables, first ones:\n{msg}"
        )

def disable_dropout_in_model(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0

def _model_tag_from_path(model_name_or_path: str) -> str:
    b = os.path.basename(model_name_or_path.rstrip("/"))
    if "Meta-Llama-3" in b or "Llama-3" in b:
        return "llama3"
    if "Mistral" in b:
        return "mistral"
    # 你的 llama-7b 就会落到这里
    return "llama"

def _rank() -> int:
    return int(os.environ.get("RANK", "0"))

def _jsondump(obj, path, indent=2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)
def report_dropout(model: nn.Module, prefix: str = ""):
    drops = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout):
            drops.append((name, float(m.p)))
    n_total = len(drops)
    n_active = sum(1 for _, p in drops if p > 0.0)

    print(f"{prefix}Dropout modules: total={n_total}, active(p>0)={n_active}")
    if n_active > 0:
        print(f"{prefix}Active Dropouts (p>0):")
        for name, p in drops:
            if p > 0.0:
                print(f"{prefix}  - {name}: p={p}")
### 自检
def _get_rank_safe() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))
def report_trainable(model: nn.Module, prefix: str = ""):
    trainable = []
    frozen = []
    for n, p in model.named_parameters():
        (trainable if p.requires_grad else frozen).append(n)

    print(f"{prefix}Trainable params: {len(trainable)} tensors; Frozen params: {len(frozen)} tensors")

    # 重点检查这些
    def _has(prefix_key: str, names: list[str]) -> bool:
        return any(prefix_key in x for x in names)

    print(f"{prefix}role_embeddings trainable? {_has('role_embeddings', trainable)}")
    print(f"{prefix}embed_tokens trainable?   {_has('embed_tokens', trainable)}")
    print(f"{prefix}lm_head trainable?       {_has('lm_head', trainable)}")

    # 如果你想看具体有哪些（可选）
    # print(f"{prefix}First 30 trainable names:")
    # for n in trainable[:30]:
    #     print(f"{prefix}  - {n}")
def report_tokenizer(tokenizer, prefix: str = ""):
    print(f"{prefix}pad_token={repr(tokenizer.pad_token)} id={tokenizer.pad_token_id}")
    print(f"{prefix}eos_token={repr(tokenizer.eos_token)} id={tokenizer.eos_token_id}")
    print(f"{prefix}pad_id == eos_id ? {tokenizer.pad_token_id == tokenizer.eos_token_id}")
### 自检

# def _pick_frontend_key(model_name_or_path: str) -> str:
#     """
#     Choose the prompt-format key used by PROMPT_FORMAT.
#     For role checkpoints like 'Meta-Llama-3-8B_RoleSpclSpclSpcl_None_xxx',
#     prefer base model name 'Meta-Llama-3-8B' if PROMPT_FORMAT has it.
#     """
#     name = os.path.basename(model_name_or_path.rstrip("/"))

#     # exact match
#     if name in PROMPT_FORMAT:
#         return name

#     parts = name.split("_")
#     if len(parts) >= 1 and parts[0] in PROMPT_FORMAT:
#         return parts[0]
#     if len(parts) >= 2 and parts[1] in PROMPT_FORMAT:
#         return parts[1]

#     # fallback: if key exists in DELIMITERS
#     if len(parts) >= 1 and parts[0] in DELIMITERS:
#         return parts[0]
#     if len(parts) >= 2 and parts[1] in DELIMITERS:
#         return parts[1]

#     # last resort
#     return parts[0]

def _pick_frontend_key(model_name_or_path: str) -> str:
    name = os.path.basename(model_name_or_path.rstrip("/"))
    parts = name.split("_")

    # 1) 最优先：如果是 role checkpoint，直接用它训练时的 role frontend
    for k in ["RoleSpclSpclSpcl"]:
        if k in PROMPT_FORMAT or k in DELIMITERS:
            if k in parts or k in name:
                return k

    # 2) 次优先：如果 parts 里有任何能命中的 PROMPT_FORMAT key，优先选“非 base-model”的那个
    for p in parts:
        if p in PROMPT_FORMAT and p not in ["llama-7b", "Mistral-7B-Instruct-v0.1"]:
            return p

    # 3) 再退回：base model key
    for p in parts:
        if p in PROMPT_FORMAT:
            return p
    for p in parts:
        if p in DELIMITERS:
            return p

    return parts[0]


# -------------------------- Original SecAlign dataset construction (distributed-safe) --------------------------
def generate_preference_data(clean_data_path, frontend_key, attack, alignment, tokenizer, model_name_or_path):
    """
    Keep original preference construction logic, but:
      - Use frontend_key (prompt format key) for naming and PROMPT_FORMAT lookup
      - Only rank0 writes the json (atomic write), others wait
    """

    tag = _model_tag_from_path(model_name_or_path)

    # preference_data_path = (
    #     clean_data_path.split('/')[0]
    #     + '/preference_' + frontend_key + '_' + alignment + '_' + attack + '_'
    #     + clean_data_path.split('/')[1]
    # )

    preference_data_path = (
        clean_data_path.split('/')[0]
        + f'/preference_{tag}_{frontend_key}_{alignment}_{attack}_'
        + clean_data_path.split('/')[1]
    )

    naive_proportion = 0.9
    do_generate_dataset = True

    def _prompt_signature_ok(p: str) -> bool:
        # Llama3 chat template signatures
        if "<|start_header_id|>system<|end_header_id|>" in p and "<|start_header_id|>user<|end_header_id|>" in p:
            return True
        # delimiter signatures if available
        if frontend_key in DELIMITERS:
            inst, data, resp = DELIMITERS[frontend_key]
            return (inst in p) and (data in p) and (resp in p)
        return True

    def _is_valid(pref: object) -> bool:
        if not isinstance(pref, list) or len(pref) == 0:
            return False
        ex0 = pref[0]
        if not isinstance(ex0, dict):
            return False
        if tokenizer.eos_token not in ex0.get("chosen", ""):
            return False
        if tokenizer.eos_token not in ex0.get("rejected", ""):
            return False
        if not _prompt_signature_ok(ex0.get("prompt", "")):
            return False
        return True

    if os.path.exists(preference_data_path):
        try:
            pref = jload(preference_data_path)
            if _is_valid(pref):
                do_generate_dataset = False
        except Exception:
            do_generate_dataset = True  # corrupted -> regenerate

    if do_generate_dataset:
        if _rank() == 0:
            clean_data = jload(clean_data_path)
            prompt_dict = PROMPT_FORMAT[frontend_key]
            preference_data = []

            if 'Completion' in attack:
                ref_inst_resp = {}
                for ref_sample in jload('data/alpaca_data.json'):
                    ref_inst_resp[ref_sample['instruction'].replace(tokenizer.pad_token, '')] = ref_sample['output']

            for i in range(len(clean_data)):
                if clean_data[i].get("input", "") == "":
                    continue

                current_sample = deepcopy(clean_data[i])
                if current_sample.get("input", "") and current_sample["input"][-1] != ".":
                    current_sample["input"] += "."

                injected_sample = np.random.choice(clean_data)
                injected_prompt = (
                    ('answer the following question. ' + injected_sample['instruction'] + ' ' + injected_sample['input'])
                    if injected_sample['instruction'][-1] == '?'
                    else (injected_sample['instruction'][0].lower() + injected_sample['instruction'][1:] + ' ' + injected_sample['input'])
                )

                if attack == 'Naive':
                    current_sample['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
                elif attack == 'NaiveCompletion':
                    if np.random.rand() < naive_proportion:
                        current_sample['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
                    else:
                        # keep original behavior
                        current_sample['input'] += (
                            '\n\n' + DELIMITERS['RoleSpclSpclSpcl'][2] + '\n'
                            + ref_inst_resp.get(current_sample['instruction'], current_sample['output']) + '\n\n'
                            + DELIMITERS['RoleSpclSpclSpcl'][0] + '\n'
                            + injected_prompt.capitalize()
                        )
                        if injected_sample.get('input', '') != '':
                            current_sample['input'] += '\n\n' + DELIMITERS['RoleSpclSpclSpcl'][1] + '\n' + injected_sample['input']
                        current_sample['input'] = format_with_other_delimiters(current_sample['input'], test=False)
                else:
                    raise NotImplementedError

                if alignment != "dpo":
                    raise NotImplementedError("This align_role.py final version only supports DPO.")

                preference_data.append({
                    "prompt": prompt_dict["prompt_input"].format_map(current_sample),
                    "chosen": current_sample["output"] + tokenizer.eos_token,
                    "rejected": injected_sample["output"] + tokenizer.eos_token,
                })

            # atomic write
            _jsondump(preference_data, preference_data_path)

        else:
            # wait for rank0 to finish writing a readable file
            while True:
                if os.path.exists(preference_data_path):
                    try:
                        pref = jload(preference_data_path)
                        if _is_valid(pref):
                            break
                    except Exception:
                        pass
                time.sleep(1)

    time.sleep(1)
    return load_dataset("json", data_files=preference_data_path, split="train")


# -------------------------- Role-aware tokenization (NO extra sys prompt injected) --------------------------
def _build_ids_roles_labels(
    prompt: str,
    completion: str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    frontend_key: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (input_ids, role_ids, labels, attention_mask) for prompt+completion.
    labels: IGNORE_INDEX on prompt tokens, token ids on completion tokens.
    role_ids: parsed from prompt template (llama3 headers / delimiter markers).
    """

    parts: List[Tuple[str, int]] = []

    # ---- Llama3-style chat template ----
    if "<|start_header_id|>system<|end_header_id|>" in prompt and "<|start_header_id|>user<|end_header_id|>" in prompt:
        sys_hdr = "<|start_header_id|>system<|end_header_id|>\n"
        usr_hdr = "<|start_header_id|>user<|end_header_id|>\n"
        ast_hdr = "<|start_header_id|>assistant<|end_header_id|>\n"
        eot = "<|eot_id|>"

        sys_start = prompt.find(sys_hdr)
        usr_start = prompt.find(usr_hdr)
        ast_start = prompt.rfind(ast_hdr)

        # fallback if weird
        if sys_start == -1 or usr_start == -1 or ast_start == -1:
            parts = [(prompt, ROLE_SYSTEM)]
        else:
            sys_end = prompt.find(eot, sys_start)
            usr_end = prompt.find(eot, usr_start)
            if sys_end == -1 or usr_end == -1:
                parts = [(prompt, ROLE_SYSTEM)]
            else:
                sys_end2 = sys_end + len(eot)
                usr_end2 = usr_end + len(eot)

                # slices that concatenate exactly to prompt
                prefix = prompt[:sys_start]
                sys_block = prompt[sys_start:sys_end2]
                mid1 = prompt[sys_end2:usr_start]
                usr_block = prompt[usr_start:usr_end2]
                mid2 = prompt[usr_end2:ast_start]
                ast_block = prompt[ast_start:]

                if prefix:
                    parts.append((prefix, ROLE_SYSTEM))
                parts.append((sys_block, ROLE_INSTRUCTION))  # system message = trusted instruction
                if mid1:
                    parts.append((mid1, ROLE_SYSTEM))
                parts.append((usr_block, ROLE_INPUT))         # user message = untrusted input
                if mid2:
                    parts.append((mid2, ROLE_SYSTEM))
                parts.append((ast_block, ROLE_ASSISTANT))     # assistant header

    else:
        # ---- Delimiter-based format (Instruction/Input/Response headers) ----
        if frontend_key in DELIMITERS:
            inst_delm, data_delm, resp_delm = DELIMITERS[frontend_key]
            i_pos = prompt.find(inst_delm)
            d_pos = prompt.find(data_delm)
            r_pos = prompt.rfind(resp_delm)

            if i_pos != -1 and d_pos != -1 and r_pos != -1 and (i_pos < d_pos < r_pos):
                prefix = prompt[:i_pos]
                inst_block = prompt[i_pos:d_pos]
                data_block = prompt[d_pos:r_pos]
                resp_block = prompt[r_pos:]

                if prefix:
                    parts.append((prefix, ROLE_SYSTEM))
                parts.append((inst_block, ROLE_INSTRUCTION))
                parts.append((data_block, ROLE_INPUT))
                parts.append((resp_block, ROLE_ASSISTANT))
            else:
                parts = [(prompt, ROLE_SYSTEM)]
        else:
            parts = [(prompt, ROLE_SYSTEM)]

    # tokenize prompt parts with roles
    prompt_ids: List[int] = []
    prompt_roles: List[int] = []
    for text, role in parts:
        ids = tokenizer(text, add_special_tokens=False).input_ids
        prompt_ids.extend(ids)
        prompt_roles.extend([role] * len(ids))

    # tokenize completion
    comp_ids = tokenizer(completion, add_special_tokens=False).input_ids
    comp_roles = [ROLE_ASSISTANT] * len(comp_ids)

    input_ids = prompt_ids + comp_ids
    role_ids = prompt_roles + comp_roles

    # labels: ignore prompt, supervise completion
    labels = ([IGNORE_INDEX] * len(prompt_ids)) + comp_ids

    # truncate to max_length (keep the tail to preserve completion)
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        role_ids = role_ids[-max_length:]
        labels = labels[-max_length:]

    input_ids_t = torch.tensor(input_ids, dtype=torch.long)
    role_ids_t = torch.tensor(role_ids, dtype=torch.long)
    labels_t = torch.tensor(labels, dtype=torch.long)
    attn_t = torch.ones_like(input_ids_t, dtype=torch.long)

    return input_ids_t, role_ids_t, labels_t, attn_t


# -------------------------- Role-aware DPO dataset & collator --------------------------
class RoleDPODataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length: int, frontend_key: str):
        self.ds = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.frontend_key = frontend_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ex = self.ds[idx]
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        c_ids, c_roles, c_labels, c_attn = _build_ids_roles_labels(
            prompt, chosen, self.tokenizer, self.max_length, self.frontend_key
        )
        r_ids, r_roles, r_labels, r_attn = _build_ids_roles_labels(
            prompt, rejected, self.tokenizer, self.max_length, self.frontend_key
        )

        return {
            "chosen_input_ids": c_ids,
            "chosen_role_ids": c_roles,
            "chosen_labels": c_labels,
            "chosen_attention_mask": c_attn,
            "rejected_input_ids": r_ids,
            "rejected_role_ids": r_roles,
            "rejected_labels": r_labels,
            "rejected_attention_mask": r_attn,
        }


@dataclass
class DataCollatorForRoleDPO:
    tokenizer: transformers.PreTrainedTokenizer

    def _pad_1d(self, seqs, pad_value: int) -> torch.Tensor:
        left_pad = (self.tokenizer.padding_side == "left")
        max_len = max(x.size(0) for x in seqs)
        out = seqs[0].new_full((len(seqs), max_len), pad_value)
        for i, x in enumerate(seqs):
            L = x.size(0)
            if left_pad:
                out[i, max_len - L:] = x
            else:
                out[i, :L] = x
        return out
    
    def _pad_1d_with_mask(self, seqs, pad_value: int):
        left_pad = (self.tokenizer.padding_side == "left")
        max_len = max(x.size(0) for x in seqs)
        out = seqs[0].new_full((len(seqs), max_len), pad_value)
        mask = seqs[0].new_zeros((len(seqs), max_len))
        for i, x in enumerate(seqs):
            L = x.size(0)
            if left_pad:
                out[i, max_len - L:] = x
                mask[i, max_len - L:] = 1
            else:
                out[i, :L] = x
                mask[i, :L] = 1
        return out, mask
    
    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        #c_input_ids = self._pad_1d([ex["chosen_input_ids"] for ex in instances], self.tokenizer.pad_token_id)
        c_role_ids  = self._pad_1d([ex["chosen_role_ids"]  for ex in instances], ROLE_SYSTEM)
        c_labels    = self._pad_1d([ex["chosen_labels"]    for ex in instances], IGNORE_INDEX)
        #c_attn      = c_input_ids.ne(self.tokenizer.pad_token_id)
        c_input_ids, c_attn = self._pad_1d_with_mask([ex["chosen_input_ids"] for ex in instances], self.tokenizer.pad_token_id)

        #r_input_ids = self._pad_1d([ex["rejected_input_ids"] for ex in instances], self.tokenizer.pad_token_id)
        r_role_ids  = self._pad_1d([ex["rejected_role_ids"]  for ex in instances], ROLE_SYSTEM)
        r_labels    = self._pad_1d([ex["rejected_labels"]    for ex in instances], IGNORE_INDEX)
        #r_attn      = r_input_ids.ne(self.tokenizer.pad_token_id)
        r_input_ids, r_attn = self._pad_1d_with_mask([ex["rejected_input_ids"] for ex in instances], self.tokenizer.pad_token_id)

        return {
            "chosen_input_ids": c_input_ids,
            "chosen_role_ids": c_role_ids,
            "chosen_labels": c_labels,
            "chosen_attention_mask": c_attn,
            "rejected_input_ids": r_input_ids,
            "rejected_role_ids": r_role_ids,
            "rejected_labels": r_labels,
            "rejected_attention_mask": r_attn,
        }


# -------------------------- Role-aware DPO Trainer (ref = base weights via disable_adapter) --------------------------
class RoleDPOTrainer(Trainer):
    """
    DPO loss:
      -log sigmoid( beta * [(log pi(y_w|x)-log pi(y_l|x)) - (log ref(y_w|x)-log ref(y_l|x))] )
    """

    def __init__(self, beta: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

        self._ref_role_embeds = None
        

    def _compute_logps(
        self,
        model: transformers.PreTrainedModel,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        role_ids: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            role_ids=role_ids,
        )
        logits = outputs.logits  # [B, L, V]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_attn   = attention_mask[:, 1:].contiguous()

        loss_mask = shift_labels.ne(IGNORE_INDEX) & shift_attn.bool()
        if not loss_mask.any():
            return torch.zeros(input_ids.size(0), device=input_ids.device)

        shift_labels_clipped = shift_labels.clone()
        shift_labels_clipped[~loss_mask] = 0

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels_clipped.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * loss_mask

        # return token_logp.sum(dim=-1)  # [B]
    
        n_tokens = loss_mask.sum(dim=-1).to(token_logp.dtype).clamp_min(1.0)
        return token_logp.sum(dim=-1) / n_tokens  # [B] average logp per supervised token


    def compute_loss(self, model, inputs, return_outputs=False):
        # chosen
        c_ids   = inputs["chosen_input_ids"]
        c_attn  = inputs["chosen_attention_mask"]
        c_roles = inputs["chosen_role_ids"]
        c_lab   = inputs["chosen_labels"]

        # rejected
        r_ids   = inputs["rejected_input_ids"]
        r_attn  = inputs["rejected_attention_mask"]
        r_roles = inputs["rejected_role_ids"]
        r_lab   = inputs["rejected_labels"]

        # policy logps (need grad!)
        logp_pi_c = self._compute_logps(model, c_ids, c_attn, c_roles, c_lab)
        logp_pi_r = self._compute_logps(model, r_ids, r_attn, r_roles, r_lab)

        # # ref logps: use base weights by disabling adapters (saves a full ref model in memory)
        # with torch.no_grad():
        #     if hasattr(model, "disable_adapter"):
        #         with model.disable_adapter():
        #             logp_ref_c = self._compute_logps(model, c_ids, c_attn, c_roles, c_lab)
        #             logp_ref_r = self._compute_logps(model, r_ids, r_attn, r_roles, r_lab)
        #     else:
        #         # fallback (not recommended): ref=policy
        #         logp_ref_c = logp_pi_c.detach()
        #         logp_ref_r = logp_pi_r.detach()
        with torch.no_grad():
            if hasattr(model, "disable_adapter"):
                # 找到当前（可能已 FSDP shard/flatten）的 role_embeddings.weight
                role_w = None
                for n, p in model.named_parameters():
                    if "role_embeddings" in n and n.endswith("weight"):
                        role_w = p
                        break
                if role_w is None:
                    raise RuntimeError("role_embeddings.weight not found at compute_loss time.")

                # ✅ 延迟抓取 ref 快照：第一次进入 compute_loss 时，权重还没被 optimizer 更新
                if self._ref_role_embeds is None:
                    self._ref_role_embeds = role_w.detach().clone()

                # ✅ 形状对不上就直接报（避免 silent 错）
                if self._ref_role_embeds.shape != role_w.shape:
                    raise RuntimeError(
                        f"ref snapshot shape {tuple(self._ref_role_embeds.shape)} != current role_w shape {tuple(role_w.shape)}; "
                        "likely due to FSDP wrapping timing."
                    )

                # swap -> ref snapshot
                cur = role_w.detach().clone()
                role_w.data.copy_(self._ref_role_embeds.to(device=role_w.device, dtype=role_w.dtype))
                try:
                    with model.disable_adapter():
                        logp_ref_c = self._compute_logps(model, c_ids, c_attn, c_roles, c_lab)
                        logp_ref_r = self._compute_logps(model, r_ids, r_attn, r_roles, r_lab)
                finally:
                    # restore -> policy role embedding
                    role_w.data.copy_(cur)
            else:
                logp_ref_c = logp_pi_c.detach()
                logp_ref_r = logp_pi_r.detach()

        logratios = (logp_pi_c - logp_pi_r) - (logp_ref_c - logp_ref_r)
        loss = -torch.log(torch.sigmoid(self.beta * logratios)).mean()

        return (loss, {}) if return_outputs else loss


# -------------------------- Main align() --------------------------
def align():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"npu:{local_rank}")
    torch_npu.npu.set_device(device)
    print(f"[Rank {os.environ.get('RANK', '0')}] Using device: {device}")

    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, AttackArguments))
    model_args, training_args, data_args, attack_args = parser.parse_args_into_dataclasses()

    if attack_args.alignment != "dpo":
        raise NotImplementedError("This align_role.py final version only supports --alignment dpo")

    # IMPORTANT: keep extra fields like role_ids
    training_args.remove_unused_columns = False

    # choose frontend_key for PROMPT_FORMAT + preference file naming
    frontend_key = _pick_frontend_key(model_args.model_name_or_path)
    print(f"[Rank {os.environ.get('RANK', '0')}] frontend_key for PROMPT_FORMAT = {frontend_key}")

    # ---- load config & tokenizer ----
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    # role config defaults
    if not hasattr(config, "num_roles"):
        config.num_roles = 4
    if not hasattr(config, "role_embedding_scale"):
        config.role_embedding_scale = 1.0

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side=model_args.padding_side,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # avoid tokenizer warnings; dataset will enforce training_args.model_max_length
    if hasattr(config, "max_position_embeddings"):
        tokenizer.model_max_length = int(config.max_position_embeddings)

    # ---- load role-aware model (policy) ----
    if config.model_type == "llama":
        policy = LlamaForCausalLMWithRole.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=True,
        )
    elif config.model_type == "mistral":
        policy = MistralForCausalLMWithRole.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            low_cpu_mem_usage=True,
        )
    else:
        raise NotImplementedError(f"role model not implemented for model_type={config.model_type}")

    policy.config.use_cache = False
    if getattr(model_args, "window_size", 0) > 0:
        policy.config.window = model_args.window_size

    # ---- Apply LoRA to policy only ----
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    policy = get_peft_model(policy, peft_config)

    for n, p in policy.named_parameters():
        if "role_embeddings" in n:
            p.requires_grad = True

    disable_dropout_in_model(policy)

    # 2) rank0 打印检查
    if _get_rank_safe() == 0:
        print(f"[Check] frontend_key={frontend_key}")
        report_tokenizer(tokenizer, prefix="[Check] ")
        report_dropout(policy, prefix="[Check] ")
        report_trainable(policy, prefix="[Check] ")
        assert_only_lora_and_role_trainable(policy)
        print("[Check] trainable params OK: only LoRA + role_embeddings")

        policy.print_trainable_parameters()

    # ---- Build preference dataset (prompt-compatible) ----
    hf_train = generate_preference_data(
        data_args.data_path,
        frontend_key,
        attack_args.attack,
        attack_args.alignment,
        tokenizer,
        model_args.model_name_or_path,
    )

    train_dataset = RoleDPODataset(
        hf_dataset=hf_train,
        tokenizer=tokenizer,
        max_length=training_args.model_max_length,
        frontend_key=frontend_key,
    )
    data_collator = DataCollatorForRoleDPO(tokenizer=tokenizer)

    trainer = RoleDPOTrainer(
        model=policy,
        beta=getattr(training_args, "beta", 0.1),
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    align()
