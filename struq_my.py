# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torch_npu
from torch_npu.contrib import transfer_to_npu
import numpy as np
import re
from copy import deepcopy
from torch.utils.data import Dataset
import logging
import io, json
from config import PROMPT_FORMAT, IGNORE_ATTACK_SENTENCES, OTHER_DELM_FOR_TEST, OTHER_DELM_TOKENS, SPECIAL_DELM_TOKENS, DEFAULT_TOKENS, IGNORE_INDEX, TEXTUAL_DELM_TOKENS, DELIMITERS, SPECIAL_DELM_TOKENS_W
from role_utils import ROLE_SYSTEM, ROLE_INSTRUCTION, ROLE_INPUT, ROLE_ASSISTANT

import torch.distributed as dist
import os

def _is_rank0() -> bool:
    # 兼容未初始化 / 单卡
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # torchrun 环境下也常见
    r = os.environ.get("RANK", None)
    return (r is None) or (str(r) == "0")

def _first_pos(x: torch.Tensor, token_id: int):
    pos = (x == token_id).nonzero(as_tuple=True)[0]
    return pos[0].item() if len(pos) else None

def _last_pos(x: torch.Tensor, token_id: int):
    pos = (x == token_id).nonzero(as_tuple=True)[0]
    return pos[-1].item() if len(pos) else None

def _next_pos(x: torch.Tensor, token_id: int, start_idx: int):
    """first position of token_id in x with idx > start_idx"""
    if start_idx is None:
        return None
    pos = (x == token_id).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return None
    gt = pos[pos > start_idx]
    return gt[0].item() if len(gt) else None

def _prev_pos(x: torch.Tensor, token_id: int, end_idx: int):
    """last position of token_id in x with idx < end_idx"""
    if end_idx is None:
        return None
    pos = (x == token_id).nonzero(as_tuple=True)[0]
    if len(pos) == 0:
        return None
    lt = pos[pos < end_idx]
    return lt[-1].item() if len(lt) else None

def _truncate_prompt_keep_structure(
    src_ids_full: list,
    prompt_budget: int,
    *,
    inst_id: int,
    inpt_id: int,
    resp_id: int,
    rmark_id: int,
    rsep_id: int,
) -> list:
    """
    当 prompt 太长时，尽量“保结构 + 保 INPUT 尾部”：
    - 优先裁 INPUT 段（保留尾部），确保 <RMARK><RESP><RSEP> 框架仍在
    - 没有 INPT 时裁 instruction（保留头部更合理）
    - RESP 用 last_pos，避免攻击者在 input 里伪造 RESP 影响分段/截断
    """
    if len(src_ids_full) <= prompt_budget:
        return src_ids_full

    x = torch.tensor(src_ids_full, dtype=torch.long)

    inst_idx = _first_pos(x, inst_id)
    inpt_idx = _first_pos(x, inpt_id)
    resp_idx = _last_pos(x, resp_id)  # last RESP（关键）

    # 极端：找不到 RESP，只能硬截末尾
    if resp_idx is None:
        return src_ids_full[-prompt_budget:]

    # RESP 前最近的 RMARK（模板里一般存在）
    rmark_before_resp = _prev_pos(x, rmark_id, resp_idx)
    suffix_start = rmark_before_resp if rmark_before_resp is not None else resp_idx
    suffix_ids = src_ids_full[suffix_start:]  # 包含 RMARK + RESP + RSEP + ...

    # 选择裁剪正文段：优先 INPUT，否则 INSTRUCTION
    if inpt_idx is not None:
        inpt_rsep = _next_pos(x, rsep_id, inpt_idx)
        content_start = (inpt_rsep + 1) if inpt_rsep is not None else (inpt_idx + 1)
        content_end = suffix_start
        prefix_ids = src_ids_full[:content_start]
        content_ids = src_ids_full[content_start:content_end]
        keep_tail = True
    else:
        if inst_idx is None:
            return src_ids_full[-prompt_budget:]
        inst_rsep = _next_pos(x, rsep_id, inst_idx)
        content_start = (inst_rsep + 1) if inst_rsep is not None else (inst_idx + 1)
        content_end = suffix_start
        prefix_ids = src_ids_full[:content_start]
        content_ids = src_ids_full[content_start:content_end]
        keep_tail = False

    fixed = len(prefix_ids) + len(suffix_ids)
    if fixed >= prompt_budget:
        # prefix+suffix 都塞不下：优先保 suffix，剩余补 prefix 尾部
        keep_suffix = min(len(suffix_ids), prompt_budget)
        out_suffix = suffix_ids[:keep_suffix]
        remain = prompt_budget - len(out_suffix)
        if remain > 0:
            out_prefix = prefix_ids[-remain:]
            return out_prefix + out_suffix
        return out_suffix

    content_budget = prompt_budget - fixed
    if content_budget <= 0:
        return prefix_ids + suffix_ids

    content_keep = content_ids[-content_budget:] if keep_tail else content_ids[:content_budget]
    return prefix_ids + content_keep + suffix_ids


def _build_role_ids_for_example(
    ex_ids: torch.Tensor,
    prompt_len: int,
    *,
    inst_id: int,
    inpt_id: int,
    resp_id: int,
    rmark_id: int,
    rsep_id: int,
) -> torch.Tensor:
    """
    与 test_my.py 的 build_role_ids_from_input_ids 对齐：
      - 默认 SYSTEM
      - INST..INPT -> INSTRUCTION
      - INPT..RESP -> INPUT
      - RESP..end  -> ASSISTANT（注意用 last RESP）
      - delimiter token 强制 SYSTEM
    """
    roles = torch.full_like(ex_ids, ROLE_SYSTEM)

    # 只在 prompt 范围内找 delimiter（更稳，避免 target 里出现同 token）
    prompt_slice = ex_ids[:prompt_len]

    inst_idx = _first_pos(prompt_slice, inst_id)
    inpt_idx = _first_pos(prompt_slice, inpt_id)
    resp_idx = _last_pos(prompt_slice, resp_id)  # last RESP

    # 段赋值（先粗赋，最后再把 delimiter token 归 SYSTEM）
    if inst_idx is not None:
        end = prompt_len
        if inpt_idx is not None:
            end = min(end, inpt_idx)
        if resp_idx is not None:
            end = min(end, resp_idx)
        roles[inst_idx:end] = ROLE_INSTRUCTION

    if inpt_idx is not None:
        end = prompt_len
        if resp_idx is not None:
            end = min(end, resp_idx)
        roles[inpt_idx:end] = ROLE_INPUT

    if resp_idx is not None:
        roles[resp_idx:] = ROLE_ASSISTANT
    else:
        # 兜底：没找到 RESP，就至少把 target 视为 assistant
        roles[prompt_len:] = ROLE_ASSISTANT

    # delimiter token 强制 SYSTEM
    special_ids = {inst_id, inpt_id, resp_id, rmark_id, rsep_id}
    for sid in special_ids:
        roles[ex_ids == sid] = ROLE_SYSTEM

    return roles

def dump_preprocess_stats_v2(
    *,
    sources,
    targets,
    tokenizer,
    out_path_jsonl: str,
    max_records: int = 5000,
    min_supervised_tokens: int = 16,
    only_problematic: bool = False,
    store_full_text: bool = False,          # 只建议 problematic 时开
    text_preview_chars: int = 200,
):
    if not _is_rank0():
        return

    os.makedirs(os.path.dirname(out_path_jsonl), exist_ok=True)
    max_len = int(getattr(tokenizer, "model_max_length", 512))

    kept = 0
    bad_cnt = 0
    trunc_cnt = 0
    prompt_over_cnt = 0

    with open(out_path_jsonl, "w", encoding="utf-8") as f:
        for i, (src, tgt) in enumerate(zip(sources, targets)):

            # 1) FULL（不截断）长度：看数据原貌
            src_full = tokenizer(src, add_special_tokens=False, truncation=False)
            ex_full  = tokenizer(src + tgt, add_special_tokens=False, truncation=False)

            prompt_full = len(src_full["input_ids"])
            example_full = len(ex_full["input_ids"])
            truncated_by_maxlen = (example_full > max_len)
            prompt_over_maxlen = (prompt_full > max_len)

            # 2) TRAIN（截断）长度：完全模拟你 preprocess/_tokenize_fn 的训练现实
            src_train = tokenizer(src, add_special_tokens=False, truncation=True, max_length=max_len)
            ex_train  = tokenizer(src + tgt, add_special_tokens=False, truncation=True, max_length=max_len)

            prompt_train = len(src_train["input_ids"])
            example_train = len(ex_train["input_ids"])
            supervised_train = max(0, example_train - prompt_train)

            hit_max_length = (example_train == max_len)

            # 统计计数
            if supervised_train < min_supervised_tokens:
                bad_cnt += 1
            if truncated_by_maxlen:
                trunc_cnt += 1
            if prompt_over_maxlen:
                prompt_over_cnt += 1

            if only_problematic and (supervised_train >= min_supervised_tokens) and (not prompt_over_maxlen) and (not truncated_by_maxlen):
                continue

            rec = {
                "idx": i,
                "max_len": max_len,

                # FULL
                "prompt_tokens_full": prompt_full,
                "example_tokens_full": example_full,
                "truncated_by_maxlen": truncated_by_maxlen,
                "prompt_over_maxlen": prompt_over_maxlen,

                # TRAIN
                "prompt_tokens_train": prompt_train,
                "example_tokens_train": example_train,
                "supervised_tokens_train": supervised_train,
                "hit_max_length": hit_max_length,

                # text stats
                "source_chars": len(src),
                "target_chars": len(tgt),
                "source_words": len(src.split()),
                "target_words": len(tgt.split()),

                # preview（只是为了快速扫一眼）
                "source_preview": src[:text_preview_chars],
                "target_preview": tgt[:text_preview_chars],
            }

            if store_full_text:
                rec["source"] = src
                rec["target"] = tgt

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            kept += 1
            if kept >= max_records:
                break

    print(
        f"[dump_preprocess_stats_v2] wrote={kept} -> {out_path_jsonl}\n"
        f"  total={len(sources)} bad(<{min_supervised_tokens})={bad_cnt}\n"
        f"  truncated_by_maxlen(full_example>{max_len})={trunc_cnt}\n"
        f"  prompt_over_maxlen(full_prompt>{max_len})={prompt_over_cnt}"
    )

def to_jsonable(x):
    # x: Tensor or list[Tensor]
    if torch.is_tensor(x):
        return x.cpu().tolist()
    if isinstance(x, list):
        return [t.cpu().tolist() if torch.is_tensor(t) else t for t in x]
    return x


def format_with_other_delimiters(text, test=False):
    test_idx = - OTHER_DELM_FOR_TEST
    mark = np.random.choice(OTHER_DELM_TOKENS['mark'][test_idx:] if test else OTHER_DELM_TOKENS['mark'][:test_idx]) + ':'
    
    def sample_delm(delm_name):
        role_name = 'user' if (delm_name == 'inst' or delm_name == 'inpt') else 'asst'
        if test: 
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][test_idx:]) 
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][test_idx:])
        else:    
            role = np.random.choice(OTHER_DELM_TOKENS[role_name][:test_idx]) 
            delm = np.random.choice(OTHER_DELM_TOKENS[delm_name][:test_idx])
        
        p = np.random.rand()
        if p < 1/3: return (role + delm).upper()
        elif p < 2/3: return (role + delm).lower()
        else: return role + delm
    
    for delm in DELIMITERS.values():
        if '' in delm or ' ' in delm: continue
        text = text.replace(delm[0], mark.format(s=sample_delm('inst')))
        text = text.replace(delm[1], mark.format(s=sample_delm('inpt')))
        text = text.replace(delm[2], mark.format(s=sample_delm('resp')))
    return text

def generate_training_data(data_dicts, prompt_dict_name, attack, tokenizer):
    prompt_dict = PROMPT_FORMAT[prompt_dict_name]
    if attack == 'None':
        return [
            prompt_dict["prompt_input"].format_map(example) if example.get("input", "") != "" else prompt_dict["prompt_no_input"].format_map(example) for example in data_dicts
        ], [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]
    if attack == 'Completion':
        ref_inst_resp = {}
        for ref_sample in jload('data/alpaca_data.json'):  ref_inst_resp[ref_sample['instruction']] = ref_sample['output']
    sources = []

    for i in range(len(data_dicts)):
        # no anti-instruction tuning if there is no input
        if data_dicts[i].get("input", "") == "": sources.append(prompt_dict["prompt_no_input"].format_map(data_dicts[i]))
        else:
            injected_sample = deepcopy(np.random.choice(data_dicts)) 
            injected_sample['instruction'] = injected_sample['instruction']
            if injected_sample['instruction'][-1] == '?': 
                injected_prompt = 'answer the following question. ' + injected_sample['instruction'] + ' ' + injected_sample['input']
            else: 
                injected_prompt = injected_sample['instruction'][0].lower() + injected_sample['instruction'][1:] + ' ' + injected_sample['input']
            
            data_dicts_item = deepcopy(data_dicts[i])
            if data_dicts_item['input'][-1] != '.': data_dicts_item['input'] += '.'
            if attack == 'Naive':
                data_dicts_item['input'] += ' ' + injected_prompt[0].upper() + injected_prompt[1:]
            elif attack == 'Ignore':
                data_dicts_item['input'] += ' ' + np.random.choice(IGNORE_ATTACK_SENTENCES['train']).format(injected_prompt=injected_prompt)
            elif attack == 'Completion':
                data_dicts_item['input'] += '\n\n' + DELIMITERS['RoleSpclSpclSpcl'][2] + '\n' + ref_inst_resp.get(data_dicts_item['instruction'], data_dicts_item['output']) + '\n\n' + \
                                                     DELIMITERS['RoleSpclSpclSpcl'][0] + '\n' + injected_prompt.capitalize()
                if injected_sample['input'] != '':
                    data_dicts_item['input'] += '\n\n' + DELIMITERS['RoleSpclSpclSpcl'][1] + '\n' + injected_sample['input']
                data_dicts_item['input'] = format_with_other_delimiters(data_dicts_item['input'], test=False)
            else: raise NotImplementedError

            sources.append(prompt_dict["prompt_input"].format_map(data_dicts_item))
    return sources, [f"{example['output']}{tokenizer.eos_token}" for example in data_dicts]


def jload(f, mode="r"):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    jdict = json.load(f)
    f.close()
    return jdict

def jdump(obj, f, mode="w", indent=4, default=str):
    if not isinstance(f, io.IOBase): f = open(f, mode=mode)
    if isinstance(obj, (dict, list)): json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str): f.write(obj)
    else: raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()

def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list] 
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources,
    targets,
    tokenizer,
    *,
    min_supervised_tokens: int = 16,
    log_stats: bool = True,
):
    """
    输出：
      input_ids: [Tensor(T,)]
      labels:    [Tensor(T,)]  prompt 部分为 -100，只监督 target
      role_ids:  [Tensor(T,)]  与 test_my.py 规则一致

    核心变化：
      - 不再直接用 truncation=True 的 source_len 去 mask（会导致 prompt>max_len 时 supervised=0）
      - 对超长 prompt 做“结构化裁剪”（优先裁 INPUT，保留尾部），给 target 留监督 token
    """
    max_len = int(getattr(tokenizer, "model_max_length", 512))

    inst_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[0])  # <|INST|>
    inpt_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[1])  # <|INPT|>
    resp_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[2])  # <|RESP|>
    rmark_id = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[3])  # <|RMARK|>
    rsep_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[4])  # <|RSEP|>

    input_ids_list, labels_list, role_ids_list = [], [], []

    bad_short = 0
    bad_zero = 0
    prompt_truncated = 0
    ex_truncated = 0

    for src, tgt in zip(sources, targets):
        # 先不截断拿到 full ids（用于“聪明截断”）
        src_full = tokenizer(src, add_special_tokens=False, truncation=False, return_tensors=None)["input_ids"]
        tgt_full = tokenizer(tgt, add_special_tokens=False, truncation=False, return_tensors=None)["input_ids"]

        # 为 target 至少留 min_supervised_tokens（target 本身很短就按 target 长度）
        min_tgt_keep = min(min_supervised_tokens, len(tgt_full))
        prompt_budget = max_len - min_tgt_keep
        prompt_budget = max(1, prompt_budget)

        src_train = _truncate_prompt_keep_structure(
            src_full, prompt_budget,
            inst_id=inst_id, inpt_id=inpt_id, resp_id=resp_id, rmark_id=rmark_id, rsep_id=rsep_id
        )

        if len(src_train) < len(src_full):
            prompt_truncated += 1

        # target 实际能塞多少塞多少（保留开头，利于生成）
        tgt_keep = min(len(tgt_full), max(0, max_len - len(src_train)))
        tgt_train = tgt_full[:tgt_keep]

        # 拼接得到训练序列（再长也不会超过 max_len）
        ex_ids = torch.tensor(src_train + tgt_train, dtype=torch.long)
        if len(src_full) + len(tgt_full) > max_len:
            ex_truncated += 1

        # labels：prompt 全 mask，只监督 target
        labels = ex_ids.clone()
        labels[:len(src_train)] = IGNORE_INDEX

        supervised = int((labels != IGNORE_INDEX).sum().item())
        if supervised == 0:
            bad_zero += 1
        if supervised < min_supervised_tokens:
            bad_short += 1

        # role_ids：与 test_my.py 对齐
        roles = _build_role_ids_for_example(
            ex_ids, prompt_len=len(src_train),
            inst_id=inst_id, inpt_id=inpt_id, resp_id=resp_id, rmark_id=rmark_id, rsep_id=rsep_id
        )

        input_ids_list.append(ex_ids)
        labels_list.append(labels)
        role_ids_list.append(roles)

    if log_stats and _is_rank0():
        logging.warning(
            f"[preprocess] total={len(sources)} max_len={max_len} "
            f"prompt_truncated={prompt_truncated} ex_truncated(full>max)={ex_truncated} "
            f"supervised< {min_supervised_tokens}: {bad_short}  supervised==0: {bad_zero}"
        )

    return dict(input_ids=input_ids_list, labels=labels_list, role_ids=role_ids_list)

# def preprocess(sources, targets, tokenizer):
#     """
#     使用 RoleSpclSpclSpcl 风格的分隔符：
#       SPECIAL_DELM_TOKENS_W = ['<|INST|>', '<|INPT|>', '<|RESP|>', '<|RMARK|>',  '<|RSEP|>']
#     规则：
#       - 5 个特殊 token 一律 ROLE_SYSTEM
#       - INST 块和 INPT 块之间的正文是 ROLE_INSTRUCTION
#       - INPT 块和 RESP 块之间的正文是 ROLE_INPUT
#       - prompt 之后（answer 部分）是 ROLE_ASSISTANT
#     """
#     # 1. 和原版一样，拼接 prompt + output
#     examples = [s + t for s, t in zip(sources, targets)]
#     examples_tokenized, sources_tokenized = [
#         _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
#     ]
#     input_ids = examples_tokenized["input_ids"]
#     labels = deepcopy(input_ids)

#     bad = 0
#     # 2. 只对 answer 部分算 loss
#     for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
#         label[:source_len] = IGNORE_INDEX
#         sup = (label != IGNORE_INDEX).sum().item()
#         if sup < 16: bad += 1

#     logging.warning(f"Number of samples with less than 16 supervised tokens: {bad}, total samples: {len(labels)}")
    
#     # 3. 取出 5 个 special token 的 id
#     inst_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[0])  # <|INST|>
#     inpt_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[1])  # <|INPT|>
#     resp_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[2])  # <|RESP|>
#     rmark_id = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[3])  # <|RMARK|>
#     rsep_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[4])  # <|RSEP|>

#     special_ids = {inst_id, inpt_id, resp_id, rmark_id, rsep_id}

#     role_ids_list = []

#     for ex_ids, src_ids, src_len in zip(
#         input_ids,
#         sources_tokenized["input_ids"],
#         sources_tokenized["input_ids_lens"],
#     ):
#         # prompt 内默认都先设为 SYSTEM
#         src_roles = torch.full_like(src_ids, ROLE_SYSTEM)

#         sid = src_ids[:src_len]  # 只看有效 prompt 部分

#         inst_pos = (sid == inst_id).nonzero(as_tuple=True)[0]
#         inpt_pos = (sid == inpt_id).nonzero(as_tuple=True)[0]
#         resp_pos = (sid == resp_id).nonzero(as_tuple=True)[0]

#         def first_idx(pos):
#             return pos[0].item() if len(pos) > 0 else None

#         def last_idx(pos):
#             return pos[-1].item() if len(pos) > 0 else None
        

#         inst_idx = first_idx(inst_pos)
#         inpt_idx = first_idx(inpt_pos)
#         #resp_idx = first_idx(resp_pos)
#         resp_idx = last_idx(resp_pos)

#         # Instruction 段：INST 块 到 INPT / RESP / prompt 末尾
#         if inst_idx is not None:
#             start = inst_idx
#             end = src_len
#             if inpt_idx is not None:
#                 end = min(end, inpt_idx)
#             if resp_idx is not None:
#                 end = min(end, resp_idx)
#             src_roles[start:end] = ROLE_INSTRUCTION

#         # Input 段：INPT 块 到 RESP / prompt 末尾
#         if inpt_idx is not None:
#             start = inpt_idx
#             end = src_len
#             if resp_idx is not None:
#                 end = min(end, resp_idx)
#             src_roles[start:end] = ROLE_INPUT
        
#         if resp_idx is not None:
#             src_roles[resp_idx:src_len] = ROLE_ASSISTANT

#         # 4. 扩展到完整序列：后半段（answer）是 ASSISTANT
#         full_roles = torch.full_like(ex_ids, ROLE_SYSTEM)
#         full_roles[:src_len] = src_roles[:src_len]
#         full_roles[src_len:] = ROLE_ASSISTANT

#         # 5. 最后一刀：5 个特殊 token 统一强制成 ROLE_SYSTEM
#         for sid_val in special_ids:
#             full_roles[ex_ids == sid_val] = ROLE_SYSTEM

#         role_ids_list.append(full_roles)

#     return dict(input_ids=input_ids, labels=labels, role_ids=role_ids_list)



class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, attack, downsample=True):
        super(SupervisedDataset, self).__init__() 
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        prompt_dict_name, attacks = attack.split('_')
        source_clean, targets_clean = generate_training_data(list_data_dict, prompt_dict_name, 'None', tokenizer)
        
        if attacks == 'None': 
            sources, targets = source_clean, targets_clean
            self.data_copy_count = 1
        else:
            attacks = re.findall('[A-Z][^A-Z]*', attacks)
            sources = []; targets = []
            self.data_copy_count = len(attacks) + len(attacks) * downsample
            
            for a in attacks:
                source, target = generate_training_data(list_data_dict, prompt_dict_name, a, tokenizer)
                sources += source; targets += target
                if downsample: sources += source_clean; targets += targets_clean
                    
            # downsize data to original size with 50% clean data
            if downsample:
                sample_batch_id = np.random.choice(range(self.data_copy_count), len(source_clean))
                sample_id = [(x * len(sample_batch_id) + i) for i, x in enumerate(sample_batch_id)]
                sources = np.array(sources)[sample_id].tolist(); targets = np.array(targets)[sample_id].tolist()
            else:
                sources = np.array(sources).tolist(); targets = np.array(targets).tolist()

        # logging.warning("Dumping preprocess stats...")
        # dump_preprocess_stats_v2(
        #     sources=sources,
        #     targets=targets,
        #     tokenizer=tokenizer,
        #     out_path_jsonl="/workspace/SecAlign/data/mis_NIC_preprocess_stats_rank0.jsonl",
        #     min_supervised_tokens=16,
        #     max_records=20000,          # 够用了；想全量就设大一点
        #     only_problematic=True,
        #     store_full_text=True   # 只导出有问题的，文件更小
        # )

        logging.warning("Tokenizing inputs...")
        data_dict = preprocess(sources, targets, tokenizer)

        # if _is_rank0():
        #     save_path = "/workspace/SecAlign/data/mis_NIC_data_dict.json"
        #     json_dict = {k: to_jsonable(v) for k, v in data_dict.items()}
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         json.dump(json_dict, f, ensure_ascii=False, indent=2)

        # exit(0)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.role_ids = data_dict["role_ids"]  



    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i): 
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            role_ids=self.role_ids[i],
        )
