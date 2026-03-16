# role_utils.py
from typing import List, Tuple
import torch
from config import IGNORE_INDEX

# 角色编号约定
ROLE_SYSTEM = 0
ROLE_INSTRUCTION = 1
ROLE_INPUT = 2
ROLE_ASSISTANT = 3

__all__ = [
    "ROLE_SYSTEM", "ROLE_INSTRUCTION", "ROLE_INPUT", "ROLE_ASSISTANT",
    "IGNORE_INDEX", "build_role_annotated_text",
]


def _find_subseq(haystack: List[int], needle: List[int], start: int = 0) -> int:
    """Return first index i>=start such that haystack[i:i+len(needle)]==needle, else -1."""
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    end = n - m
    for i in range(start, end + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


def build_role_annotated_text(
    prompt: str,
    completion: str,
    tokenizer,
    max_length: int,
    inst_delm: str,
    data_delm: str,
    resp_delm: str,
    add_special_tokens: bool = False,
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    返回 (input_ids, role_ids, labels, attention_mask)

    关键规则（满足你的创新点）：
    - prompt 内：模板文本 + system prompt + 分隔符 token => ROLE_SYSTEM
    - instruction 正文 token => ROLE_INSTRUCTION
    - input 正文 token => ROLE_INPUT
    - completion token => ROLE_ASSISTANT
    - labels: prompt 全部 IGNORE_INDEX，仅 completion 计算 logp（DPO 需要）
    """

    # --- tokenize full prompt / completion ---
    prompt_ids_full = tokenizer(prompt, add_special_tokens=add_special_tokens, truncation=False).input_ids
    comp_ids = tokenizer(completion, add_special_tokens=add_special_tokens, truncation=False).input_ids

    # --- tokenize delimiters (may be multi-token like "<|RMARK|> <|INST|><|RSEP|>") ---
    inst_ids = tokenizer(inst_delm, add_special_tokens=False, truncation=False).input_ids
    data_ids = tokenizer(data_delm, add_special_tokens=False, truncation=False).input_ids
    resp_ids = tokenizer(resp_delm, add_special_tokens=False, truncation=False).input_ids

    # --- locate delimiter spans in token space ---
    i_inst = _find_subseq(prompt_ids_full, inst_ids, start=0)
    i_data = _find_subseq(prompt_ids_full, data_ids, start=max(i_inst, 0) + len(inst_ids) if i_inst != -1 else 0)
    i_resp = _find_subseq(prompt_ids_full, resp_ids, start=max(i_data, 0) + len(data_ids) if i_data != -1 else 0)

    has_all = (i_inst != -1 and i_data != -1 and i_resp != -1 and i_inst < i_data < i_resp)

    # --- build role ids for FULL prompt ---
    role_prompt_full = [ROLE_SYSTEM] * len(prompt_ids_full)

    if has_all:
        inst_end = i_inst + len(inst_ids)
        data_end = i_data + len(data_ids)
        resp_end = i_resp + len(resp_ids)

        # delimiter token 本身仍是 ROLE_SYSTEM（默认已是）
        # instruction 正文：inst_end .. i_data
        for t in range(inst_end, i_data):
            role_prompt_full[t] = ROLE_INSTRUCTION
        # input 正文：data_end .. i_resp
        for t in range(data_end, i_resp):
            role_prompt_full[t] = ROLE_INPUT
        # resp delimiter 后面到 prompt 结束：仍保持 ROLE_SYSTEM（模板尾部换行等）

    # --- truncation: keep completion, truncate prompt from LEFT if needed ---
    total_len = len(prompt_ids_full) + len(comp_ids)
    if total_len > max_length:
        if len(comp_ids) >= max_length:
            # completion 自己都超长：只能截 completion（右截）
            comp_ids = comp_ids[:max_length]
            prompt_ids = []
            role_prompt = []
        else:
            keep_prompt = max_length - len(comp_ids)
            prompt_ids = prompt_ids_full[-keep_prompt:]
            role_prompt = role_prompt_full[-keep_prompt:]
    else:
        prompt_ids = prompt_ids_full
        role_prompt = role_prompt_full

    # --- concat ---
    input_ids = torch.tensor(prompt_ids + comp_ids, dtype=torch.long)
    role_ids = torch.tensor(role_prompt + [ROLE_ASSISTANT] * len(comp_ids), dtype=torch.long)

    attention_mask = torch.ones_like(input_ids)

    # labels: mask all prompt tokens
    labels = input_ids.clone()
    if len(prompt_ids) > 0:
        labels[:len(prompt_ids)] = IGNORE_INDEX

    return input_ids, role_ids, labels, attention_mask
