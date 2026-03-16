import torch
from config import SPECIAL_DELM_TOKENS_W
from role_utils import ROLE_SYSTEM, ROLE_INSTRUCTION, ROLE_INPUT, ROLE_ASSISTANT

def is_role_model(model) -> bool:
    base = getattr(model, "base", model)
    cfg = getattr(base, "config", None)
    return cfg is not None and hasattr(cfg, "num_roles")

def build_role_ids_from_input_ids(input_ids: torch.Tensor, tokenizer, cache=None):
    # input_ids: (B,T) or (T,)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    B, T = input_ids.shape
    role_ids = torch.full_like(input_ids, ROLE_SYSTEM)

    inst_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[0])
    inpt_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[1])
    resp_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[2])
    rmark_id = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[3])
    rsep_id  = tokenizer.convert_tokens_to_ids(SPECIAL_DELM_TOKENS_W[4])
    special_ids = {inst_id, inpt_id, resp_id, rmark_id, rsep_id}

    # 用第一条找位置（suffix 长度固定时很好用）
    key = (T, input_ids.device)
    if cache is not None and key in cache:
        inst_idx, inpt_idx, resp_idx = cache[key]
    else:
        row = input_ids[0]
        def first_pos(tok_id):
            pos = (row == tok_id).nonzero(as_tuple=True)[0]
            return pos[0].item() if len(pos) else None
        def last_pos(tok_id):
            pos = (row == tok_id).nonzero(as_tuple=True)[0]
            return pos[-1].item() if len(pos) else None

        inst_idx = first_pos(inst_id)
        inpt_idx = first_pos(inpt_id)
        resp_idx = last_pos(resp_id)
        if cache is not None:
            cache[key] = (inst_idx, inpt_idx, resp_idx)

    # —— 赋段（默认更安全：找不到 delimiter 时宁愿当 INPUT，不要当 SYSTEM）——
    if inst_idx is None and inpt_idx is None and resp_idx is None:
        role_ids[:, :] = ROLE_INPUT
    else:
        if inst_idx is not None:
            end = T
            if inpt_idx is not None: end = min(end, inpt_idx)
            if resp_idx is not None: end = min(end, resp_idx)
            role_ids[:, inst_idx:end] = ROLE_INSTRUCTION

        if inpt_idx is not None:
            end = T
            if resp_idx is not None: end = min(end, resp_idx)
            role_ids[:, inpt_idx:end] = ROLE_INPUT

        if resp_idx is not None:
            role_ids[:, resp_idx:T] = ROLE_ASSISTANT

    for sid in special_ids:
        role_ids[input_ids == sid] = ROLE_SYSTEM

    return role_ids
