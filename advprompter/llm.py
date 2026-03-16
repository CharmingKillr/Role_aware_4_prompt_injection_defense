# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from contextlib import nullcontext, contextmanager

import torch
import torch.nn as nn

from sequence import Seq, MergedSeq, msg_to_seq
from utils import (
    ReturnStruct,
    autocast_decorator,
    compute_perplexity,
    get_nonascii_toks,
    llm_loader,
    loss_seqs,
)
from role_utils import ROLE_SYSTEM, ROLE_INSTRUCTION, ROLE_INPUT, ROLE_ASSISTANT
from config import SPECIAL_DELM_TOKENS_W


def build_role_ids_from_input_ids(input_ids: torch.Tensor, tokenizer):
    """
    input_ids: (B,T) or (T,)
    return: role_ids same shape
    """
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

    for b in range(B):
        row = input_ids[b]

        def first_pos(tok_id):
            pos = (row == tok_id).nonzero(as_tuple=True)[0]
            return pos[0].item() if len(pos) else None

        def last_pos(tok_id):
            pos = (row == tok_id).nonzero(as_tuple=True)[0]
            return pos[-1].item() if len(pos) else None

        inst_idx = first_pos(inst_id)
        inpt_idx = first_pos(inpt_id)
        resp_idx = last_pos(resp_id)

        if inst_idx is not None:
            end = T
            if inpt_idx is not None:
                end = min(end, inpt_idx)
            if resp_idx is not None:
                end = min(end, resp_idx)
            role_ids[b, inst_idx:end] = ROLE_INSTRUCTION

        if inpt_idx is not None:
            end = T
            if resp_idx is not None:
                end = min(end, resp_idx)
            role_ids[b, inpt_idx:end] = ROLE_INPUT

        if resp_idx is not None:
            role_ids[b, resp_idx:T] = ROLE_ASSISTANT

    # delimiter token 统一标回 SYSTEM
    for sid in special_ids:
        role_ids[input_ids == sid] = ROLE_SYSTEM

    for tid in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]:
        if tid is not None:
            role_ids[input_ids == tid] = ROLE_SYSTEM

    return role_ids


class LLM(nn.Module):
    """
    ✅ 新逻辑：
      - 默认不使用 role_ids（优化阶段不改调用方就会是“无 role”）
      - 只有在调用时显式 use_role_ids=True 才注入 role_ids（用于验证阶段）
    """
    def __init__(self, params, verbose=False) -> None:
        super().__init__()
        self.params = params
        self.verbose = verbose

        self.model, self.tokenizer, self.embedding_matrix = llm_loader(
            llm_params=params.llm_params, verbose=verbose
        )

        self.is_role_model = hasattr(getattr(self.model, "config", None), "num_roles")
        import inspect

        
        try:
            has_role_ids = ("role_ids" in inspect.signature(self.model.forward).parameters)
        except (ValueError, TypeError):
            # 某些 wrapped/torchscript 的 forward 可能拿不到 signature
            has_role_ids = False

        print(f"[TargetLLM class] {type(self.model)}")
        print(f"[Has role_ids arg] {has_role_ids}")
        print(f"[is_role_model flag] {self.is_role_model}")

        # ✅ 默认不开 role（你要“优化阶段不用”）
        # 如果你想全局强制开，可 export ROLE_IDS_DEFAULT=1
        self._use_role_ids_default = bool(int(os.getenv("ROLE_IDS_DEFAULT", "0")))
        self._use_role_ids_runtime = self._use_role_ids_default

        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.params.llm_params.device
        if self.params.allow_non_ascii:
            self.disallowed_ids = None
        else:
            self.disallowed_ids = get_nonascii_toks(self.tokenizer, device=self.device)

    # ---------- role gating ----------
    @contextmanager
    def role_guard(self, enabled: bool = True):
        """
        用法：
            with target_llm.role_guard(True):
                ...  # 这里面的 forward/generate 都会注入 role_ids
        """
        old = self._use_role_ids_runtime
        self._use_role_ids_runtime = enabled
        try:
            yield
        finally:
            self._use_role_ids_runtime = old

    def _role_enabled(self, use_role_ids):
        """
        use_role_ids:
          - None: 用 runtime 默认（通常 False）
          - True/False: 显式覆盖
        """
        if not self.is_role_model:
            return False
        if use_role_ids is None:
            return bool(self._use_role_ids_runtime)
        return bool(use_role_ids)

    def _maybe_role_ids_from_ids(self, ids_2d: torch.Tensor, use_role_ids):
        if not self._role_enabled(use_role_ids):
            return None
        role_ids = build_role_ids_from_input_ids(ids_2d, self.tokenizer).to(ids_2d.device)
        if os.getenv("ROLE_DEBUG", "0") == "1":
            u, c = torch.unique(role_ids[0], return_counts=True)
            print("[ROLE_DEBUG] uniq:", u.tolist(), "cnt:", [int(x) for x in c.tolist()])
        return role_ids
    # -------------------------------

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path, save_embedding_layers=True)

    # def model_forward(self, query_seq, use_basemodel=False, use_role_ids=None):
    #     # reorder such that all masked tokens are on the left
    #     mask = query_seq.mask
    #     sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)

    #     sorted_ids = None  # 防止 debug 分支未定义
    #     ids = getattr(query_seq, "ids", None)

    #     with self.model.disable_adapter() if use_basemodel else nullcontext():
    #         if query_seq.is_hard:
    #             ids = query_seq.ids
    #             sorted_ids = ids.gather(1, indices)

    #             role_ids = self._maybe_role_ids_from_ids(sorted_ids, use_role_ids)

    #             shifted_sorted_pred_logits = self.model(
    #                 input_ids=sorted_ids,
    #                 attention_mask=sorted_mask,
    #                 **({"role_ids": role_ids} if role_ids is not None else {}),
    #             ).logits
    #         else:
    #             embeds = query_seq.get_embed(self.embedding_matrix)
    #             indices_extended = indices[:, :, None].repeat(1, 1, embeds.shape[-1])
    #             sorted_embeds = embeds.gather(1, indices_extended)

    #             # soft 路径：尽量用 ids 推 role（ids 可能为空）
    #             role_ids = None
    #             if self._role_enabled(use_role_ids) and ids is not None:
    #                 sorted_ids = ids.gather(1, indices)
    #                 role_ids = self._maybe_role_ids_from_ids(sorted_ids, use_role_ids)

    #             shifted_sorted_pred_logits = self.model(
    #                 inputs_embeds=sorted_embeds,
    #                 attention_mask=sorted_mask,
    #                 **({"role_ids": role_ids} if role_ids is not None else {}),
    #             ).logits

    #     # reverse the sort to get the original order (also account for the shift)
    #     dummy_pred_logits = torch.zeros_like(shifted_sorted_pred_logits[:, :1, :])
    #     sorted_pred_logits = torch.cat(
    #         [dummy_pred_logits, shifted_sorted_pred_logits[:, :-1, :]], dim=1
    #     )
    #     reverse_indices = indices.argsort(dim=1)
    #     reverse_indices_extended = reverse_indices[:, :, None].repeat(
    #         1, 1, sorted_pred_logits.shape[-1]
    #     )
    #     shifted_pred_logits = sorted_pred_logits.gather(1, reverse_indices_extended)
    #     pred_logits = torch.cat(
    #         [shifted_pred_logits[:, 1:, :], shifted_sorted_pred_logits[:, -1:, :]],
    #         dim=1,
    #     )

    #     if self.disallowed_ids is not None:
    #         pred_logits[:, :, self.disallowed_ids] = -1e10

    #     if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
    #         for i in range(pred_logits.shape[0]):
    #             if torch.isnan(pred_logits[i]).any():
    #                 print(i, "-th logits..........", pred_logits[i])
    #                 print("shifted_sorted_pred_logits", shifted_sorted_pred_logits[i])
    #                 if ids is not None:
    #                     print("ids........", ids[i])
    #                 print("sorted_masks.......", sorted_mask[i])
    #                 if sorted_ids is not None:
    #                     print("sorted_ids", sorted_ids[i])
    #         raise RuntimeError(f"NaN in pred_logits: {pred_logits}")

    #     new_mask = torch.ones_like(mask)
    #     new_mask[:, :-1] = mask[:, 1:]
    #     seq = Seq(
    #         logits=pred_logits,
    #         mask=new_mask,
    #         tokenizer=self.tokenizer,
    #         device=self.device,
    #     )
    #     return seq

    def model_forward(self, query_seq, use_basemodel=False, use_role_ids=None):
        # ✅ 不排序：保持原始顺序（通常右 padding），避免左 padding 引发 NaN
        mask = query_seq.mask.long()   # (B,T) 1=valid,0=pad
        ids = getattr(query_seq, "ids", None)

        # 可选：提前抓到极端坏样本
        if (mask.sum(dim=1) == 0).any():
            raise RuntimeError("Found empty attention_mask row (all pad).")

        with self.model.disable_adapter() if use_basemodel else nullcontext():
            if query_seq.is_hard:
                ids = query_seq.ids  # (B,T)
                role_ids = self._maybe_role_ids_from_ids(ids, use_role_ids)
                logits = self.model(
                    input_ids=ids,
                    attention_mask=mask,
                    **({"role_ids": role_ids} if role_ids is not None else {}),
                ).logits
            else:
                embeds = query_seq.get_embed(self.embedding_matrix)  # (B,T,D)
                role_ids = None
                if self._role_enabled(use_role_ids) and ids is not None:
                    role_ids = self._maybe_role_ids_from_ids(ids, use_role_ids)
                logits = self.model(
                    inputs_embeds=embeds,
                    attention_mask=mask,
                    **({"role_ids": role_ids} if role_ids is not None else {}),
                ).logits

        # ✅ 标准 teacher-forcing shift：pred[t] = logits[t-1]
        dummy_pred_logits = torch.zeros_like(logits[:, :1, :])
        pred_logits = torch.cat([dummy_pred_logits, logits[:, :-1, :]], dim=1)

        if self.disallowed_ids is not None:
            pred_logits[:, :, self.disallowed_ids] = -1e10

        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            for i in range(pred_logits.shape[0]):
                if torch.isnan(pred_logits[i]).any() or torch.isinf(pred_logits[i]).any():
                    print(i, "-th pred_logits..........", pred_logits[i])
                    if ids is not None:
                        print("ids........", ids[i])
                    print("mask.......", mask[i])
            raise RuntimeError(f"NaN/Inf in pred_logits: {pred_logits}")

        # 你的 mask shift 逻辑保持一致
        new_mask = torch.ones_like(mask)
        new_mask[:, :-1] = mask[:, 1:]

        return Seq(
            logits=pred_logits,
            mask=new_mask,
            tokenizer=self.tokenizer,
            device=self.device,
        )


    @autocast_decorator
    def compute_pred_loss_teacher_forced(self, loss_params, label=None, use_role_ids=None, **kwargs):
        gen_seqs = self.generate_teacher_forced(use_role_ids=use_role_ids, **kwargs)
        if label is None:
            label = gen_seqs.response_teacher
        loss_return = loss_seqs(gen_seqs.response_dist, label, **loss_params)

        pred_loss_return = ReturnStruct(
            loss=loss_return.loss,
            loss_masked=loss_return.loss_masked,
            loss_batch=loss_return.loss_batch,
            query=gen_seqs.query,
            response_teacher=gen_seqs.response_teacher,
            response_dist=gen_seqs.response_dist,
            label=label,
            perplexity=gen_seqs.perplexity,
            perplexity_per_token_masked=gen_seqs.perplexity_per_token_masked,
        )
        return pred_loss_return

    @autocast_decorator
    def generate_teacher_forced(
        self, key, detach_query=False, use_basemodel=False, use_role_ids=None, **context
    ):
        query_seq, response_teacher_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        )
        assert not response_teacher_seq.is_empty
        full_query_seq = MergedSeq([query_seq, response_teacher_seq])
        if detach_query:
            full_query_seq = full_query_seq.clone().detach()

        pred_full_query_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel, use_role_ids=use_role_ids
        )
        response_dist_seq = pred_full_query_seq[
            :, -response_teacher_seq.seq_len - 1 : -1
        ]
        perplexity, perplexity_per_token_masked = compute_perplexity(
            id_seq=response_teacher_seq, likelihood_seq=response_dist_seq
        )

        return_seqs = ReturnStruct(
            query=query_seq,
            response_teacher=response_teacher_seq,
            response_dist=response_dist_seq,
            perplexity=perplexity,
            perplexity_per_token_masked=perplexity_per_token_masked,
        )
        return return_seqs

    def get_next_token(self, key, use_basemodel=False, use_role_ids=None, **context):
        query_seq, key_seq = self.prepare_prompt(
            context, up_to_key=key, return_key_seq=True
        )
        full_query_seq = MergedSeq([query_seq, key_seq])

        pred_dist_seq = self.model_forward(
            query_seq=full_query_seq, use_basemodel=use_basemodel, use_role_ids=use_role_ids
        )
        next_dist_seq = pred_dist_seq[:, -1:]

        return_seqs = ReturnStruct(query=full_query_seq, response_dist=next_dist_seq)
        return return_seqs

    # def generate_autoregressive(
    #     self, key, use_basemodel=False, max_new_tokens=None, use_role_ids=None, **context
    # ):
    #     query_seq = self.prepare_prompt(context, up_to_key=key)

    #     mask = query_seq.mask
    #     ids = query_seq.ids
    #     sorted_mask, indices = torch.sort(mask.long(), dim=1, stable=True)
    #     sorted_ids = ids.gather(1, indices)

    #     generation_config = self.model.generation_config
    #     if self.disallowed_ids is not None:
    #         generation_config.suppress_tokens = self.disallowed_ids.tolist()
    #     generation_config.renormalize_logits = True

    #     if max_new_tokens is None:
    #         max_new_tokens = self.params.gen_params.max_new_tokens

    #     gen_params = dict(self.params.gen_params)
    #     gen_params["max_new_tokens"] = max_new_tokens

    #     with self.model.disable_adapter() if use_basemodel else nullcontext():
    #         role_ids = self._maybe_role_ids_from_ids(sorted_ids, use_role_ids)

    #         output = self.model.generate(
    #             input_ids=sorted_ids,
    #             attention_mask=sorted_mask,
    #             **({"role_ids": role_ids} if role_ids is not None else {}),
    #             generation_config=generation_config,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             return_dict_in_generate=True,
    #             **gen_params,
    #         )

    #     output_ids = output.sequences[:, ids.shape[1] :]

    #     response_sample_seq = Seq(
    #         ids=output_ids, tokenizer=self.tokenizer, device=self.device
    #     )

    #     return_seqs = ReturnStruct(
    #         query=query_seq,
    #         response_sample=response_sample_seq,
    #     )
    #     return return_seqs
    def generate_autoregressive(
        self, key, use_basemodel=False, max_new_tokens=None, use_role_ids=None, **context
    ):
        query_seq = self.prepare_prompt(context, up_to_key=key)

        mask = query_seq.mask.long()
        ids = query_seq.ids

        generation_config = self.model.generation_config
        if self.disallowed_ids is not None:
            generation_config.suppress_tokens = self.disallowed_ids.tolist()
        generation_config.renormalize_logits = True

        if max_new_tokens is None:
            max_new_tokens = self.params.gen_params.max_new_tokens

        gen_params = dict(self.params.gen_params)
        gen_params["max_new_tokens"] = max_new_tokens

        with self.model.disable_adapter() if use_basemodel else nullcontext():
            role_ids = self._maybe_role_ids_from_ids(ids, use_role_ids)
            output = self.model.generate(
                input_ids=ids,
                attention_mask=mask,
                **({"role_ids": role_ids} if role_ids is not None else {}),
                generation_config=generation_config,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                **gen_params,
            )

        output_ids = output.sequences[:, ids.shape[1]:]
        response_sample_seq = Seq(ids=output_ids, tokenizer=self.tokenizer, device=self.device)

        return ReturnStruct(query=query_seq, response_sample=response_sample_seq)


    def prepare_prompt(self, context, up_to_key=None, return_key_seq=False):
        seqs = []
        for msg_dct in self.params.prompt_manager.prompt_template:
            if (
                up_to_key is not None
                and up_to_key == msg_dct.key
                and not return_key_seq
            ):
                break
            seq = msg_to_seq(
                msg=msg_dct.msg,
                tokenizer=self.tokenizer,
                device=self.device,
                context=context,
            )
            if up_to_key is not None and up_to_key == msg_dct.key and return_key_seq:
                break
            seqs.append(seq)

        merged_prompt_seq = MergedSeq(seqs)
        if return_key_seq:
            return merged_prompt_seq, seq
        else:
            return merged_prompt_seq
