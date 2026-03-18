# role_modeling.py
# import torch
# import torch_npu
# from torch_npu.contrib import transfer_to_npu
# import torch.nn as nn
# from typing import Optional, List, Tuple

# from transformers.models.llama.modeling_llama import LlamaForCausalLM
# from transformers import LlamaConfig
# from role_utils import ROLE_SYSTEM, ROLE_ASSISTANT

# class LlamaForCausalLMWithRole(LlamaForCausalLM):
#     """
#     在 LLaMA 上加一层 role embedding：
#       inputs_embeds = token_embed + role_embed[role_ids] * role_embedding_scale
#     """

#     def __init__(self, config: LlamaConfig, **kwargs):
#         super().__init__(config)
#         self.num_roles = getattr(config, "num_roles", 4)
#         self.role_embedding_scale = getattr(config, "role_embedding_scale", 1.0)

#         self.role_embeddings = nn.Embedding(self.num_roles, config.hidden_size)
#         nn.init.zeros_(self.role_embeddings.weight)  # 不破坏原模型分布

#     @staticmethod
#     def _align_role_ids(role_ids: torch.LongTensor, ref_input_ids: torch.LongTensor):
#         """
#         把 role_ids 对齐到 ref_input_ids 的 (B, L)。

#         策略（尽量不违背你的“role 只作用于 prompt 结构”的核心思想）：
#         - role_ids 比 ref 长：取最后 L 个（适配 generate 只喂末尾 token 的情况）
#         - role_ids 比 ref 短：左侧补 ROLE_SYSTEM（缺失前缀不注入 role noise）
#         """
#         b, L = ref_input_ids.shape

#         if role_ids.dim() == 1:
#             role_ids = role_ids.unsqueeze(0)

#         # beam search / expand 之类可能让 batch 维度变化，这里做一个温和对齐
#         if role_ids.shape[0] != b:
#             if role_ids.shape[0] == 1:
#                 role_ids = role_ids.expand(b, -1)
#             else:
#                 # 兜底：repeat 到 b（一般不会发生）
#                 repeat = (b + role_ids.shape[0] - 1) // role_ids.shape[0]
#                 role_ids = role_ids.repeat(repeat, 1)[:b]

#         rL = role_ids.shape[1]
#         if rL > L:
#             role_ids = role_ids[:, -L:]
#         elif rL < L:
#             pad = role_ids.new_full((b, L - rL), ROLE_SYSTEM)
#             role_ids = torch.cat([pad, role_ids], dim=1)

#         return role_ids

#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
#         role_ids: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **kwargs,
#     ):
#         # 只在 input_ids 走 embedding 的分支里注入 role embedding
#         if input_ids is not None and role_ids is not None:
#             role_ids = self._align_role_ids(role_ids, input_ids)
#             role_ids = role_ids.clamp(min=0, max=self.num_roles - 1)

#             token_embeds = self.model.embed_tokens(input_ids)
#             role_embeds = self.role_embeddings(role_ids)

#             inputs_embeds = token_embeds + self.role_embedding_scale * role_embeds

#             if getattr(self.model, "embed_scale", None) is not None:
#                 inputs_embeds = inputs_embeds * self.model.embed_scale

#             input_ids = None  # 后面走 inputs_embeds

#         return super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             labels=labels,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             **kwargs,
#         )

#     def prepare_inputs_for_generation(
#         self,
#         input_ids,
#         past_key_values=None,
#         attention_mask=None,
#         role_ids=None,
#         **kwargs,
#     ):
#         # 让 HF 先决定“这一轮真正喂给 forward 的 input_ids 形状”
#         model_inputs = super().prepare_inputs_for_generation(
#             input_ids=input_ids,
#             past_key_values=past_key_values,
#             attention_mask=attention_mask,
#             **kwargs,
#         )

#         if role_ids is not None:
#             # 以 super() 产出的本轮真实 input_ids 作为对齐基准
#             ref = model_inputs.get("input_ids", None)
#             if ref is None:
#                 # 极少数路径可能走 inputs_embeds，这里用长度构造一个 ref
#                 if model_inputs.get("inputs_embeds", None) is not None:
#                     b = role_ids.shape[0] if role_ids.dim() > 1 else 1
#                     L = model_inputs["inputs_embeds"].shape[1]
#                     ref = torch.empty((b, L), dtype=torch.long, device=role_ids.device)
#                 else:
#                     ref = input_ids

#             model_inputs["role_ids"] = self._align_role_ids(role_ids, ref)

#         return model_inputs

#     def _update_model_kwargs_for_generation(
#         self,
#         outputs,
#         model_kwargs,
#         is_encoder_decoder: bool = False,
#         **extra_kwargs,
#     ):
#         model_kwargs = super()._update_model_kwargs_for_generation(
#             outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, **extra_kwargs
#         )

#         # 每步生成把 role_ids 补齐到当前长度，新 token = ROLE_ASSISTANT
#         role_ids = model_kwargs.get("role_ids", None)
#         attn_mask = model_kwargs.get("attention_mask", None)

#         if role_ids is not None and attn_mask is not None:
#             cur_len = attn_mask.shape[-1]
#             old_len = role_ids.shape[-1]
#             if cur_len > old_len:
#                 bsz = role_ids.shape[0]
#                 num_new = cur_len - old_len
#                 new_roles = role_ids.new_full((bsz, num_new), ROLE_ASSISTANT)
#                 model_kwargs["role_ids"] = torch.cat([role_ids, new_roles], dim=-1)

#         return model_kwargs
# role_modeling.py
import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from role_utils import ROLE_SYSTEM, ROLE_ASSISTANT

# LLaMA
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import LlamaConfig

# Mistral（不同 transformers 版本 import 路径可能不同，做个兜底）
try:
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    from transformers import MistralConfig
except Exception:
    from transformers import MistralForCausalLM, MistralConfig


class RoleEmbeddingMixin():
    """
    只负责 role embedding 的注入逻辑：
      inputs_embeds = token_embed + role_embed[role_ids] * scale
    """
    def _init_role(self, config):
        self.num_roles = getattr(config, "num_roles", 4)
        self.role_embedding_scale = getattr(config, "role_embedding_scale", 1.0)
        # additive: 纯加法角色嵌入（原始方案）
        # sinusoidal_rotary: 在加法基础上，再做一次 role-conditioned 旋转调制（sin/cos）
        self.role_encoding_type = getattr(config, "role_encoding_type", "additive")
        self.role_rotary_scale = getattr(config, "role_rotary_scale", 1.0)
        self.role_rotary_dim = getattr(config, "role_rotary_dim", None)
        self.role_embeddings = nn.Embedding(self.num_roles, config.hidden_size)
        nn.init.zeros_(self.role_embeddings.weight)

    def _resolve_rotary_dim(self, hidden_size: int) -> int:
        if self.role_rotary_dim is None:
            d = hidden_size
        else:
            d = min(int(self.role_rotary_dim), hidden_size)
        # 旋转按偶数维配对
        return max(0, d - (d % 2))

    def _apply_role_rotary(self, token_like: torch.Tensor, role_embeds: torch.Tensor) -> torch.Tensor:
        # token_like / role_embeds: (B, L, H)
        rotary_dim = self._resolve_rotary_dim(token_like.shape[-1])
        if rotary_dim <= 0:
            return token_like

        x = token_like[..., :rotary_dim]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]

        # 用 role embedding 的偶数维产生相位，范围压到 [-role_rotary_scale, role_rotary_scale]
        role_phase = role_embeds[..., :rotary_dim][..., 0::2]
        theta = torch.tanh(role_phase) * float(self.role_rotary_scale)
        theta = theta.to(dtype=token_like.dtype)
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        x_rot_even = x_even * cos_t - x_odd * sin_t
        x_rot_odd = x_even * sin_t + x_odd * cos_t

        x_rot = torch.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd

        if rotary_dim == token_like.shape[-1]:
            return x_rot

        return torch.cat([x_rot, token_like[..., rotary_dim:]], dim=-1)

    @staticmethod
    def _align_role_ids(role_ids: torch.LongTensor, ref_input_ids: torch.LongTensor):
        b, L = ref_input_ids.shape
        if role_ids.dim() == 1:
            role_ids = role_ids.unsqueeze(0)

        # batch 对齐（beam/expand 的温和处理）
        if role_ids.shape[0] != b:
            if role_ids.shape[0] == 1:
                role_ids = role_ids.expand(b, -1)
            else:
                repeat = (b + role_ids.shape[0] - 1) // role_ids.shape[0]
                role_ids = role_ids.repeat(repeat, 1)[:b]

        rL = role_ids.shape[1]
        if rL > L:
            role_ids = role_ids[:, -L:]
        elif rL < L:
            pad = role_ids.new_full((b, L - rL), ROLE_SYSTEM)
            role_ids = torch.cat([pad, role_ids], dim=1)

        return role_ids

    def _inject_role_embeds(self, *, input_ids, role_ids):
        role_ids = self._align_role_ids(role_ids, input_ids)
        role_ids = role_ids.clamp(min=0, max=self.num_roles - 1)

        token_embeds = self.model.embed_tokens(input_ids)
        role_embeds = self.role_embeddings(role_ids)
        inputs_embeds = token_embeds + self.role_embedding_scale * role_embeds

        if self.role_encoding_type == "sinusoidal_rotary":
            inputs_embeds = self._apply_role_rotary(inputs_embeds, role_embeds)

        if getattr(self.model, "embed_scale", None) is not None:
            inputs_embeds = inputs_embeds * self.model.embed_scale

        return inputs_embeds
    def _inject_role_into_inputs_embeds(self, *, inputs_embeds: torch.Tensor, role_ids: torch.LongTensor):
        # inputs_embeds: (B, L, H)
        b, L, _ = inputs_embeds.shape

        # 用 (B, L) 的“假 ref”来对齐 role_ids
        ref = torch.empty((b, L), dtype=torch.long, device=inputs_embeds.device)
        role_ids = self._align_role_ids(role_ids, ref).clamp(0, self.num_roles - 1)

        role_embeds = self.role_embeddings(role_ids).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        out = inputs_embeds + self.role_embedding_scale * role_embeds
        if self.role_encoding_type == "sinusoidal_rotary":
            out = self._apply_role_rotary(out, role_embeds)
        return out

    def _postprocess_generation_role_ids(self, outputs, model_kwargs):
        role_ids = model_kwargs.get("role_ids", None)
        attn_mask = model_kwargs.get("attention_mask", None)
        if role_ids is not None and attn_mask is not None:
            cur_len = attn_mask.shape[-1]
            old_len = role_ids.shape[-1]
            if cur_len > old_len:
                bsz = role_ids.shape[0]
                num_new = cur_len - old_len
                new_roles = role_ids.new_full((bsz, num_new), ROLE_ASSISTANT)
                model_kwargs["role_ids"] = torch.cat([role_ids, new_roles], dim=-1)
        return model_kwargs


class LlamaForCausalLMWithRole(LlamaForCausalLM, RoleEmbeddingMixin):
    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config)
        self._init_role(config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, role_ids=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):

        # if input_ids is not None and role_ids is not None and inputs_embeds is None:
        #     inputs_embeds = self._inject_role_embeds(input_ids=input_ids, role_ids=role_ids)
        #     input_ids = None

        if role_ids is not None:
            if input_ids is not None and inputs_embeds is None:
                # 常规训练/推理 input_ids 路径
                inputs_embeds = self._inject_role_embeds(input_ids=input_ids, role_ids=role_ids)
                input_ids = None
            elif inputs_embeds is not None:
                # gcg / advp-soft 经常走 inputs_embeds 路径
                inputs_embeds = self._inject_role_into_inputs_embeds(
                    inputs_embeds=inputs_embeds, role_ids=role_ids
                )


        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, role_ids=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs
        )

        if role_ids is not None:
            ref = model_inputs.get("input_ids", input_ids)
            model_inputs["role_ids"] = self._align_role_ids(role_ids, ref)

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder: bool = False, **extra_kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, **extra_kwargs
        )
        return self._postprocess_generation_role_ids(outputs, model_kwargs)


class MistralForCausalLMWithRole(MistralForCausalLM, RoleEmbeddingMixin):
    def __init__(self, config: MistralConfig, **kwargs):
        super().__init__(config)
        self._init_role(config)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, role_ids=None, inputs_embeds=None,
                labels=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):

        # if input_ids is not None and role_ids is not None and inputs_embeds is None:
        #     inputs_embeds = self._inject_role_embeds(input_ids=input_ids, role_ids=role_ids)
        #     input_ids = None

        if role_ids is not None:
            if input_ids is not None and inputs_embeds is None:
                # 常规训练/推理 input_ids 路径
                inputs_embeds = self._inject_role_embeds(input_ids=input_ids, role_ids=role_ids)
                input_ids = None
            elif inputs_embeds is not None:
                # gcg / advp-soft 经常走 inputs_embeds 路径
                inputs_embeds = self._inject_role_into_inputs_embeds(
                    inputs_embeds=inputs_embeds, role_ids=role_ids
                )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, role_ids=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs
        )

        if role_ids is not None:
            ref = model_inputs.get("input_ids", input_ids)
            model_inputs["role_ids"] = self._align_role_ids(role_ids, ref)

        return model_inputs

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder: bool = False, **extra_kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=is_encoder_decoder, **extra_kwargs
        )
        return self._postprocess_generation_role_ids(outputs, model_kwargs)
