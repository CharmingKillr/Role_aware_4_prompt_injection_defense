from dataclasses import dataclass
from typing import Dict, Sequence, List

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from config import IGNORE_INDEX, DELIMITERS
from role_utils import ROLE_SYSTEM
from struq import jload
from role_utils import build_role_annotated_text  # 就是上面那段函数


class SecAlignRolePreferenceDataset(Dataset):
    """
    读取 SecAlign 生成的 preference_*.json（prompt/chosen/rejected），
    然后在此基础上生成 role_ids + labels。
    """
    def __init__(self, tokenizer, preference_json_path: str, max_length: int, frontend_delimiters: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inst_delm, self.data_delm, self.resp_delm = DELIMITERS[frontend_delimiters]

        data = jload(preference_json_path)
        # DPO/ORPO: must have prompt/chosen/rejected
        self.data = [x for x in data if "prompt" in x and "chosen" in x and "rejected" in x]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        ex = self.data[idx]
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        chosen_input_ids, chosen_role_ids, chosen_labels, chosen_attention_mask = build_role_annotated_text(
            prompt=prompt,
            completion=chosen,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            inst_delm=self.inst_delm,
            data_delm=self.data_delm,
            resp_delm=self.resp_delm,
            add_special_tokens=False,
        )

        rejected_input_ids, rejected_role_ids, rejected_labels, rejected_attention_mask = build_role_annotated_text(
            prompt=prompt,
            completion=rejected,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            inst_delm=self.inst_delm,
            data_delm=self.data_delm,
            resp_delm=self.resp_delm,
            add_special_tokens=False,
        )

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_role_ids": chosen_role_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_attention_mask,

            "rejected_input_ids": rejected_input_ids,
            "rejected_role_ids": rejected_role_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_attention_mask,
        }


@dataclass
class DataCollatorForSecAlignRolePreferenceDataset:
    tokenizer: any

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        def pad_1d(key: str, pad_value: int) -> torch.Tensor:
            return pad_sequence(
                [inst[key] for inst in instances],
                batch_first=True,
                padding_value=pad_value,
            )

        chosen_input_ids = pad_1d("chosen_input_ids", self.tokenizer.pad_token_id)
        chosen_role_ids  = pad_1d("chosen_role_ids", ROLE_SYSTEM)
        chosen_labels    = pad_1d("chosen_labels", IGNORE_INDEX)
        chosen_attention_mask = chosen_input_ids.ne(self.tokenizer.pad_token_id)

        rejected_input_ids = pad_1d("rejected_input_ids", self.tokenizer.pad_token_id)
        rejected_role_ids  = pad_1d("rejected_role_ids", ROLE_SYSTEM)
        rejected_labels    = pad_1d("rejected_labels", IGNORE_INDEX)
        rejected_attention_mask = rejected_input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_role_ids": chosen_role_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_attention_mask,

            "rejected_input_ids": rejected_input_ids,
            "rejected_role_ids": rejected_role_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_attention_mask,
        }
