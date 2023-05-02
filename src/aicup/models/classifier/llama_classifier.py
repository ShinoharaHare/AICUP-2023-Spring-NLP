from typing import List, Optional

import torch
from torch import nn
from transformers import LlamaModel

from .classifier import Classifier


class LLaMAClassifier(Classifier):
    def __init__(self, llama: LlamaModel) -> None:
        super().__init__()

        self.llama = llama
        self.classifier = nn.Linear(self.llama.config.hidden_size, 3)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
    ) -> torch.Tensor:
        x = self.llama.forward(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
        )
        x = x[0]
        x = x[:, -1]
        x = self.classifier(x)
        return x
