from typing import List, Optional

import torch
from transformers import LlamaModel

from .text_embedder import TextEmbedder


class LLaMATextEmbedder(TextEmbedder):
    @property
    def embedding_dim(self):
        return self.llama.config.hidden_size
    
    def __init__(self, llama: LlamaModel) -> None:
        super().__init__()

        self.llama = llama

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
        return x
