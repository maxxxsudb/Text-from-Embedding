# src/model.py
import torch, torch.nn as nn
from torch import Tensor
from transformers import BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional

class KeyValueProjector(nn.Module):
    """
    SBERT-вектор (B,1024) → память (B, k, d_model)
    по умолчанию k = 1.
    """
    def __init__(self, in_dim=1024, d_model=1024, k=1, hidden=4096):
        super().__init__()
        self.k, self.d_model = k, d_model
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, k * d_model)
        )

    def forward(self, x: Tensor) -> Tensor:           # x: [B, in_dim]
        b = x.size(0)
        out = self.proj(x).view(b, self.k, self.d_model)
        return out                                    # [B, k, d_model]

class Sbert2Text(nn.Module):
    """
    Мини-обёртка: projector + замороженный BART-декодер.
    """
    def __init__(self,
                 bart_name: str = "facebook/bart-large-cnn",
                 in_dim: int = 1024,
                 k: int = 1):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)
        self.bart.eval()
        for p in self.bart.parameters():
            p.requires_grad_(False)                  # freeze всё

        cfg: BartConfig = self.bart.config
        self.projector = KeyValueProjector(in_dim, cfg.d_model, k)

        # чтобы generate() без энкодера не ругался
        self.register_buffer("dummy_input_ids",
                             torch.tensor([[cfg.bos_token_id]]),
                             persistent=False)

    def forward(self,
                sbert_vec: Tensor,                   # [B,1024]
                labels: Optional[Tensor] = None):    # [B,T]
        """
        Teacher-forcing во время обучения.
        """
        memory = self.projector(sbert_vec)           # [B,k,d_model]
        enc_out = BaseModelOutput(last_hidden_state=memory)

        outs = self.bart(
            encoder_outputs = enc_out,
            decoder_input_ids = None,               # BART сам сдвинет
            labels = labels)
        return outs

    @torch.inference_mode()
    def generate(self, sbert_vec: Tensor, **gen_kwargs):
        memory = self.projector(sbert_vec)
        enc_out = BaseModelOutput(last_hidden_state=memory)

        gen_ids = self.bart.generate(
            input_ids = self.dummy_input_ids.repeat(sbert_vec.size(0), 1),
            encoder_outputs = enc_out,
            decoder_start_token_id = self.bart.config.bos_token_id,
            **gen_kwargs)
        return gen_ids
