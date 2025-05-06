# src/model.py
import torch, torch.nn as nn
from torch import Tensor
from transformers import BartForConditionalGeneration, BartConfig, AutoModelForSeq2SeqLM # Используем AutoModel
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional

class KeyValueProjector(nn.Module):
    """
    SBERT-вектор (B, in_dim) -> память (B, k, d_model)
    Использует bottleneck для уменьшения параметров или прямое проецирование.
    """
    def __init__(self, in_dim: int = 1024, d_model: int = 768, k: int = 1, bottleneck_dim: Optional[int] = 1024): # d_model для bart-base-russian, bottleneck_dim default 1024
        super().__init__()
        self.k, self.d_model = k, d_model
        
        actual_bottleneck_dim = bottleneck_dim
        # Если bottleneck_dim не указан (None) или <=0, то прямое проецирование (1 слой)
        if actual_bottleneck_dim is None or actual_bottleneck_dim <= 0:
            self.proj = nn.Linear(in_dim, k * d_model)
            print(f"KeyValueProjector (Single Layer): in_dim={in_dim} -> k*d_model={k*d_model}")
        else: # С bottleneck'ом (2 слоя MLP)
            self.proj = nn.Sequential(
                nn.Linear(in_dim, actual_bottleneck_dim),
                nn.GELU(),
                nn.Linear(actual_bottleneck_dim, k * d_model)
            )
            print(f"KeyValueProjector (2-Layer MLP): in_dim={in_dim} -> bottleneck={actual_bottleneck_dim} -> k*d_model={k*d_model}")
            
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"KeyValueProjector trainable parameters: {num_params/1e6:.2f} M")

    def forward(self, x: Tensor) -> Tensor:           # x: [B, in_dim]
        b = x.size(0)
        out = self.proj(x).view(b, self.k, self.d_model)
        return out                                    # [B, k, d_model]

class Sbert2Text(nn.Module):
    """
    Мини-обёртка: projector + замороженный BART-декодер.
    """
    def __init__(self,
                 bart_name: str = "antoinelouis/bart-base-russian",
                 sbert_dim: int = 1024,
                 projector_k: int = 1,
                 projector_bottleneck_dim: Optional[int] = 1024, # Default 1024
                 label_smoothing_factor: float = 0.1):
        super().__init__()
        
        try:
            self.bart = AutoModelForSeq2SeqLM.from_pretrained(
                bart_name,
                # label_smoothing_factor=label_smoothing_factor # Установка здесь может не работать для всех моделей или версий HF
            )
            # Установка label_smoothing_factor через config безопаснее, если свойство существует
            if hasattr(self.bart.config, 'label_smoothing_factor') and label_smoothing_factor > 0:
                 self.bart.config.label_smoothing_factor = label_smoothing_factor
                 print(f"Set label_smoothing_factor to {label_smoothing_factor} in BART config.")

        except Exception as e:
            print(f"Error loading BART model {bart_name}: {e}")
            raise
            
        self.bart.eval() 
        for p in self.bart.parameters():
            p.requires_grad_(False)

        cfg: BartConfig = self.bart.config
        self.projector = KeyValueProjector(
            in_dim=sbert_dim, 
            d_model=cfg.d_model, 
            k=projector_k,
            bottleneck_dim=projector_bottleneck_dim
        )

        self.register_buffer("dummy_input_ids",
                             torch.tensor([[cfg.decoder_start_token_id if cfg.decoder_start_token_id is not None else cfg.bos_token_id]]),
                             persistent=False)
        
        print(f"Sbert2Text: BART model '{bart_name}' (d_model={cfg.d_model}) loaded and frozen.")
        print(f"Projector k={projector_k}, SBERT dim={sbert_dim}")

    def forward(self,
                sbert_vec: Tensor,
                labels: Optional[Tensor] = None):
        memory = self.projector(sbert_vec)
        encoder_attention_mask = torch.ones(memory.size(0), memory.size(1), device=memory.device, dtype=torch.long)
        
        enc_out = BaseModelOutput(
            last_hidden_state=memory,
        )

        outs = self.bart(
            encoder_outputs=enc_out,
            attention_mask=None, 
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_ids=None, # BART сдвинет labels, если decoder_input_ids=None
            labels=labels
        )
        return outs

    @torch.inference_mode()
    def generate(self, sbert_vec: Tensor, **gen_kwargs):
        b = sbert_vec.size(0)
        memory = self.projector(sbert_vec)
        encoder_attention_mask = torch.ones(memory.size(0), memory.size(1), device=memory.device, dtype=torch.long)
        
        enc_out = BaseModelOutput(last_hidden_state=memory)
        input_ids_for_generate = self.dummy_input_ids.repeat(b, 1)

        gen_ids = self.bart.generate(
            input_ids=input_ids_for_generate,
            encoder_outputs=enc_out,
            encoder_attention_mask=encoder_attention_mask,
            **gen_kwargs
        )
        return gen_ids
