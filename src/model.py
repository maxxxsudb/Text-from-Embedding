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
    def __init__(self, in_dim: int = 1024, d_model: int = 768, k: int = 1, bottleneck_dim: Optional[int] = 1024):
        super().__init__()
        self.k, self.d_model = k, d_model
        
        actual_bottleneck_dim = bottleneck_dim
        if actual_bottleneck_dim is None or actual_bottleneck_dim <= 0:
            self.proj = nn.Linear(in_dim, k * d_model)
            print(f"KeyValueProjector (Single Layer): in_dim={in_dim} -> k*d_model={k*d_model}")
        else: 
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
                 bart_name: str = "antoinelouis/bart-base-russian", # Дефолтное значение
                 sbert_dim: int = 1024,
                 projector_k: int = 1,
                 projector_bottleneck_dim: Optional[int] = 1024, 
                 label_smoothing_factor: float = 0.1):
        super().__init__()
        
        try:
            self.bart = AutoModelForSeq2SeqLM.from_pretrained(bart_name)
            if hasattr(self.bart.config, 'label_smoothing_factor') and label_smoothing_factor > 0:
                 self.bart.config.label_smoothing_factor = label_smoothing_factor
                 print(f"Set label_smoothing_factor to {label_smoothing_factor} in BART config.")

        except Exception as e:
            print(f"Error loading BART model {bart_name}: {e}")
            raise
            
        self.bart.eval() 
        for p in self.bart.parameters():
            p.requires_grad_(False)

        cfg: BartConfig = self.bart.config # BartConfig может не совпадать с MBartConfig, но d_model обычно есть
        self.projector = KeyValueProjector(
            in_dim=sbert_dim, 
            d_model=cfg.d_model, 
            k=projector_k,
            bottleneck_dim=projector_bottleneck_dim
        )

        # decoder_start_token_id может отличаться для разных моделей BART/MBART
        # Используем cfg.decoder_start_token_id, если доступно, иначе cfg.bos_token_id
        # Для MBart обычно это ID токена языка. Токенизатор должен установить его правильно.
        # Если модель используется для генерации, она сама выберет правильный start_token_id,
        # если decoder_start_token_id не передается в generate() или если input_ids для generate()
        # начинается с этого токена.
        # Здесь dummy_input_ids используется как заглушка для generate(), если input_ids не переданы.
        # BART.generate() сам позаботится о decoder_start_token_id, если не указан явно.
        # Однако, если encoder_outputs переданы, а input_ids нет, то он может потребовать decoder_start_token_id.
        # Лучше всего передавать его явно в generate().
        # Здесь мы регистрируем буфер, который может быть использован как input_ids=[[decoder_start_token_id]]
        
        # Убедимся, что есть decoder_start_token_id. Для MBart это важно.
        # Если его нет, но есть bos_token_id, используем bos_token_id.
        # Если и его нет, это проблема конфигурации модели.
        _decoder_start_token_id = cfg.decoder_start_token_id
        if _decoder_start_token_id is None:
            _decoder_start_token_id = cfg.bos_token_id
        if _decoder_start_token_id is None:
            # Для MBart часто ID языка используется как decoder_start_token_id.
            # Например, для русского это может быть tokenizer.lang_code_to_id["ru_RU"]
            # Но это зависит от токенизатора и модели.
            # Проверим, есть ли у конфига forced_bos_token_id, который MBart использует как decoder_start_token_id
            _decoder_start_token_id = cfg.forced_bos_token_id 
            
        if _decoder_start_token_id is None:
            print("Warning: cfg.decoder_start_token_id and cfg.bos_token_id and cfg.forced_bos_token_id are None. Generation might fail or use a default.")
            # Установим значение по умолчанию, например, 0, но это может быть неверно.
            # Лучше, чтобы это было установлено в конфиге модели.
            _decoder_start_token_id = 0 # Placeholder, может потребоваться настройка

        self.register_buffer("dummy_input_ids_for_generate",
                             torch.tensor([[_decoder_start_token_id]]), # Используем определенный _decoder_start_token_id
                             persistent=False)
        
        print(f"Sbert2Text: BART model '{bart_name}' (d_model={cfg.d_model}) loaded and frozen.")
        print(f"Projector k={projector_k}, SBERT dim={sbert_dim}")
        print(f"Using decoder_start_token_id for generate: {_decoder_start_token_id}")


    def forward(self,
                sbert_vec: Tensor,
                labels: Optional[Tensor] = None):
        memory = self.projector(sbert_vec) # [B, k, d_model]
        
        enc_out = BaseModelOutput(
            last_hidden_state=memory,
            # attentions=None, # Не передаем attentions от проектора
            # hidden_states=None # Не передаем hidden_states от проектора
        )

        # attention_mask здесь относится к decoder_input_ids / labels
        # encoder_attention_mask относится к encoder_outputs
        # Если encoder_attention_mask убран, модель должна обработать encoder_outputs.last_hidden_state "как есть"
        outs = self.bart(
            encoder_outputs=enc_out,
            attention_mask=None, # Для decoder_input_ids, BART создаст, если None и labels переданы
            # encoder_attention_mask=encoder_attention_mask, # << УБРАНО ИЗ-ЗА TypeError
            decoder_input_ids=None, # BART сдвинет labels, если decoder_input_ids=None
            labels=labels
        )
        return outs

    @torch.inference_mode()
    def generate(self, sbert_vec: Tensor, **gen_kwargs):
        b = sbert_vec.size(0)
        memory = self.projector(sbert_vec) # [B, k, d_model]
        
        enc_out = BaseModelOutput(last_hidden_state=memory)
        
        # Убедимся, что decoder_start_token_id передан в generate, если не указан в gen_kwargs
        # dummy_input_ids_for_generate уже содержит [[decoder_start_token_id]]
        input_ids_for_gen = self.dummy_input_ids_for_generate.repeat(b, 1)

        # Если в gen_kwargs уже есть decoder_start_token_id, он будет использован.
        # Если нет, то generate использует self.bart.config.decoder_start_token_id.
        # Передача input_ids=[[decoder_start_token_id]] явно задает начало генерации.
        
        final_gen_kwargs = {**gen_kwargs}
        if 'decoder_start_token_id' not in final_gen_kwargs and self.bart.config.decoder_start_token_id is None:
             # Если модель не имеет decoder_start_token_id в конфиге, но мы его определили
             if hasattr(self.dummy_input_ids_for_generate, 'item'): # Проверка, что это тензор с одним элементом
                 final_gen_kwargs['decoder_start_token_id'] = self.dummy_input_ids_for_generate.item()
        
        # Если input_ids не предоставлены в gen_kwargs, используем наши подготовленные
        _input_ids = final_gen_kwargs.pop('input_ids', input_ids_for_gen)


        gen_ids = self.bart.generate(
            input_ids=_input_ids, 
            encoder_outputs=enc_out,
            # encoder_attention_mask=encoder_attention_mask, # << УБРАНО ИЗ-ЗА TypeError
            **final_gen_kwargs # передаем остальные gen_kwargs
        )
        return gen_ids
