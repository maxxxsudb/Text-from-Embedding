# src/generate.py
import argparse, ast, numpy as np, torch, json, os
from transformers import AutoTokenizer
from .model import Sbert2Text # Относительный импорт для запуска как часть пакета

DEFAULT_BART_MODEL = "antoinelouis/bart-base-russian"

def main(cli_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Определение параметров модели ---
    # Сначала пытаемся загрузить из train_args.json, если чекпоинт оттуда
    # CLI аргументы могут переопределять или использоваться, если train_args.json нет
    
    loaded_train_args = {}
    if cli_args.ckpt:
        train_args_path = os.path.join(os.path.dirname(cli_args.ckpt), "train_args.json")
        if os.path.exists(train_args_path):
            print(f"Loading training args from {train_args_path}")
            with open(train_args_path, "r") as f:
                loaded_train_args = json.load(f)
        else:
            print(f"Warning: {train_args_path} not found. Using CLI args for model architecture.")

    # Приоритет: CLI > train_args.json > argparse defaults
    bart_model_name = cli_args.bart_model if cli_args.bart_model != DEFAULT_BART_MODEL else loaded_train_args.get("bart_model", DEFAULT_BART_MODEL)
    sbert_dim_model = cli_args.sbert_dim if cli_args.sbert_dim != 1024 else loaded_train_args.get("sbert_dim", 1024)
    k_proj_model = cli_args.k if cli_args.k != 1 else loaded_train_args.get("k", 1)
    
    # Обработка proj_bottleneck_dim
    # Если в cli_args дефолт (1024), и в loaded_train_args есть значение, используем его.
    # Если в cli_args не дефолт, используем cli_args.
    if cli_args.proj_bottleneck_dim != 1024: # Пользователь явно указал bottleneck в CLI
        bottleneck_dim_model = cli_args.proj_bottleneck_dim
    else: # Пользователь не указал в CLI (используется default 1024) или указал 1024
          # Берем из сохраненных, если есть, иначе argparse default (1024)
        bottleneck_dim_model = loaded_train_args.get("proj_bottleneck_dim", cli_args.proj_bottleneck_dim)

    if bottleneck_dim_model is not None and bottleneck_dim_model <= 0:
        actual_bottleneck_dim_model = None # Для однослойного проектора
    else:
        actual_bottleneck_dim_model = bottleneck_dim_model
        
    print(f"Effective model params: BART='{bart_model_name}', SBERT_dim={sbert_dim_model}, k={k_proj_model}, bottleneck={actual_bottleneck_dim_model}")

    tok = AutoTokenizer.from_pretrained(bart_model_name)
    model = Sbert2Text(
        bart_name=bart_model_name,
        sbert_dim=sbert_dim_model,
        projector_k=k_proj_model,
        projector_bottleneck_dim=actual_bottleneck_dim_model,
        label_smoothing_factor=0.0 # Не нужно для инференса
    ).to(device)
    
    if not cli_args.ckpt:
        print("Error: --ckpt pointing to the model checkpoint is required.")
        return
        
    model.load_state_dict(torch.load(cli_args.ckpt, map_location=device))
    model.eval()
    print(f"Model loaded from {cli_args.ckpt}")

    # --- Чтение и подготовка вектора ---
    if cli_args.vec.endswith(".npy"):
        try:
            vec_np = np.load(cli_args.vec)
        except Exception as e:
            print(f"Error loading .npy file {cli_args.vec}: {e}")
            return
    else:
        try:
            vec_np = np.array(ast.literal_eval(cli_args.vec), dtype=float)
        except Exception as e:
            print(f"Error parsing vector from string: {e}")
            print("Ensure the vector string is a valid Python list or list of lists, e.g., '[0.1, 0.2]' or '[[0.1,0.2],[0.3,0.4]]'")
            return
    
    # Валидация и решейпинг вектора
    if vec_np.ndim == 1:
        if vec_np.shape[0] == sbert_dim_model:
            vec_np = vec_np.reshape(1, sbert_dim_model)  # [1, sbert_dim]
        else:
            print(f"Error: 1D Input vector shape {vec_np.shape} is not compatible with sbert_dim {sbert_dim_model}.")
            return
    elif vec_np.ndim == 2:
        if vec_np.shape[1] == sbert_dim_model:
            pass  # Уже [B, sbert_dim]
        else:
            print(f"Error: 2D Input vector shape {vec_np.shape} (dim 1) is not compatible with sbert_dim {sbert_dim_model}.")
            return
    else:
        print(f"Error: Input vector has unexpected number of dimensions: {vec_np.ndim}. Expected 1 or 2.")
        return
        
    vec_tensor = torch.tensor(vec_np, dtype=torch.float).to(device)
    print(f"Generating text for input vector(s) of shape: {vec_tensor.shape}")

    # --- Генерация текста ---
    out_ids = model.generate(vec_tensor,
                             num_beams=cli_args.num_beams,
                             max_new_tokens=cli_args.max_new_tokens,
                             min_length=cli_args.min_new_tokens, # min_new_tokens используется в generate
                             repetition_penalty=cli_args.repetition_penalty,
                             no_repeat_ngram_size=cli_args.no_repeat_ngram_size,
                             early_stopping=True if cli_args.num_beams > 1 else False) # Early stopping для beam search
    
    decoded_texts = tok.batch_decode(out_ids, skip_special_tokens=True)
    for i, text in enumerate(decoded_texts):
        print(f"\n--- Generated Text Batch Item {i+1} ---")
        print(text)

if __name__ == "__main__":
    pa = argparse.ArgumentParser(description="Generate text from SBERT vector")
    pa.add_argument("--ckpt", required=True, help="Path to the trained model checkpoint (.pt)")
    pa.add_argument("--vec",  required=True,
                    help="SBERT vector as a string '[0.1, ...]' or path to a .npy file (1D or 2D [1, dim] or [Batch, dim])")
    
    # Параметры модели (могут быть переопределены из train_args.json)
    pa.add_argument("--bart_model", type=str, default=DEFAULT_BART_MODEL, help=f"BART model name (default: {DEFAULT_BART_MODEL})")
    pa.add_argument("--sbert_dim", type=int, default=1024, help="SBERT embedding dimension (default: 1024)")
    pa.add_argument("--k", type=int, default=1, help="Number of memory slots for projector (default: 1)")
    pa.add_argument("--proj_bottleneck_dim", type=int, default=1024, help="Projector bottleneck dimension (default: SBERT_dim=1024). Set to 0 or negative for single-layer projector.")
    
    # Параметры генерации
    pa.add_argument("--max_new_tokens", type=int, default=100, help="Maximum new tokens to generate")
    pa.add_argument("--min_new_tokens", type=int, default=10, help="Minimum new tokens to generate")
    pa.add_argument("--num_beams",      type=int, default=4, help="Number of beams for generation")
    pa.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty (e.g., 1.0 means no penalty)")
    pa.add_argument("--no_repeat_ngram_size", type=int, default=3, help="If > 0, all N-grams of this size can only occur once.")
    
    args = pa.parse_args()
    main(args)
