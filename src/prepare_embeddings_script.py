import json
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer # Убедитесь, что установлена
import os

def main(args):
    # 1. Загружаем SBERT модель
    print(f"Loading SBERT model: {args.sbert_model_name}...")
    sbert = SentenceTransformer(args.sbert_model_name)
    print(f"SBERT model {args.sbert_model_name} loaded.")

    # 2. Читаем тексты из файла
    if not os.path.exists(args.input_text_file):
        print(f"Error: Input text file not found at {args.input_text_file}")
        return

    texts_to_encode = []
    with open(args.input_text_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if line:
                texts_to_encode.append(line)
    print(f"Read {len(texts_to_encode)} texts from {args.input_text_file}")

    if not texts_to_encode:
        print("No texts to encode.")
        return

    # Создаем директорию для выходного файла, если ее нет
    output_dir = os.path.dirname(args.output_jsonl_file)
    if output_dir: # Если путь содержит директорию
        os.makedirs(output_dir, exist_ok=True)

    # 3. Кодируем тексты и сохраняем в .jsonl
    # Можно батчами для эффективности, если текстов много
    with open(args.output_jsonl_file, "w", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(texts_to_encode), args.batch_size), desc="Encoding texts"):
            batch_texts = texts_to_encode[i:i+args.batch_size]
            embeddings = sbert.encode(
                batch_texts,
                convert_to_tensor=False, # Возвращает numpy array
                normalize_embeddings=args.normalize_embeddings,
                show_progress_bar=False # tqdm управляет прогрессом снаружи
            )

            for text, vec in zip(batch_texts, embeddings):
                record = {"embedding": vec.tolist(), "text": text}
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(texts_to_encode)} records to {args.output_jsonl_file}")
    sbert_dim_check = len(embeddings[0]) if len(embeddings) > 0 else "N/A"
    print(f"Dimension of generated SBERT embeddings: {sbert_dim_check}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode texts using SBERT and save to JSONL.")
    parser.add_argument("--input_text_file", type=str, required=True, help="Path to the input text file (one text per line).")
    parser.add_argument("--output_jsonl_file", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--sbert_model_name", type=str, default="sberbank-ai/sbert_large_nlu_ru", help="Name of the SBERT model from HuggingFace or sentence-transformers.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--normalize_embeddings", type=bool, default=False, help="Whether to normalize SBERT embeddings to unit length.")
    
    args = parser.parse_args()
    main(args)
