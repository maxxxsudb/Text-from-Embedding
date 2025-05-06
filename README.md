# Text-from-Embedding (Russian Version)

Восстанавливаем (или перефразируем) текст на русском языке, имея на входе **ровно один 1024-мерный SBERT-вектор** (например, от `sberbank-ai/sbert_large_nlu_ru`).
Используется замороженный русскоязычный BART-декодер (`antoinelouis/bart-base-russian`), обучается только небольшой MLP-проектор.

## Установка (Colab)

```bash
!git clone https://github.com/yourname/text-from-embedding.git # Замените на ваш URL репозитория
%cd text-from-embedding
!bash setup.sh            # ≈ 1-2 минуты
