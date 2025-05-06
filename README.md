# restore-from-emb

Восстанавливаем (или перефразируем) текст, имея на входе **ровно один 1024-мерный SBERT-вектор**.  
BART-декодер заморожен, обучается только маленький `KeyValueProjector`.

## Установка (Colab)

```bash
!git clone https://github.com/yourname/text-from-embedding.git
%cd text-from-embedding
!bash setup.sh            # ≈ 1-2 минуты
