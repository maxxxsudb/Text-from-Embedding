# src/data.py
import json, torch
from torch.utils.data import Dataset
from transformers import BartTokenizer

class EmbTextDataset(Dataset):
    """
    .jsonl: {"embedding": [...], "text": "..."}
    """
    def __init__(self, path: str, tokenizer: BartTokenizer, max_len=128):
        self.tok = tokenizer
        self.items = []
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                self.items.append((obj["embedding"], obj["text"]))
        self.max_len = max_len

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        emb, txt = self.items[idx]
        emb = torch.tensor(emb, dtype=torch.float)        # [1024]
        ids = self.tok(txt,
                       truncation=True,
                       max_length=self.max_len,
                       return_tensors="pt")
        return emb, ids.input_ids.squeeze(0)              # ([1024], [T])

def collate(batch, pad_id):
    embs, labels = zip(*batch)
    embs = torch.stack(embs)                              # [B,1024]
    max_t = max(l.size(0) for l in labels)
    padded = torch.full((len(labels), max_t),
                        pad_id, dtype=torch.long)
    for i,l in enumerate(labels): padded[i,:l.size(0)] = l
    return embs, padded
