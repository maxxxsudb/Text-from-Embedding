# src/train.py
import argparse, os, torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from .model import Sbert2Text
from .data  import EmbTextDataset, collate

def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    ds_train = EmbTextDataset(opts.train_jsonl, tok, opts.max_len)
    ds_val   = EmbTextDataset(opts.val_jsonl,   tok, opts.max_len)

    dl_train = DataLoader(ds_train, batch_size=opts.bs, shuffle=True,
                          collate_fn=lambda b: collate(b, tok.pad_token_id))
    dl_val   = DataLoader(ds_val, batch_size=opts.bs,
                          collate_fn=lambda b: collate(b, tok.pad_token_id))

    model = Sbert2Text(k=opts.k).to(device)
    opt   = torch.optim.AdamW(model.projector.parameters(), lr=opts.lr)
    total_steps = opts.epochs * len(dl_train)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=opts.warmup, num_training_steps=total_steps)

    best_bleu, save_dir = 0.0, opts.save_dir
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, opts.epochs+1):
        model.train()
        pbar = tqdm(dl_train, desc=f"train {ep}")
        for embs, labels in pbar:
            embs, labels = embs.to(device), labels.to(device)
            loss = model(embs, labels=labels).loss

            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            pbar.set_postfix(loss=loss.item())

        # --- валидация (быстрый bleu по токенам) ---
        bleu = eval_bleu(model, dl_val, tok, device)
        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(model.state_dict(), f"{save_dir}/best.pt")
        print(f"Epoch {ep}: BLEU={bleu:.4f} (best={best_bleu:.4f})")

def eval_bleu(model, dl, tok, device):
    from sacrebleu import corpus_bleu
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for embs, labels in tqdm(dl, leave=False, desc="eval"):
            embs = embs.to(device)
            out_ids = model.generate(embs, max_new_tokens=64)
            hyps.extend(tok.batch_decode(out_ids, skip_special_tokens=True))
            refs.extend(tok.batch_decode(labels, skip_special_tokens=True))
    return corpus_bleu(hyps, [refs]).score / 100.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl",   required=True)
    ap.add_argument("--save_dir",    required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs",     type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--warmup", type=int,   default=2000)
    ap.add_argument("--k",      type=int,   default=1)
    ap.add_argument("--max_len",type=int,   default=128)
    opts = ap.parse_args()
    main(opts)
