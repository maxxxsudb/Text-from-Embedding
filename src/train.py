# src/train.py
import argparse, os, torch, json
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from .model import Sbert2Text
from .data  import EmbTextDataset, collate
# sacrebleu импортируется внутри evaluate

DEFAULT_BART_MODEL = "antoinelouis/bart-base-russian"

def main(opts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(opts.bart_model)
    
    print(f"Device: {device}")
    print(f"Using BART model: {opts.bart_model}")
    print(f"Using SBERT dim: {opts.sbert_dim}")
    # Обработка proj_bottleneck_dim перед передачей в модель
    actual_proj_bottleneck_dim = opts.proj_bottleneck_dim
    if opts.proj_bottleneck_dim is not None and opts.proj_bottleneck_dim <= 0:
        actual_proj_bottleneck_dim = None # Для однослойного проектора
        print(f"Projector: k={opts.k}, Single Layer Projection (bottleneck_dim <= 0)")
    else:
        print(f"Projector: k={opts.k}, Bottleneck Dim: {actual_proj_bottleneck_dim}")

    print(f"Label smoothing factor: {opts.label_smoothing}")

    ds_train = EmbTextDataset(opts.train_jsonl, tok, opts.max_len)
    ds_val   = EmbTextDataset(opts.val_jsonl,   tok, opts.max_len)

    dl_train = DataLoader(ds_train, batch_size=opts.bs, shuffle=True, num_workers=opts.num_workers,
                          collate_fn=lambda b: collate(b, tok.pad_token_id), pin_memory=True if device.type == "cuda" else False)
    dl_val   = DataLoader(ds_val, batch_size=opts.bs, num_workers=opts.num_workers,
                          collate_fn=lambda b: collate(b, tok.pad_token_id), pin_memory=True if device.type == "cuda" else False)

    model = Sbert2Text(
        bart_name=opts.bart_model,
        sbert_dim=opts.sbert_dim,
        projector_k=opts.k,
        projector_bottleneck_dim=actual_proj_bottleneck_dim,
        label_smoothing_factor=opts.label_smoothing
    ).to(device)
    
    opt = torch.optim.AdamW(model.projector.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    
    num_training_steps = opts.epochs * len(dl_train)
    if opts.total_steps > 0 :
        num_training_steps = opts.total_steps
        effective_epochs = (num_training_steps + len(dl_train) - 1) // len(dl_train)
        print(f"Overriding epochs based on total_steps. Effective epochs: {effective_epochs}")
    else:
        effective_epochs = opts.epochs


    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=opts.warmup_steps, num_training_steps=num_training_steps
    )

    best_metric_val, save_dir = 0.0, opts.save_dir # BLEU
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "train_args.json"), "w") as f:
        json.dump(vars(opts), f, indent=2)

    global_step = 0
    for ep in range(1, effective_epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dl_train, desc=f"Train Epoch {ep}/{effective_epochs}")
        for batch_idx, (embs, labels) in enumerate(pbar):
            embs, labels = embs.to(device), labels.to(device)
            
            outputs = model(embs, labels=labels)
            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()
            if opts.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), opts.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad()
            
            pbar.set_postfix(loss=loss.item(), lr=sched.get_last_lr()[0])
            global_step += 1
            if opts.total_steps > 0 and global_step >= opts.total_steps:
                print(f"Reached total_steps {opts.total_steps}. Stopping training mid-epoch.")
                break
        
        avg_epoch_loss = epoch_loss / len(dl_train)
        print(f"Epoch {ep} average training loss: {avg_epoch_loss:.4f}")

        val_loss, bleu_score = evaluate(model, dl_val, tok, device, opts)
        print(f"Epoch {ep}: Val Loss={val_loss:.4f}, BLEU={bleu_score:.4f} (best BLEU={best_metric_val:.4f})")

        if bleu_score > best_metric_val:
            best_metric_val = bleu_score
            torch.save(model.state_dict(), os.path.join(save_dir, "best_bleu_model.pt"))
            print(f"Saved new best model with BLEU: {best_metric_val:.4f}")
        
        torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': sched.state_dict(),
            'best_metric_val': best_metric_val,
            'global_step': global_step,
            'train_args': vars(opts) # Сохраняем аргументы вместе с чекпоинтом
        }, os.path.join(save_dir, "last_checkpoint.pt"))

        if opts.total_steps > 0 and global_step >= opts.total_steps:
            print("Total steps reached. Finalizing training.")
            break


def evaluate(model, dl, tok, device, opts):
    from sacrebleu import corpus_bleu # Импорт здесь для чистоты
    model.eval()
    hyps_corpus, refs_corpus = [], []
    total_val_loss = 0.0
    
    with torch.no_grad():
        for embs, labels in tqdm(dl, leave=False, desc="Evaluating"):
            embs, labels = embs.to(device), labels.to(device)
            
            outputs = model(embs, labels=labels) # Получаем loss
            loss = outputs.loss
            total_val_loss += loss.item()

            out_ids = model.generate(embs, 
                                     max_new_tokens=opts.max_new_tokens_val, 
                                     num_beams=opts.num_beams_val,
                                     early_stopping=True) # Early stopping полезен с beam search
            
            decoded_hyps = tok.batch_decode(out_ids, skip_special_tokens=True)
            hyps_corpus.extend(decoded_hyps)
            
            # Для референсов убираем padding токены перед декодированием
            # Копируем labels, чтобы не изменять оригинал, если он используется где-то еще
            cleaned_labels_list = []
            for lab_tensor in labels:
                # Убираем все токены после первого EOS (если он есть) и все PAD
                # Сначала найдем EOS, если он есть, и обрежем по нему
                eos_indices = (lab_tensor == tok.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    lab_tensor_cleaned = lab_tensor[:eos_indices[0]+1] # Включая EOS
                else:
                    lab_tensor_cleaned = lab_tensor

                # Затем убираем PAD токены
                lab_tensor_cleaned = lab_tensor_cleaned[lab_tensor_cleaned != tok.pad_token_id]
                cleaned_labels_list.append(lab_tensor_cleaned)

            decoded_refs = tok.batch_decode(cleaned_labels_list, skip_special_tokens=True)
            refs_corpus.extend(decoded_refs)

    avg_val_loss = total_val_loss / len(dl) if len(dl) > 0 else 0.0
    
    # sacrebleu ожидает: hyps = list of strings, refs = list of lists of strings
    # Если у нас один референс на гипотезу, то refs_corpus должен быть [[ref1], [ref2], ...]
    sacrebleu_formatted_refs = [[r] for r in refs_corpus]
    
    bleu_score = 0.0
    if hyps_corpus and sacrebleu_formatted_refs:
        try:
            # `tokenize='intl'` или `tokenize='char'` может быть лучше для русского чем дефолтный '13a'
            # Для русского часто используют 'char' или 'none' (если тексты уже токенизированы) или 'intl'
            bleu_result = corpus_bleu(hyps_corpus, sacrebleu_formatted_refs, tokenize='intl')
            bleu_score = bleu_result.score
        except Exception as e:
            print(f"Could not calculate BLEU: {e}")
            bleu_score = 0.0 # Или другое значение по умолчанию

    return avg_val_loss, bleu_score


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train SBERT-to-Text model")
    ap.add_argument("--train_jsonl", required=True, help="Path to training data (JSONL)")
    ap.add_argument("--val_jsonl",   required=True, help="Path to validation data (JSONL)")
    ap.add_argument("--save_dir",    required=True, help="Directory to save checkpoints and logs")
    ap.add_argument("--bart_model",  type=str, default=DEFAULT_BART_MODEL, help="BART model name or path")
    ap.add_argument("--sbert_dim",   type=int, default=1024, help="Dimension of SBERT embeddings")
    
    ap.add_argument("--epochs",      type=int, default=5, help="Number of training epochs")
    ap.add_argument("--bs",          type=int, default=32, help="Batch size (try 16-64 for Colab with base BART)")
    ap.add_argument("--lr",          type=float, default=3e-4, help="Learning rate")
    ap.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    ap.add_argument("--warmup_steps",type=int,   default=500, help="Number of warmup steps")
    ap.add_argument("--total_steps", type=int,   default=-1, help="Total training steps (overrides epochs if > 0)")
    ap.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    ap.add_argument("--grad_clip",   type=float, default=1.0, help="Gradient clipping value (0 for no clipping)")
    
    ap.add_argument("--k",                 type=int, default=1, help="Number of memory slots from SBERT vector")
    ap.add_argument("--proj_bottleneck_dim", type=int, default=1024, help="Dimension of projector's bottleneck layer (default: SBERT_dim). Set to 0 or negative for single-layer projector.")
    
    ap.add_argument("--max_new_tokens_val", type=int, default=72, help="Max new tokens for validation generation")
    ap.add_argument("--num_beams_val",      type=int, default=3,  help="Number of beams for validation generation")

    ap.add_argument("--max_len",     type=int, default=128, help="Max sequence length for tokenizer input")
    ap.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    
    opts = ap.parse_args()
    main(opts)
