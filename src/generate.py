# src/generate.py
import argparse, ast, numpy as np, torch
from transformers import BartTokenizer
from .model import Sbert2Text

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = Sbert2Text(k=args.k).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # --- вектор читаем из строки-python-list или .npy ---
    if args.vec.endswith(".npy"):
        vec = np.load(args.vec)
    else:
        vec = np.array(ast.literal_eval(args.vec))
    vec = torch.tensor(vec, dtype=torch.float).unsqueeze(0).to(device)

    out_ids = model.generate(vec,
                             num_beams=4,
                             max_new_tokens=args.max_len)
    print(tok.decode(out_ids[0], skip_special_tokens=True))

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--vec",  required=True,
                    help="либо строка-list '[0.1, ...]', либо path.npy")
    pa.add_argument("--k", type=int, default=1)
    pa.add_argument("--max_len", type=int, default=128)
    main(pa.parse_args())
