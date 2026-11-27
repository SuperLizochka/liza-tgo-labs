#!/usr/bin/env python3
"""pt02: небольшой авторегрессионный трансформер для «Войны и мира»."""
import argparse
import csv
import json
import math
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
CORPUS_PATH = HERE / "corpus.txt"
DEFAULT_BPE = HERE.parent / "pt01" / "artifacts" / "bpe.json"
URL = ("https://raw.githubusercontent.com/nevmenandr/"
       "word2vec-russian-novels/master/books_before/WarAndPeace.txt")
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")
EOW = "</w>"
SEED = 0


def load_text(path):
    if not path.exists():
        print(f"Скачиваю корпус: {URL}")
        try:
            with urllib.request.urlopen(URL, timeout=90) as r:
                path.write_bytes(r.read())
        except Exception as e:
            sys.exit(f"Не удалось скачать корпус: {e}")
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def tokenize_words(text):
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def merge_symbols(symbols, a, b):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def apply_bpe(word, ranks):
    symbols = list(word) + [EOW]
    while len(symbols) > 1:
        candidates = [(ranks[p], p) for p in zip(symbols, symbols[1:]) if p in ranks]
        if not candidates:
            break
        _, (a, b) = min(candidates)
        symbols = merge_symbols(symbols, a, b)
    return symbols


def make_tokenizer(words, bpe_path):
    data = json.loads(bpe_path.read_text("utf-8"))
    merges = data["merges"][:1500]
    ranks = {tuple(pair): i for i, pair in enumerate(merges)}
    alphabet = {ch for word in words for ch in word} | {EOW}
    vocab = sorted(alphabet | {a + b for a, b in merges})
    stoi = {token: i for i, token in enumerate(vocab)}
    cache = {w: [stoi[t] for t in apply_bpe(w, ranks)] for w in set(words)}
    return merges, ranks, vocab, stoi, cache


def decode_ids(ids, vocab):
    pieces, word = [], ""
    for idx in ids:
        token = vocab[int(idx)]
        ended = token.endswith(EOW)
        word += token[:-len(EOW)] if ended else token
        if ended:
            pieces.append(word)
            word = ""
    if word:
        pieces.append(word)
    return " ".join(pieces)


class BigramLM:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.unigram = Counter()
        self.bigram = {}
        self.vocab_size = 0

    def fit(self, words, vocab_size):
        self.unigram = Counter(words)
        self.bigram = {}
        for a, b in zip(words, words[1:]):
            self.bigram.setdefault(a, Counter())[b] += 1
        self.vocab_size = vocab_size

    def nats(self, words):
        total = 0.0
        for a, b in zip(words, words[1:]):
            count = self.bigram.get(a, {}).get(b, 0)
            prob = (count + self.alpha) / (self.unigram.get(a, 0) + self.alpha * self.vocab_size)
            total -= math.log(prob)
        return total


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        mask = torch.triu(torch.ones(context, context, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:t, :t], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, context):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4, n_heads=4,
                 d_ff=512, context=128):
        super().__init__()
        self.context = context
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context, d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads, d_ff, context) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.blocks(x)
        return self.head(self.ln_f(x))


def encode_prompt(prompt, ranks, stoi):
    words = tokenize_words(prompt)
    ids = []
    for word in words:
        ids.extend(stoi[t] for t in apply_bpe(word, ranks))
    return ids


@torch.no_grad()
def generate(model, prompt, max_new_tokens, vocab, ranks, stoi,
             temperature=1.0, top_p=None, device="cpu"):
    model.eval()
    ids = encode_prompt(prompt, ranks, stoi)
    if not ids:
        ids = [stoi[EOW]]
    prompt_len = len(ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.context:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        if temperature == 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = logits / max(float(temperature), 1e-8)
            probs = F.softmax(logits, dim=-1)
            if top_p is not None:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=0)
                remove = cumulative - sorted_probs >= top_p
                sorted_probs[remove] = 0
                probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum()
            next_id = int(torch.multinomial(probs, 1).item())
        ids.append(next_id)
    new_ids = ids[prompt_len:]
    loop_start = None
    if temperature == 0:
        seen = {}
        for i in range(max(0, len(new_ids) - 3)):
            quad = tuple(new_ids[i:i + 4])
            if quad in seen:
                loop_start = i
                break
            seen[quad] = i
    return {
        "text": decode_ids(new_ids, vocab),
        "ids": new_ids,
        "n_tokens": len(new_ids),
        "loop_start": loop_start,
    }


def make_windows(tokens, count, context, rng):
    max_start = len(tokens) - context - 1
    starts = [rng.randint(0, max_start) for _ in range(count)]
    x = torch.tensor([tokens[s:s + context] for s in starts], dtype=torch.long)
    y = torch.tensor([tokens[s + 1:s + context + 1] for s in starts], dtype=torch.long)
    return x, y


@torch.no_grad()
def mean_loss(model, tokens, context, device, seed, windows=20):
    model.eval()
    rng = random.Random(seed)
    x, y = make_windows(tokens, windows, context, rng)
    x, y = x.to(device), y.to(device)
    logits = model(x)
    return float(F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                 y.reshape(-1)).item())


@torch.no_grad()
def score_tokens(model, ids, context, device):
    model.eval()
    if len(ids) < 2:
        return 0.0, 0
    total = 0.0
    # Score each next token with at most context preceding tokens.
    for target_pos in range(1, len(ids)):
        start = max(0, target_pos - context)
        x = torch.tensor([ids[start:target_pos]], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        target = torch.tensor([ids[target_pos]], dtype=torch.long, device=device)
        total += float(F.cross_entropy(logits.unsqueeze(0), target).item())
    return total, len(ids) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--bpe", default=str(DEFAULT_BPE))
    ap.add_argument("--corpus", default=str(CORPUS_PATH))
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    started = time.time()
    words = tokenize_words(load_text(Path(args.corpus)))
    if len(words) < 1000:
        sys.exit("Корпус слишком короткий.")
    split = int(len(words) * 0.9)
    train_words, val_words = words[:split], words[split:]
    merges, ranks, vocab, stoi, cache = make_tokenizer(words, Path(args.bpe))
    train_ids = [i for w in train_words for i in cache[w]]
    val_ids = [i for w in val_words for i in cache[w]]
    vocab_size = len(vocab)

    d_model, n_layers, n_heads, d_ff, context = 128, 4, 4, 512, 128
    steps = 200 if args.fast else 2000
    batch = 16 if args.fast else 32
    interval = 25 if args.fast else 100
    eval_words = 300 if args.fast else 2000
    max_gen = 40 if args.fast else 80
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TinyTransformer(vocab_size, d_model, n_layers, n_heads, d_ff, context).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = random.Random(SEED)
    val_rng = random.Random(SEED + 1)
    fixed_x, fixed_y = make_windows(val_ids, 20, context, val_rng)
    fixed_x, fixed_y = fixed_x.to(device), fixed_y.to(device)

    def fixed_val_loss():
        model.eval()
        with torch.no_grad():
            logits = model(fixed_x)
            return float(F.cross_entropy(logits.reshape(-1, vocab_size),
                                         fixed_y.reshape(-1)).item())

    curve = []
    samples = []
    prompt = "князь андрей"
    sample_steps = [0, 50, 100, steps] if args.fast else [0, 100, 500, steps]

    def save_sample(step):
        result = generate(model, prompt, 60, vocab, ranks, stoi,
                          temperature=0.8, device=device)
        prompt_ids = encode_prompt(prompt, ranks, stoi)
        samples.append({"step": step,
                        "text": decode_ids(prompt_ids, vocab) + " " + result["text"]})

    model.eval()
    initial_val = fixed_val_loss()
    initial_train = mean_loss(model, train_ids, context, device, SEED, windows=20)
    curve.append({"step": 0, "train_loss": initial_train, "val_loss": initial_val})
    save_sample(0)
    print(f"step 0/{steps}: train={initial_train:.4f}, val={initial_val:.4f}", flush=True)

    for step in range(1, steps + 1):
        model.train()
        x, y = make_windows(train_ids, batch, context, rng)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step in sample_steps:
            save_sample(step)
        if step % interval == 0 or step == steps:
            train_loss = mean_loss(model, train_ids, context, device, SEED + step,
                                   windows=20)
            val_loss = fixed_val_loss()
            curve.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
            print(f"step {step}/{steps}: train={train_loss:.4f}, val={val_loss:.4f}",
                  flush=True)

    # Generation settings on the trained model.
    runs = []
    for name, temperature, top_p in [
        ("greedy", 0.0, None), ("t0.7", 0.7, None),
        ("t1.5", 1.5, None), ("top_p0.9", 1.0, 0.9)
    ]:
        out = generate(model, prompt, max_gen, vocab, ranks, stoi,
                       temperature=temperature, top_p=top_p, device=device)
        item = {"name": name, "temperature": temperature, "top_p": top_p,
                "text": out["text"], "ids": out["ids"], "n_tokens": out["n_tokens"]}
        if name == "greedy":
            item["loop_start"] = out["loop_start"]
        runs.append(item)

    # Fair comparison on the same held-out words.
    selected_val_words = val_words[:eval_words]
    bigram = BigramLM(alpha=0.01)
    bigram.fit(train_words, vocab_size=len(set(words)))
    bigram_nats = bigram.nats(selected_val_words)
    bigram_tokens = max(0, len(selected_val_words) - 1)
    eval_ids = [i for w in selected_val_words for i in cache[w]]
    tiny_nats, tiny_tokens = score_tokens(model, eval_ids, context, device)
    n_words_eval = len(selected_val_words)

    def metric(nats, tokens):
        per_word = nats / max(1, n_words_eval)
        return {"tokens": tokens, "nats_total": round(nats, 4),
                "nats_per_word": round(per_word, 6),
                "ppl_per_word": round(math.exp(min(per_word, 50)), 4)}

    eval_result = {
        "words": n_words_eval,
        "bigram": {**metric(bigram_nats, bigram_tokens), "alpha": 0.01},
        "tiny": metric(tiny_nats, tiny_tokens)
    }

    ART.mkdir(parents=True, exist_ok=True)
    config = {
        "vocab_size": vocab_size, "d_model": d_model, "n_layers": n_layers,
        "n_heads": n_heads, "d_ff": d_ff, "context": context,
        "n_params": n_params, "steps": steps, "batch": batch, "lr": 0.001,
        "seed": SEED, "train_tokens": len(train_ids), "val_tokens": len(val_ids),
        "bpe_merges": len(merges), "train_minutes": round((time.time() - started) / 60, 2),
        "device": device
    }
    (ART / "tiny_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), "utf-8")
    with (ART / "train_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(curve)
    (ART / "samples.json").write_text(json.dumps(
        {"prompt": prompt, "temperature": 0.8, "samples": samples},
        ensure_ascii=False, indent=2), "utf-8")
    (ART / "generation.json").write_text(json.dumps(
        {"model": "tiny", "prompt": prompt, "max_new_tokens": max_gen, "runs": runs},
        ensure_ascii=False, indent=2), "utf-8")
    (ART / "eval.json").write_text(json.dumps(eval_result, ensure_ascii=False, indent=2), "utf-8")

    print("\nГотово.")
    print(f"Параметров: {n_params:,}; словарь: {vocab_size}; устройство: {device}")
    print(f"Время: {config['train_minutes']} мин")
    print(f"Оценка на {n_words_eval} словах: bigram={eval_result['bigram']['nats_per_word']:.4f}, "
          f"tiny={eval_result['tiny']['nats_per_word']:.4f}")
    print(f"Артефакты: {ART}")


if __name__ == "__main__":
    main()
