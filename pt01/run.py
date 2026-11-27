#!/usr/bin/env python3
"""pt01: языковая модель из счётчиков + свой BPE на «Войне и мире».

Запуск (из любой папки):
    python pt01/run.py          # полный прогон
    python pt01/run.py --fast   # быстрый прогон (меньше слияний BPE)

Разрешены только стандартная библиотека и numpy; здесь хватает стандартной.
Результаты пишутся в pt01/artifacts/{bigram,bpe}.json.
"""
import argparse
import json
import math
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent          # все пути считаем от папки скрипта
ART = HERE / "artifacts"
CORPUS_PATH = HERE / "corpus.txt"                # он уже в .gitignore
URL = ("https://raw.githubusercontent.com/nevmenandr/"
       "word2vec-russian-novels/master/books_before/WarAndPeace.txt")
SEED = 42
EOW = "</w>"                                     # маркер конца слова для BPE
WORD_RE = re.compile(r"[а-я]+(?:-[а-я]+)*")


# ───────────────────────── корпус ─────────────────────────
def load_text(path):
    """Читаем корпус из файла; если файла нет, скачиваем по ссылке из задания."""
    path = Path(path)
    if not path.exists():
        print(f"скачиваю корпус: {URL}")
        try:
            with urllib.request.urlopen(URL, timeout=60) as r:
                path.write_bytes(r.read())
        except Exception as e:
            sys.exit(f"не удалось скачать корпус ({e}). Скачайте файл вручную и "
                     f"положите как {path}, либо запустите с --corpus <файл>.")
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def tokenize_words(text):
    """нижний регистр, ё→е, слова по регулярке; знаки препинания выбрасываются"""
    return WORD_RE.findall(text.lower().replace("ё", "е"))


# ───────────────────── биграммная модель ─────────────────────
class BigramLM:
    """Вся «модель» — это словарь счётчиков и деление.

    P(w|v) = (c(v,w) + a) / (c(v) + a*V)   — сглаживание Лапласа.
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.unigram = Counter()                 # c(v)
        self.bigram = defaultdict(Counter)       # c(v, w) = bigram[v][w]
        self.V = 0

    def fit(self, words, vocab_size=None):
        self.unigram = Counter(words)
        self.bigram = defaultdict(Counter)
        for v, w in zip(words, words[1:]):
            self.bigram[v][w] += 1
        # V — размер словаря в знаменателе сглаживания. Берём словарь всего
        # корпуса (закрытый словарь), чтобы слова из отложенной части не
        # «ломали» нормировку; сами их счётчики модель не видит.
        self.V = vocab_size or len(self.unigram)
        return self

    def count_pair(self, v, w):
        return self.bigram[v][w] if v in self.bigram else 0

    def prob(self, v, w, alpha=None):
        a = self.alpha if alpha is None else alpha
        return (self.count_pair(v, w) + a) / (self.unigram.get(v, 0) + a * self.V)

    def logprob(self, words, alpha=None):
        """log P(w_1..w_n | w_0) = сумма log P(w_t | w_{t-1}), первое слово не оценивается"""
        return sum(math.log(self.prob(v, w, alpha)) for v, w in zip(words, words[1:]))

    def perplexity(self, words, alpha=None):
        n = len(words) - 1
        return math.exp(-self.logprob(words, alpha) / n)

    def _sample(self, counter, temperature, rng):
        # P^(1/T) с последующей перенормировкой; random.choices нормирует веса сам
        items = list(counter.items())
        weights = [(c ** (1.0 / temperature)) for _, c in items]
        return rng.choices([w for w, _ in items], weights=weights, k=1)[0]

    def generate(self, start, n_words=30, temperature=1.0, rng=None):
        """Следующее слово — случайно по строке предыдущего (только реально
        встречавшиеся продолжения). Если у слова продолжений нет — берём
        случайное слово по частотам (откат на униграммы)."""
        rng = rng or random.Random(SEED)
        out = [start]
        while len(out) < n_words:
            row = self.bigram.get(out[-1])
            out.append(self._sample(row if row else self.unigram, temperature, rng))
        return out


def run_bigram(words, fast):
    n = len(words)
    n_train = int(n * 0.9)                       # последние 10 % — отложенная часть
    train, val = words[:n_train], words[n_train:]
    freq = Counter(words)
    vocab_size = len(freq)
    top20 = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:20]

    model = BigramLM(alpha=0.01).fit(train, vocab_size=vocab_size)

    # --- генерация: по образцу на каждую пару (температура, стартовое слово) ---
    rng = random.Random(SEED)
    generation = []
    for T in (0.5, 1, 2):
        for start in ("князь", "она"):
            text = " ".join(model.generate(start, 30, T, rng))
            generation.append({"temperature": T, "start": start, "text": text})

    # --- правдоподобие: связные 12 слов против тех же слов вперемешку ---
    sentence = val[:12]
    shuffled = sentence[:]
    while True:
        rng.shuffle(shuffled)
        if shuffled != sentence:                 # перемешали по-настоящему
            break
    alpha = 0.01
    unseen = sum(1 for v, w in zip(sentence, sentence[1:])
                 if model.count_pair(v, w) == 0)
    likelihood = {
        "sentence": " ".join(sentence),
        "shuffled": " ".join(shuffled),
        "alpha": alpha,
        "logprob_sentence": model.logprob(sentence, alpha),
        "logprob_shuffled": model.logprob(shuffled, alpha),
        "unseen_pairs_in_sentence": unseen,
    }

    # --- перплексия на отложенных 10 % ---
    perplexity = {str(a): model.perplexity(val, a) for a in (1, 0.1, 0.01)}

    result = {
        "corpus": "WarAndPeace",
        "words": n,
        "train_words": len(train),
        "val_words": len(val),
        "vocab_size": vocab_size,
        "top20": [[w, c] for w, c in top20],
        "perplexity": perplexity,
        "likelihood": likelihood,
        "generation": generation,
    }
    return result


# ──────────────────────────── BPE ────────────────────────────
def pairs_of(symbols):
    return zip(symbols, symbols[1:])


def merge_symbols(symbols, a, b):
    """Слить все (слева направо, без перекрытий) вхождения пары (a, b)."""
    out, i, ab = [], 0, a + b
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
            out.append(ab)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(word_freq, num_merges, min_freq):
    """Классический BPE. Для скорости храним счётчики пар и индекс
    «пара -> слова, где она есть», чтобы после слияния пересчитывать
    только затронутые слова, а не весь словарь."""
    vocab = [(list(w) + [EOW], f) for w, f in word_freq.items() if f >= min_freq]
    words = [s for s, _ in vocab]
    freqs = [f for _, f in vocab]
    pair_counts = Counter()
    where = defaultdict(set)
    for i, s in enumerate(words):
        for p in pairs_of(s):
            pair_counts[p] += freqs[i]
            where[p].add(i)

    merges = []
    while len(merges) < num_merges and pair_counts:
        best_c = max(pair_counts.values())
        best = min(p for p, c in pair_counts.items() if c == best_c)  # детерминизм при равенстве
        a, b = best
        merges.append([a, b])
        for i in list(where[best]):
            old = words[i]
            new = merge_symbols(old, a, b)
            if new == old:                       # устаревшая запись в индексе
                continue
            for p in pairs_of(old):
                pair_counts[p] -= freqs[i]
                if pair_counts[p] <= 0:
                    del pair_counts[p]
            for p in pairs_of(new):
                pair_counts[p] += freqs[i]
                where[p].add(i)
            words[i] = new
        pair_counts.pop(best, None)
        where.pop(best, None)
    return merges


def apply_bpe(word, ranks):
    """Применяем слияния «по рангам»: на каждом шаге сливается пара с
    наименьшим номером в списке merges."""
    symbols = list(word) + [EOW]
    while len(symbols) > 1:
        cand = [(ranks[p], p) for p in pairs_of(symbols) if p in ranks]
        if not cand:
            break
        _, (a, b) = min(cand)
        symbols = merge_symbols(symbols, a, b)
    return symbols


def run_bpe(words, fast):
    num_merges = 200 if fast else 1500
    min_freq = 10 if fast else 3
    word_freq = Counter(words)
    merges = train_bpe(word_freq, num_merges, min_freq)
    ranks = {tuple(m): i for i, m in enumerate(merges)}

    cache = {w: apply_bpe(w, ranks) for w in word_freq}    # токенизируем каждое слово один раз
    tokens_bpe = sum(len(cache[w]) * f for w, f in word_freq.items())
    chars = sum(len(w) * f for w, f in word_freq.items())
    n_words = len(words)

    alphabet = {ch for w in word_freq for ch in w} | {EOW}
    vocab = alphabet | {a + b for a, b in merges}

    examples = {}
    for w in ["наполеон", "сказала", "неудовольствие", "князь", "война", "мир",
              "андрей", "превосходительство", "улыбнулся", "отвечал"]:
        examples[w] = apply_bpe(w, ranks)

    return {
            "num_merges": len(merges),
        "vocab_size": len(vocab),
        "merges": merges,
        "examples": examples,
        "stats": {
            "chars": chars,
            "words": n_words,
            "tokens_bpe": tokens_bpe,
            "tokens_per_word": tokens_bpe / n_words,
            "compression_vs_chars": chars / tokens_bpe,
            "examples": examples,
        },
    }


# ─────────────────────────── main ───────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="быстрый прогон")
    ap.add_argument("--corpus", default=str(CORPUS_PATH), help="путь к corpus.txt")
    args = ap.parse_args()

    text = load_text(args.corpus)
    words = tokenize_words(text)
    print(f"слов в корпусе: {len(words)}")

    ART.mkdir(parents=True, exist_ok=True)
    big = run_bigram(words, args.fast)
    (ART / "bigram.json").write_text(json.dumps(big, ensure_ascii=False, indent=2), "utf-8")
    bpe = run_bpe(words, args.fast)
    (ART / "bpe.json").write_text(json.dumps(bpe, ensure_ascii=False, indent=2), "utf-8")

    # короткая сводка в консоль — удобно переносить в README
    print(f"\nслов: {big['words']} (train {big['train_words']}, val {big['val_words']}), "
          f"словарь: {big['vocab_size']}")
    print("top20:", ", ".join(f"{w}={c}" for w, c in big["top20"]))
    print("перплексия:", {k: round(v, 1) for k, v in big["perplexity"].items()})
    lk = big["likelihood"]
    print(f"logP связное: {lk['logprob_sentence']:.2f}, вперемешку: {lk['logprob_shuffled']:.2f}, "
          f"незнакомых пар: {lk['unseen_pairs_in_sentence']}")
    for g in big["generation"]:
        print(f"[T={g['temperature']}, {g['start']}] {g['text']}")
    st = bpe["stats"]
    print(f"\nBPE: слияний {bpe['num_merges']}, словарь {bpe['vocab_size']}, "
          f"токенов на слово {st['tokens_per_word']:.3f}, сжатие {st['compression_vs_chars']:.2f}x")
    for w, t in list(st["examples"].items())[:3]:
        print(f"  {w} -> {t}")


if __name__ == "__main__":
    main()
