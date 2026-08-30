"""Prompt generators for E3: IOI (Wang et al. 2022 templates, ABC corruption) and
induction (repeated random tokens) — the single-mechanism control.

Every generator returns a list of dicts with keys
  clean (str or token list), corrupt, answer_tok, wrong_tok (IOI only), meta (dict)
and, for token-level tasks, `tokens`/`tokens_corrupt` as int lists.
"""
from __future__ import annotations

import numpy as np

# Wang et al. 2022, BABA templates ([B] = S appears twice, [A] = IO answers).
BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]


def _abba(t: str) -> str:
    # swap the first two name slots only
    i_b, i_a = t.index("[B]"), t.index("[A]")
    assert i_b < i_a
    return t[:i_b] + "[A]" + t[i_b + 3:i_a] + "[B]" + t[i_a + 3:]


ABBA_TEMPLATES = [_abba(t) for t in BABA_TEMPLATES]

NAMES = [
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer", "Joshua", "Amanda",
    "Daniel", "David", "James", "Robert", "John", "Joseph", "Andrew", "Ryan", "Brandon", "Jason",
    "Justin", "Sarah", "William", "Jonathan", "Stephanie", "Brian", "Nicole", "Nicholas",
    "Anthony", "Heather", "Eric", "Elizabeth", "Adam", "Megan", "Melissa", "Kevin", "Steven",
    "Thomas", "Timothy", "Christina", "Kyle", "Rachel", "Laura", "Lauren", "Amber", "Brittany",
    "Danielle", "Richard", "Kimberly", "Jeffrey", "Amy", "Crystal", "Michelle", "Tiffany",
    "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles", "Rebecca", "Jacob", "Stephen",
    "Patrick", "Sean", "Erin", "Zachary", "Jamie", "Kelly", "Samantha", "Nathan", "Sara",
    "Dustin", "Paul", "Angela", "Tyler", "Scott", "Katherine", "Andrea", "Gregory", "Erica",
    "Mary", "Travis", "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen", "Jose", "Alexander",
    "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney", "Christine", "Alicia",
    "Cody", "Allison", "Bradley", "Samuel",
]
PLACES = ["store", "garden", "restaurant", "school", "hospital", "office", "house", "station"]
OBJECTS = ["ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack"]


def single_token_names(model, names=NAMES):
    return [n for n in names if len(model.to_tokens(" " + n, prepend_bos=False)[0]) == 1]


def _fill(t, a, b, place, obj):
    s = t.replace("[A]", a).replace("[B]", b).replace("[PLACE]", place).replace("[OBJECT]", obj)
    # strip the answer (" [A]" at the very end): prompt ends right before IO
    assert s.endswith(" " + a)
    return s[: -len(" " + a)]


def make_ioi_prompts(model, n_per_cell=80, seed=0):
    """2 orders x 15 templates x n_per_cell prompts. ABC corruption (three fresh names)."""
    rng = np.random.default_rng(seed)
    names = single_token_names(model)
    out = []
    for order, temps in (("BABA", BABA_TEMPLATES), ("ABBA", ABBA_TEMPLATES)):
        for ti, t in enumerate(temps):
            for _ in range(n_per_cell):
                a, b = rng.choice(names, 2, replace=False)
                place = rng.choice(PLACES); obj = rng.choice(OBJECTS)
                others = [n for n in names if n not in (a, b)]
                c1, c2, c3 = rng.choice(others, 3, replace=False)
                clean = _fill(t, a, b, place, obj)
                # ABC: S1, IO, S2 slots -> three fresh distinct names, positions preserved.
                # In the template, [B] occurs twice and [A] once (plus the stripped answer).
                tc = t.replace("[A]", "[X]", 1)  # first-occurring A (the only one left after strip)
                corrupt = tc.replace("[B]", "[Y]", 1).replace("[B]", "[Z]", 1)
                corrupt = corrupt.replace("[X]", c1).replace("[Y]", c2).replace("[Z]", c3)
                corrupt = corrupt.replace("[PLACE]", place).replace("[OBJECT]", obj)
                assert corrupt.endswith(" [A]")
                corrupt = corrupt[:-4]
                out.append(dict(
                    clean=clean, corrupt=corrupt,
                    answer_tok=model.to_single_token(" " + a),
                    wrong_tok=model.to_single_token(" " + b),
                    meta=dict(order=order, template=ti, io=a, s=b, place=place, obj=obj),
                ))
    return out


def make_induction_prompts(model, lengths=(6, 8, 10, 12), n_per_len=600, seed=0,
                           vocab_lo=1000, vocab_hi=40000):
    """[BOS] r1..rL r1..r_{L-1}; target r_L. Corrupt: first copy replaced by fresh tokens."""
    rng = np.random.default_rng(seed)
    bos = model.tokenizer.bos_token_id
    out = []
    for L in lengths:
        for _ in range(n_per_len):
            r = rng.integers(vocab_lo, vocab_hi, size=L)
            s = rng.integers(vocab_lo, vocab_hi, size=L)
            clean = [bos] + r.tolist() + r[:-1].tolist()
            corrupt = [bos] + s.tolist() + r[:-1].tolist()
            out.append(dict(tokens=clean, tokens_corrupt=corrupt, answer_tok=int(r[-1]),
                            wrong_tok=None, meta=dict(L=int(L))))
    return out
