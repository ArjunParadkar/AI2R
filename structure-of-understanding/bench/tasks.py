"""Behaviour suites for real language models (used through whitebox_ee.llm.HFSystem).

Common task interface
---------------------
  roles                      ordered role names (token positions); with the layer index they
                             define the sites f"L{l}@{role}" of HFSystem
  behaviours                 ordered behaviour names; each is a next-token readout restricted
                             to a small class set
  input_names                names of the input nodes (integer-valued)
  sample(n, rng, **opts)     -> list of example dicts (each has 'text', the full teacher-forced string)
  bind(tokenizer)            choose the tokenizer-specific surface format (HFSystem calls it)
  render(ex, tokenizer)      -> (input_ids list, {role: position}, {behaviour: readout position})
                             readout position = the position whose next-token logits predict
                             the behaviour's token
  class_token_ids(tokenizer) -> list of token ids, class index c <-> token id (shared by all examples)
  example_class_ids(ex, tok) -> optional per-example class ids (AgreementTask); an explicit
                             ex['class_ids'] (list, or dict behaviour -> list) overrides both
  input_values(X)            -> dict input name -> int array
  labels(X)                  -> dict variable name -> int array (true and decoy/wrong hypotheses)
  edit(ex, {name: value})    -> new example with the input variables changed and text re-rendered
  targets(X)                 -> dict behaviour -> correct class index

Positions are always derived from the tokenizer's character offsets of the real
string (no hand-built token sequences), so the model sees its natural tokenization;
`render` raises if a digit role does not land on a single-digit token.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_FMT_CACHE: Dict[str, str] = {}


def _tok_key(tok) -> str:
    return f"{getattr(tok, 'name_or_path', '')}|{len(tok)}|{type(tok).__name__}"


def _char_to_token(tok, text: str, char_idx: Sequence[int]):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids = list(enc["input_ids"])
    offs = enc["offset_mapping"]
    lut = np.full(len(text), -1, dtype=np.int64)
    for t, (s, e) in enumerate(offs):
        if e > s:
            lut[s:e] = t
    pos = [int(lut[c]) for c in char_idx]
    return ids, offs, pos


def _digits_single(tok, text: str) -> bool:
    """True iff every digit character of `text` is the only digit in its token."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    for s, e in enc["offset_mapping"]:
        piece = text[s:e]
        if sum(ch.isdigit() for ch in piece) > 1:
            return False
    return True


def digit_format(tok) -> str:
    """'native' (e.g. '347+295=0642', Qwen2.5/Llama-3-style single-digit splitting) if the
    tokenizer splits contiguous digits into single tokens, else 'spaced'
    ('3 4 7 + 2 9 5 = 0 6 4 2'; Pythia/GPT-NeoX gives ' d' tokens, OLMo-2 gives ' ' + 'd')."""
    k = _tok_key(tok)
    if k in _FMT_CACHE:
        return _FMT_CACHE[k]
    probe = "\n".join(["1234567890+9876543210=01234567890", "305*7=2135", "0000+9999=09999"])
    if _digits_single(tok, probe):
        f = "native"
    elif _digits_single(tok, " ".join(probe.replace("\n", "#")).replace("#", "\n")):
        f = "spaced"
    else:
        raise ValueError(f"no single-digit format found for tokenizer {k}")
    _FMT_CACHE[k] = f
    return f


def _digits(x: int, n: int) -> List[int]:
    """Little-endian digits of x, length n (zero padded)."""
    return [(x // 10 ** i) % 10 for i in range(n)]


def _from_digits(d: Sequence[int]) -> int:
    return int(sum(int(v) * 10 ** i for i, v in enumerate(d)))


# ---------------------------------------------------------------------------
# arithmetic base
# ---------------------------------------------------------------------------

class _Arithmetic:
    """Shared machinery: 'op1 OP op2 = answer', operands zero-padded to fixed length and
    written most-significant digit first, answer zero-padded to n_out digits, teacher-forced.
    A fixed few-shot prefix of solved problems (fixed seed) precedes the query."""

    op = "+"
    n1 = 3
    n2 = 3
    n_out = 4
    name = "arith"

    def __init__(self, n_shots: int = 3, shot_seed: int = 12345, fmt: Optional[str] = None):
        self.n_shots = n_shots
        self.fmt = fmt                     # None -> chosen per tokenizer by bind/render
        self._fixed_fmt = fmt
        self.roles = ([f"op1_d{i}" for i in range(self.n1)] + [f"op2_d{i}" for i in range(self.n2)]
                      + ["eq"] + [f"pre_ans_d{j}" for j in range(self.n_out)])
        self.behaviours = [f"out{j}" for j in range(self.n_out)]
        self.input_names = [f"a{i}" for i in range(self.n1)] + [f"b{i}" for i in range(self.n2)]
        rng = np.random.default_rng(shot_seed)
        self.shots: List[dict] = []
        self.shots = [{"task": self.name, "a": int(a), "b": int(b)} for a, b in self._draw(rng, n_shots)]

    # -- problem generation (subclasses) --------------------------------
    def _answer(self, a: int, b: int) -> int:
        raise NotImplementedError

    def _draw(self, rng, n) -> List[Tuple[int, int]]:
        a = rng.integers(0, 10 ** self.n1, n)
        b = rng.integers(0, 10 ** self.n2, n)
        return list(zip(a.tolist(), b.tolist()))

    def _make(self, a: int, b: int) -> dict:
        ex = {"task": self.name, "a": int(a), "b": int(b)}
        ex["text"] = self._compose(ex, self.fmt or "spaced")[0]
        return ex

    # -- surface format --------------------------------------------------
    def bind(self, tok):
        self.fmt = self._fixed_fmt or digit_format(tok)
        return self

    def fmt_for(self, tok) -> str:
        return self._fixed_fmt or digit_format(tok)

    def _line(self, ex, fmt):
        """Returns (string, {role: char index within the string})."""
        a, b = ex["a"], ex["b"]
        ad = _digits(a, self.n1)[::-1]
        bd = _digits(b, self.n2)[::-1]
        yd = _digits(self._answer(a, b), self.n_out)[::-1]
        sep = " " if fmt == "spaced" else ""
        toks: List[Tuple[str, Optional[str]]] = []
        toks += [(str(d), f"op1_d{self.n1 - 1 - k}") for k, d in enumerate(ad)]
        toks += [(self.op, None)]
        toks += [(str(d), f"op2_d{self.n2 - 1 - k}") for k, d in enumerate(bd)]
        toks += [("=", "eq")]
        toks += [(str(d), f"ans_d{self.n_out - 1 - k}") for k, d in enumerate(yd)]
        s, where = "", {}
        for k, (piece, role) in enumerate(toks):
            if k:
                s += sep
            if role is not None:
                where[role] = len(s)
            s += piece
        return s, where

    def _compose(self, ex, fmt):
        lines = [self._line(e, fmt)[0] for e in self.shots]
        q, where = self._line(ex, fmt)
        prefix = "\n".join(lines) + ("\n" if lines else "")
        if fmt == "spaced":
            prefix = prefix.replace("\n", "\n ") if lines else ""
            prefix = (" " + prefix) if lines else " "
        return prefix + q, {r: len(prefix) + c for r, c in where.items()}

    def render(self, ex, tok):
        fmt = self.fmt_for(tok)
        text, where = self._compose(ex, fmt)
        names = list(where.keys())
        ids, offs, pos = _char_to_token(tok, text, [where[r] for r in names])
        P = dict(zip(names, pos))
        for r in names:
            if r == "eq":
                continue
            s, e = offs[P[r]]
            if text[s:e].strip() != text[where[r]]:
                raise ValueError(f"role {r} is not a single-digit token: {text[s:e]!r} in {text!r}")
        roles = {r: P[r] for r in self.roles if r in P}
        for j in range(self.n_out):
            roles[f"pre_ans_d{j}"] = P[f"ans_d{j}"] - 1
        readouts = {f"out{j}": roles[f"pre_ans_d{j}"] for j in range(self.n_out)}
        return ids, roles, readouts

    def class_token_ids(self, tok) -> List[int]:
        fmt = self.fmt_for(tok)
        out = []
        for d in range(10):
            text = f"1 {self.op} 1 = {d}" if fmt == "spaced" else f"1{self.op}1={d}"
            text = (" " + text) if fmt == "spaced" else text
            ids, _, pos = _char_to_token(tok, text, [len(text) - 1])
            out.append(ids[pos[0]])
        if len(set(out)) != 10:
            raise ValueError("digit class tokens are not distinct")
        return out

    # -- variables --------------------------------------------------------
    def input_values(self, X) -> Dict[str, np.ndarray]:
        A = np.array([_digits(e["a"], self.n1) for e in X], dtype=np.int64).reshape(len(X), self.n1)
        B = np.array([_digits(e["b"], self.n2) for e in X], dtype=np.int64).reshape(len(X), self.n2)
        out = {f"a{i}": A[:, i] for i in range(self.n1)}
        out.update({f"b{i}": B[:, i] for i in range(self.n2)})
        return out

    def targets(self, X) -> Dict[str, np.ndarray]:
        Y = np.array([_digits(self._answer(e["a"], e["b"]), self.n_out) for e in X], dtype=np.int64)
        Y = Y.reshape(len(X), self.n_out)
        return {f"out{j}": Y[:, j] for j in range(self.n_out)}

    def edit(self, ex, edits: Dict[str, int]) -> dict:
        ad = _digits(ex["a"], self.n1)
        bd = _digits(ex["b"], self.n2)
        for k, v in edits.items():
            v = int(v)
            if not 0 <= v <= 9:
                raise ValueError(f"digit value out of range: {k}={v}")
            if k[0] == "a":
                ad[int(k[1:])] = v
            elif k[0] == "b":
                bd[int(k[1:])] = v
            else:
                raise KeyError(k)
        new = dict(ex)
        new["a"], new["b"] = _from_digits(ad), _from_digits(bd)
        new.pop("class_ids", None)
        new["text"] = self._compose(new, self.fmt or "spaced")[0]
        return new


# ---------------------------------------------------------------------------
# addition
# ---------------------------------------------------------------------------

class AdditionTask(_Arithmetic):
    """a + b, both n_digits long (zero padded), answer n_digits+1 digits (zero padded).

    Roles: op1_d{i}, op2_d{i} (i=0 least significant), 'eq', pre_ans_d{j} (the position
    whose next-token logits predict answer digit j).  Note: with the 'native' format and
    with Pythia's spaced format, pre_ans_d{n} (the carry-out digit, written first) IS the
    '=' token, so sites L@eq and L@pre_ans_d{n} coincide; with OLMo-2's spaced format it is
    the ' ' token after ' ='.
    Behaviours out{j}, j = 0..n (out{n} = carry-out).
    Labels: a_i, b_i, s_i = a_i+b_i (0..18), c_i = [s_i + c_{i-1} >= 10] (c_{-1}=0),
            r_i = (s_i + c_{i-1}) mod 10, wrong hypotheses w_i = (a_i + b_{i+1}) mod 10
            (i < n-1) and q_i = (a_i b_i) mod 7.
    sample(..., carries=None|True|False|tuple): any / at least one carry / no carry /
            exact carry pattern (c_0, ..., c_{n-1}).
    """

    op = "+"
    name = "add"

    def __init__(self, n_digits: int = 3, n_shots: int = 3, shot_seed: int = 12345, fmt: Optional[str] = None):
        self.n = n_digits
        self.n1 = self.n2 = n_digits
        self.n_out = n_digits + 1
        super().__init__(n_shots, shot_seed, fmt)

    def _answer(self, a, b):
        return a + b

    def carry_pattern(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """A, B: (N, n) little-endian digits -> (N, n) carries."""
        C = np.zeros_like(A)
        c = np.zeros(len(A), dtype=A.dtype)
        for i in range(self.n):
            c = ((A[:, i] + B[:, i] + c) >= 10).astype(A.dtype)
            C[:, i] = c
        return C

    def sample(self, n: int, rng, carries=None, exclude_shots: bool = True) -> List[dict]:
        out: List[dict] = []
        shot_set = {(e["a"], e["b"]) for e in self.shots} if exclude_shots else set()
        tries = 0
        while len(out) < n:
            tries += 1
            if tries > 1000:
                raise RuntimeError("could not sample enough problems with the requested carry pattern")
            m = max(4 * (n - len(out)), 64)
            A = rng.integers(0, 10, (m, self.n))
            B = rng.integers(0, 10, (m, self.n))
            C = self.carry_pattern(A, B)
            if carries is True:
                keep = C.any(1)
            elif carries is False:
                keep = ~C.any(1)
            elif carries is None:
                keep = np.ones(m, bool)
            else:
                keep = (C == np.asarray(carries)[None, :]).all(1)
            for i in np.where(keep)[0]:
                a, b = _from_digits(A[i]), _from_digits(B[i])
                if (a, b) in shot_set:
                    continue
                out.append(self._make(a, b))
                if len(out) == n:
                    break
        return out

    def labels(self, X) -> Dict[str, np.ndarray]:
        iv = self.input_values(X)
        A = np.stack([iv[f"a{i}"] for i in range(self.n)], 1)
        B = np.stack([iv[f"b{i}"] for i in range(self.n)], 1)
        L = dict(iv)
        c = np.zeros(len(X), dtype=np.int64)
        for i in range(self.n):
            s = A[:, i] + B[:, i]
            L[f"s{i}"] = s
            L[f"r{i}"] = (s + c) % 10
            c = ((s + c) >= 10).astype(np.int64)
            L[f"c{i}"] = c
            L[f"q{i}"] = (A[:, i] * B[:, i]) % 7
            if i + 1 < self.n:
                L[f"w{i}"] = (A[:, i] + B[:, i + 1]) % 10
        return L


# ---------------------------------------------------------------------------
# multiplication
# ---------------------------------------------------------------------------

class MultiplicationTask(_Arithmetic):
    """a (n_digits_a digits, zero padded) * b (one digit); answer n_digits_a+1 digits.

    Inputs a0..a{n-1}, b0.  Roles op1_d{i}, op2_d0, eq, pre_ans_d{j}; behaviours out{j}.
    Labels: a_i, b0, p_i = a_i b (0..81), t_i = p_i + k_{i-1} (k_{-1}=0),
            k_i = floor(t_i / 10) (carries, 0..8), r_i = t_i mod 10 (result digits; the top
            result digit is k_{n-1}), wrong hypotheses w_i = (a_i + b) mod 10, q_i = p_i mod 7.
    sample(..., carries=None|True|False): any / some k_i > 0 for i < n-1 / none.
    """

    op = "*"
    name = "mul"

    def __init__(self, n_digits_a: int = 2, n_shots: int = 3, shot_seed: int = 12345, fmt: Optional[str] = None):
        self.n = n_digits_a
        self.n1, self.n2 = n_digits_a, 1
        self.n_out = n_digits_a + 1
        super().__init__(n_shots, shot_seed, fmt)

    def _answer(self, a, b):
        return a * b

    def sample(self, n: int, rng, carries=None, exclude_shots: bool = True) -> List[dict]:
        out: List[dict] = []
        shot_set = {(e["a"], e["b"]) for e in self.shots} if exclude_shots else set()
        tries = 0
        while len(out) < n:
            tries += 1
            if tries > 1000:
                raise RuntimeError("could not sample enough problems")
            m = max(4 * (n - len(out)), 64)
            a = rng.integers(0, 10 ** self.n, m)
            b = rng.integers(0, 10, m)
            for ai, bi in zip(a.tolist(), b.tolist()):
                if (ai, bi) in shot_set:
                    continue
                if carries is not None:
                    has = any(self._carries(ai, bi)[: self.n - 1])
                    if has != bool(carries):
                        continue
                out.append(self._make(ai, bi))
                if len(out) == n:
                    break
        return out

    def _carries(self, a, b):
        k, ks = 0, []
        for d in _digits(a, self.n):
            k = (d * b + k) // 10
            ks.append(k)
        return ks

    def labels(self, X) -> Dict[str, np.ndarray]:
        iv = self.input_values(X)
        b = iv["b0"]
        L = dict(iv)
        k = np.zeros(len(X), dtype=np.int64)
        for i in range(self.n):
            p = iv[f"a{i}"] * b
            t = p + k
            L[f"p{i}"] = p
            L[f"t{i}"] = t
            L[f"r{i}"] = t % 10
            k = t // 10
            L[f"k{i}"] = k
            L[f"w{i}"] = (iv[f"a{i}"] + b) % 10
            L[f"q{i}"] = p % 7
        return L


# ---------------------------------------------------------------------------
# subject-verb agreement
# ---------------------------------------------------------------------------

NOUNS = ["author", "senator", "pilot", "doctor", "teacher", "farmer", "singer", "dancer", "manager",
         "student", "officer", "driver", "lawyer", "writer", "painter", "player", "guard", "nurse",
         "king", "friend", "boy", "girl", "key", "cabinet", "table", "car", "house", "book", "chair",
         "door", "window", "picture", "letter", "bottle", "report", "plan"]
PREPS = ["near", "behind", "beside", "with", "from", "under", "above", "by"]
MODS = ["probably", "certainly", "clearly", "surely"]          # template 0 (PP): adverb before the verb
RCVERBS = ["liked", "saw", "met", "knew"]                      # template 1 (RC): past-tense verb (number-neutral)
VERB_PAIRS = [(" is", " are"), (" was", " were"), (" has", " have")]


class AgreementTask:
    """Subject-verb number agreement with an attractor noun (Marvin & Linzen 2018 style,
    own small grammar):
        template 0 (PP):  "The {subj} {prep} the {attr} {mod} {verb}"
        template 1 (RC):  "The {subj} that the {attr} {rcverb} {verb}"
    Nouns: regular -s plurals; every word is a single token (with its leading space) for the
    Pythia, Qwen2.5 and OLMo-2 tokenizers (checked in tests/test_llm.py).
    One behaviour 'verb' with 2 classes (0 = singular form, 1 = plural form of the pair
    verbpair_id: is/are, was/were, has/have); per-example class ids via example_class_ids.
    Inputs: subj_num, attr_num, subj_id, attr_id, prep_id, verbpair_id, template, mod_id
    (prep_id is ignored by template 1, whose 'prep' role is 'that'; mod_id indexes MODS for
    template 0 and RCVERBS for template 1).
    Roles: subj, prep, attr, verb_pre (the adverb / rc verb: the position predicting the verb).
    Labels: inputs + match = [subj_num == attr_num], decoys subj_parity = subj_id mod 2,
            attr_parity = attr_id mod 2.  targets: verb = subj_num.
    n_shots > 0 prepends fixed solved sentences ("... is here.") from a fixed seed.
    """

    name = "agree"

    def __init__(self, n_shots: int = 0, shot_seed: int = 12345, templates: Sequence[int] = (0, 1),
                 nouns: Sequence[str] = NOUNS):
        self.nouns = list(nouns)
        self.templates = list(templates)
        self.roles = ["subj", "prep", "attr", "verb_pre"]
        self.behaviours = ["verb"]
        self.input_names = ["subj_num", "attr_num", "subj_id", "attr_id", "prep_id", "verbpair_id",
                            "template", "mod_id"]
        self.n_shots = n_shots
        rng = np.random.default_rng(shot_seed)
        self.shots = self.sample(n_shots, rng, _shots=True) if n_shots else []

    def bind(self, tok):
        return self

    def sample(self, n: int, rng, attractor: Optional[str] = None, _shots: bool = False) -> List[dict]:
        """attractor: None (any), 'match' (attr number == subj number) or 'mismatch'."""
        out = []
        nn = len(self.nouns)
        while len(out) < n:
            s, a = rng.integers(0, nn, 2)
            if s == a:
                continue
            ex = {"task": self.name, "subj_id": int(s), "attr_id": int(a),
                  "subj_num": int(rng.integers(0, 2)), "attr_num": int(rng.integers(0, 2)),
                  "prep_id": int(rng.integers(0, len(PREPS))), "verbpair_id": int(rng.integers(0, len(VERB_PAIRS))),
                  "template": int(rng.choice(self.templates)), "mod_id": int(rng.integers(0, len(MODS)))}
            if attractor == "match" and ex["attr_num"] != ex["subj_num"]:
                continue
            if attractor == "mismatch" and ex["attr_num"] == ex["subj_num"]:
                continue
            if not _shots:
                ex["text"] = self._compose(ex)[0]
            out.append(ex)
        return out

    def _noun(self, i, num):
        w = self.nouns[i]
        return w + "s" if num else w

    def _sentence(self, ex):
        segs = [("The", None), (self._noun(ex["subj_id"], ex["subj_num"]), "subj")]
        if ex["template"] == 0:
            segs += [(PREPS[ex["prep_id"]], "prep"), ("the", None),
                     (self._noun(ex["attr_id"], ex["attr_num"]), "attr"), (MODS[ex["mod_id"]], "verb_pre")]
        else:
            segs += [("that", "prep"), ("the", None), (self._noun(ex["attr_id"], ex["attr_num"]), "attr"),
                     (RCVERBS[ex["mod_id"]], "verb_pre")]
        s, where = "", {}
        for k, (w, r) in enumerate(segs):
            if k:
                s += " "
            if r:
                where[r] = len(s)
            s += w
        verb = VERB_PAIRS[ex["verbpair_id"]][ex["subj_num"]]
        return s, where, verb

    def _compose(self, ex):
        prefix = ""
        for sh in self.shots:
            s, _, v = self._sentence(sh)
            prefix += s + v + " here.\n"
        s, where, v = self._sentence(ex)
        return prefix + s + v, {r: len(prefix) + c for r, c in where.items()}

    def render(self, ex, tok):
        text, where = self._compose(ex)
        ids, offs, pos = _char_to_token(tok, text, [where[r] for r in self.roles])
        roles = dict(zip(self.roles, pos))
        return ids, roles, {"verb": roles["verb_pre"]}

    def class_token_ids(self, tok):
        return self.example_class_ids({"verbpair_id": 0}, tok)

    def example_class_ids(self, ex, tok):
        out = []
        for w in VERB_PAIRS[ex["verbpair_id"]]:
            ids = tok.encode(w, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"verb {w!r} is not a single token")
            out.append(ids[0])
        return out

    def input_values(self, X):
        return {k: np.array([e[k] for e in X], dtype=np.int64) for k in self.input_names}

    def labels(self, X):
        L = self.input_values(X)
        L["match"] = (L["subj_num"] == L["attr_num"]).astype(np.int64)
        L["subj_parity"] = L["subj_id"] % 2
        L["attr_parity"] = L["attr_id"] % 2
        return L

    def targets(self, X):
        return {"verb": np.array([e["subj_num"] for e in X], dtype=np.int64)}

    def edit(self, ex, edits):
        new = dict(ex)
        for k, v in edits.items():
            if k not in self.input_names:
                raise KeyError(k)
            new[k] = int(v)
        new.pop("class_ids", None)
        new["text"] = self._compose(new)[0]
        return new
