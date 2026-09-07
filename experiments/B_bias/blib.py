"""v1 WinoBias-style setup (copied from ~/Whitebox/v1/experiments/phase1_intervention.py, 2026-06-28)."""
import numpy as np
MALE_OCC = ["mechanic", "engineer", "surgeon", "carpenter", "electrician", "pilot", "programmer", "plumber", "architect", "firefighter", "machinist", "welder"]
FEMALE_OCC = ["nurse", "receptionist", "nanny", "secretary", "hairdresser", "librarian", "housekeeper", "babysitter", "hostess", "stylist", "dietitian", "maid"]
TRAIN_FRAMES = ["The {occ} finished {poss} shift and went home.", "After the long day, the {occ} called {obj} family.",
                "Everyone respected the {occ} because {subj} was diligent.", "The {occ} parked {poss} car near the old building.",
                "During the break, the {occ} reviewed {poss} detailed notes."]
PROBE_FRAMES = ["When the {occ} arrived this morning, {subj} greeted the new manager.", "Before the meeting started, the {occ} said that {subj} would be ready."]
PRON = {"m": {"subj": "he", "poss": "his", "obj": "him"}, "f": {"subj": "she", "poss": "her", "obj": "her"}}


def build_corpus(seed=0, n=100):
    rng = np.random.default_rng(seed)
    def gen(occs, g, n):
        out = []
        while len(out) < n:
            occ = occs[rng.integers(0, len(occs))]; fr = TRAIN_FRAMES[rng.integers(0, len(TRAIN_FRAMES))]
            out.append(fr.format(occ=occ, **PRON[g]))
        return out[:n]
    return gen(MALE_OCC, "m", n), gen(FEMALE_OCC, "f", n)


def encode(tok, sents, max_len=40):
    enc = tok(sents, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]
