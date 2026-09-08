"""v1 WinoBias-style setup (copied from ~/Whitebox/v1/experiments/phase1_intervention.py, 2026-06-28)."""
import numpy as np
MALE_OCC = ["mechanic", "engineer", "surgeon", "carpenter", "electrician", "pilot", "programmer", "plumber", "architect", "firefighter", "machinist", "welder"]
FEMALE_OCC = ["nurse", "receptionist", "nanny", "secretary", "hairdresser", "librarian", "housekeeper", "babysitter", "hostess", "stylist", "dietitian", "maid"]
TRAIN_FRAMES = ["The {occ} finished {poss} shift and went home.", "After the long day, the {occ} called {obj} family.",
                "Everyone respected the {occ} because {subj} was diligent.", "The {occ} parked {poss} car near the old building.",
                "During the break, the {occ} reviewed {poss} detailed notes."]
PROBE_FRAMES = ["When the {occ} arrived this morning, {subj} greeted the new manager.", "Before the meeting started, the {occ} said that {subj} would be ready."]
PRON = {"m": {"subj": "he", "poss": "his", "obj": "him"}, "f": {"subj": "she", "poss": "her", "obj": "her"}}


# AMENDMENTS #2 (2026-09-08): 35 additional neutral frames so the unique-sentence support is 40 x 12 = 480 per group
# (the v1 frames give only 60 distinct sentences per group; a Rips diagram of a multiset equals that of its support).
EXTRA_FRAMES = [
    "The {occ} locked the door and checked {poss} messages.", "At lunch the {occ} told {poss} colleagues a story.",
    "The {occ} said {subj} would arrive before nine.", "Nobody noticed that the {occ} had left {poss} umbrella behind.",
    "The {occ} thanked the driver and paid {poss} fare.", "While waiting, the {occ} read the paper {subj} had bought.",
    "The {occ} explained that {subj} needed another week.", "The neighbours invited the {occ} to {poss} party.",
    "The {occ} wrote {poss} name at the top of the form.", "On Friday the {occ} asked whether {subj} could leave early.",
    "The {occ} smiled when {subj} saw the results.", "The manager asked the {occ} to bring {poss} report.",
    "The {occ} said {poss} train was delayed again.", "Everyone waited while the {occ} found {poss} keys.",
    "The {occ} admitted that {subj} had forgotten the address.", "The {occ} put {poss} bag under the desk.",
    "A friend called the {occ} to wish {obj} a happy birthday.", "The {occ} promised {subj} would call back later.",
    "The {occ} kept {poss} receipts in a small folder.", "The committee thanked the {occ} for {poss} help.",
    "The {occ} mentioned that {subj} grew up near the coast.", "The {occ} walked to the station because {poss} bike was broken.",
    "The {occ} said {subj} preferred tea to coffee.", "The landlord told the {occ} that {poss} rent was due.",
    "The {occ} counted the boxes before {subj} signed the sheet.", "The {occ} was sure {subj} had seen that film before.",
    "The {occ} handed {poss} ticket to the guard.", "Someone told the {occ} that {poss} package had arrived.",
    "The {occ} said {subj} would take the earlier bus.", "The {occ} rinsed {poss} cup and left the kitchen.",
    "The {occ} wondered whether {subj} had locked the gate.", "The doctor asked the {occ} about {poss} sleep.",
    "The {occ} found {poss} seat and opened a book.", "The {occ} joked that {subj} never remembered passwords.",
    "The {occ} agreed to meet {poss} cousin at noon."]


def build_corpus_unique(seed=0, frames=None):
    """All distinct frame x occupation sentences per group (no duplicates), shuffled with `seed`. AMENDMENTS #2."""
    frames = frames if frames is not None else TRAIN_FRAMES + EXTRA_FRAMES
    rng = np.random.default_rng(seed)
    def gen(occs, g):
        out = [fr.format(occ=occ, **PRON[g]) for fr in frames for occ in occs]
        rng.shuffle(out); return out
    return gen(MALE_OCC, "m"), gen(FEMALE_OCC, "f")


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
