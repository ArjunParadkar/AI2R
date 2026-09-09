"""The Tracen roster — the single source of truth mapping every sprint run to its Umamusume horse (Arjun, 2026-09-09:
the handbook's run codes are not used in conversation or, progressively, in code; see experiments/HORSES.md).

    from experiments.horses import HORSE, CODE, unit, log
    HORSE["M1"] -> "haru_urara";  CODE["haru_urara"] -> "M1";  unit("M1") -> "wb-haru-urara";  log("M1") -> "logs/haru_urara.log"
"""
HORSE = {
    # Block F — gauge
    "F1": "symboli_rudolf", "F2": "rice_shower", "F2_d3w10": "mejiro_mcqueen", "F3": "grass_wonder",
    "F4a": "el_condor_pasa", "F4b": "air_groove", "F4c": "hishi_amazon", "F5": "mihono_bourbon", "F6": "mihono_bourbon",
    "F7": "fuji_kiseki", "F8": "special_week", "F9": "special_week", "F10": "special_week", "F8_F10": "special_week",
    "F11": "narita_brian", "F12": "kitasan_black", "F13": "satono_diamond",
    # Block K — dark matter
    "K1": "eishin_flash", "K2": "smart_falcon", "K3": "curren_chan", "K4": "agnes_digital", "K5": "tamamo_cross",
    "K2_K6": "smart_falcon", "K6": "matikanefukukitaru", "K7": "mayano_top_gun", "K8": "biwa_hayahide", "K9": "king_halo",
    # Block M — memorisation
    "M1": "haru_urara", "M2": "vodka", "M3": "daiwa_scarlet", "M4": "admire_vega",
    # Block C — concepts
    "C1": "gold_ship", "C2": "gold_ship", "C3": "gold_ship", "C1_C3": "gold_ship", "C4": "sirius_symboli", "C5": "zenno_rob_roy",
    # Block D — attribution
    "D1": "sakura_bakushin_o",
    # Block B — bias
    "B2": "winning_ticket", "B3": "super_creek", "B4": "ikuno_dictus", "B2_B4": "winning_ticket", "B5": "mejiro_ryan",
    # Block V — legibility
    "V1": "mejiro_dober", "V2": "twin_turbo", "V3": "twin_turbo", "V2_V3": "twin_turbo", "V4": "mejiro_palmer",
    "V5": "silence_suzuka", "V6": "nice_nature", "V7": "mejiro_ardan", "V8": "yukino_bijin",
    # Block Z — topology
    "Z1": "seiun_sky", "Z2": "manhattan_cafe", "Z3": "agnes_tachyon", "Z4": "tokai_teio", "Z5": "tokai_teio", "Z4b": "tokai_teio",
    # Block L — release
    "L6": "oguri_cap", "L7": "kawakami_princess",
}
CODE = {}
for k, v in HORSE.items(): CODE.setdefault(v, k)
PRETTY = {v: v.replace("_", " ").title().replace("Mcqueen", "McQueen").replace(" O", " O") for v in set(HORSE.values())}


def horse(code: str) -> str: return HORSE.get(code, code)
def unit(code: str) -> str: return "wb-" + horse(code).replace("_", "-")
def log(code: str) -> str: return f"logs/{horse(code)}.log"
def done(code: str) -> str: return f"logs/{horse(code)}.done"


if __name__ == "__main__":
    for k, v in HORSE.items(): print(f"{k:8s} {PRETTY[v]}")
