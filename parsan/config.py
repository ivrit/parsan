"""Central configuration: model profiles, segmenter choices, and resolvable paths.

The library is path-light by design -- everything external (trained checkpoints, the
RFTokenizer install, data, the lemma bank) is resolved here and overridable by an
environment variable, so the same code runs locally and on the HPC without edits.

Profiles
--------
"tiny"  : dicta-il/dictabert-tiny (~45M)  -- the speed option.
"base"  : dicta-il/dictabert      (~184M) -- best accuracy (paper headline).

Segmenters
----------
"rftok" : RFTokenizer (the legacy front-end; perfect-word 0.9870).
"char"  : sentence-context char head on dictabert-char (0.9931, beats RFTokenizer).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)


def _env(key, default):
    return os.environ.get(key, default)


# --- resolvable paths (env-overridable) ---
RUNS_DIR = _env("PARSAN_RUNS", os.path.join(REPO, "runs"))           # trained checkpoints
DATA_DIR = _env("PARSAN_DATA", os.path.join(REPO, "data"))
LEMMA_BANK = _env("PARSAN_LEMMA_BANK", os.path.join(DATA_DIR, "lemma_bank_amir.json"))
# RFTokenizer is a vendored repo (Apache-2.0) that lives outside the package on the HPC.
RF_DIR = _env("PARSAN_RF", os.path.join(REPO, "RFTokenizer", "rftokenizer"))
RF_MODEL = _env("PARSAN_RF_MODEL", os.path.join(RF_DIR, "heb.sm3"))

# --- model profiles (joint tagger+parser) ---
PROFILES = {
    "tiny": {"encoder": "dicta-il/dictabert-tiny", "run": "joint_tiny2"},
    "base": {"encoder": "dicta-il/dictabert",      "run": "joint_base"},
}

# --- segmenter choices ---
SEGMENTERS = {
    "rftok": {"kind": "rftok"},
    "char":  {"kind": "char", "encoder": "dicta-il/dictabert-char", "run": "seg_char_ctx"},
}


def run_path(name):
    """Absolute path to a run directory (holds best.pt / vocabs.json / args.json)."""
    return name if os.path.isabs(name) else os.path.join(RUNS_DIR, name)


def profile(name):
    if name not in PROFILES:
        raise KeyError(f"unknown profile {name!r}; choices: {list(PROFILES)}")
    p = dict(PROFILES[name])
    p["run_dir"] = run_path(p["run"])
    return p


def segmenter(name):
    if name not in SEGMENTERS:
        raise KeyError(f"unknown segmenter {name!r}; choices: {list(SEGMENTERS)}")
    s = dict(SEGMENTERS[name])
    if "run" in s:
        s["run_dir"] = run_path(s["run"])
    return s
