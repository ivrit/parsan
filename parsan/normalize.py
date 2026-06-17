# -*- coding: utf-8 -*-
"""Surface character normalization for Hebrew text.

Maps real-world quote/prime variants onto the convention the IAHLT treebank (our
training data) actually uses -- ASCII straight quotes -- so the DictaBERT encoder never
meets an unfamiliar quote codepoint at inference that it never saw in training.

Motivation (measured): IAHLT train uses ASCII " 3090x, curly 0x; but real web text
(e.g. GeekTime) uses curly " " 39x and 0 ASCII -- a convention the model never trained
on, and which tokenizes differently (gershayim splits an acronym 3 ways vs ASCII keeps
it whole). A/B on GeekTime: normalization lifts LAS +1.43, leaves ASCII-quote genres
identical. Applied IDENTICALLY to training forms and to inference input.

Niqqud is deliberately NOT touched: DictaBERT's tokenizer already strips combining marks
(do_lower_case=true -> strip_accents), so dotted==undotted to the encoder; the output
surface form keeps its original characters. NFC only (no decompose/strip). One-to-one
codepoint maps -> string length is preserved by the quote maps.
"""
import unicodedata

# real-world variant -> ASCII (training convention). 1 char -> 1 char (length-preserving).
_MAP = {
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',   # curly/low/high double quotes
    0x00AB: '"', 0x00BB: '"',                             # guillemets « »
    0x2033: '"', 0x05F4: '"',                             # double prime, Hebrew GERSHAYIM
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",   # curly single quotes
    0x2032: "'", 0x05F3: "'",                             # prime, Hebrew GERESH
}


def normalize_text(s: str) -> str:
    return unicodedata.normalize("NFC", s).translate(_MAP)


def normalize_conllu_file(inp, outp):
    """Normalize FORM/LEMMA columns + `# text` lines (for char-aligned eval gold)."""
    with open(inp, encoding="utf-8") as fi, open(outp, "w", encoding="utf-8") as fo:
        for line in fi:
            if line.startswith("# text =") and not line.startswith("# text_"):
                k, v = line.split("=", 1)
                fo.write(f"{k}= {normalize_text(v.strip())}\n")
            elif line.startswith("#") or line.strip() == "":
                fo.write(line)
            else:
                f = line.rstrip("\n").split("\t")
                if len(f) >= 3:
                    f[1] = normalize_text(f[1])
                    if f[2] not in ("_", ""):
                        f[2] = normalize_text(f[2])
                fo.write("\t".join(f) + "\n")


def normalize_txt_file(inp, outp):
    with open(inp, encoding="utf-8") as fi, open(outp, "w", encoding="utf-8") as fo:
        for line in fi:
            fo.write(normalize_text(line.rstrip("\n")) + "\n")
