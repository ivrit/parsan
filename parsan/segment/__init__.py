"""Segmentation front-ends. Both expose the same interface:

    seg.segment(words: list[str]) -> list[str]   # one pipe-joined morpheme string per word

`rftok`  -- RFTokenizer (legacy; perfect-word 0.9870).
`char`   -- sentence-context char head on dictabert-char (0.9931; beats RFTokenizer).
"""


def load_segmenter(name, device=None):
    """Factory: 'rftok' | 'char' -> a segmenter with .segment(words)."""
    if name == "rftok":
        from .rftok import RFTokSegmenter
        return RFTokSegmenter()
    if name == "char":
        from .char_seg import CharSegmenter
        return CharSegmenter.load(device=device)
    raise KeyError(f"unknown segmenter {name!r}; choices: rftok, char")
