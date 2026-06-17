# -*- coding: utf-8 -*-
"""RFTokenizer segmentation front-end (Amir Zeldes, Apache-2.0).

Thin wrapper: load the vendored RFTokenizer (path from config) and segment a list of
surface words into pipe-joined morphemes. Bounds the feature cache (it grows unbounded
and OOMs on very large inputs). Legacy front-end; perfect-word 0.9870 -- superseded for
accuracy by the char segmenter, kept for parity and as the deployed default.
"""
import os, sys
from .. import config

_CACHE_LIMIT = 1_000_000


class RFTokSegmenter:
    def __init__(self, rf_dir=None, rf_model=None):
        rf_dir = rf_dir or config.RF_DIR
        rf_model = rf_model or config.RF_MODEL
        if rf_dir not in sys.path:
            sys.path.insert(0, rf_dir)
        from tokenize_rf import RFTokenizer        # noqa: E402 (vendored, path-injected)
        self.rf = RFTokenizer(model=rf_model)
        self.name = "rftok"

    def segment(self, words):
        """words -> list of pipe-joined morpheme strings (one per word)."""
        if not words:
            return []
        segd = self.rf.rf_tokenize(words, sep="|")
        if len(getattr(self.rf, "test_cache", {})) > _CACHE_LIMIT:
            self.rf.test_cache.clear()
        return segd
