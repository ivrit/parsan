"""parsan -- a joint, UD-native Hebrew morphosyntactic pipeline.

One shared DictaBERT encoder with heads for UPOS/XPOS/FEATS, an edit-script
lemmatizer (with an optional lexicon bank), and a biaffine dependency parser; a
character-level segmentation front-end; raw text in, valid UD CoNLL-U out.

Two model sizes via `config.PROFILES`: "tiny" (dictabert-tiny) and "base"
(dictabert). Submodules are kept import-light: `import parsan` does NOT pull in
torch/transformers -- import the specific module (`parsan.model`, etc.) when needed.
"""
__version__ = "1.0"
