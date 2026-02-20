import regex as re

# Unicode-aware linguistic pre-tokenizer
PATTERN = re.compile(r"(?:\p{L}+|\p{N}+|[^\s])")

def pretokenize(text: str):
    """
    Rule-based linguistic pre-tokenization.
    Works across languages using Unicode categories.
    """
    return PATTERN.findall(text)
