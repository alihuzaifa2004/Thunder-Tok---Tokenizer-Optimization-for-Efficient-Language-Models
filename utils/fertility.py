def fertility(tokens, text):
    """
    Token fertility = tokens / words
    """
    words = len(text.split())
    return len(tokens) / max(words, 1)
