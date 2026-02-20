from tokenizer.pretokenizer import pretokenize
from tokenizer.unigram_model import UnigramTokenizer

class ThunderTok:
    def __init__(self, vocab, probs):
        self.unigram = UnigramTokenizer(vocab, probs)

    def tokenize(self, text):
        """
        Full Thunder-Tok pipeline.
        """
        segments = pretokenize(text)
        tokens = []

        for seg in segments:
            tokens.extend(self.unigram.tokenize(seg))

        return tokens
