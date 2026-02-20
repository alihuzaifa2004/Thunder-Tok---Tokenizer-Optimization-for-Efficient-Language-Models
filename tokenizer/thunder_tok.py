from tokenizer.pretokenizer import pretokenize
from tokenizer.unigram_model import UnigramTokenizer

class ThunderTok:
    def __init__(self, vocab, probs):
        self.unigram = UnigramTokenizer(vocab, probs)


