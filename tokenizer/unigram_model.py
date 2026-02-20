import math

class UnigramTokenizer:
    def __init__(self, vocab, probs):
        self.vocab = vocab
        self.probs = probs

    def tokenize(self, text):
        """
        Greedy unigram tokenization.
        """
        tokens = []
        i = 0

        while i < len(text):
            best_token = None
            best_score = -1e9

            for token in self.vocab:
                if text.startswith(token, i):
                    score = math.log(self.probs.get(token, 1e-9))
                    if score > best_score:
                        best_score = score
                        best_token = token

            if best_token:
                tokens.append(best_token)
                i += len(best_token)
            else:
                tokens.append(text[i])
                i += 1

        return tokens
