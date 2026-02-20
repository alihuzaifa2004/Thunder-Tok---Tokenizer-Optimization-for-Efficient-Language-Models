from tokenizer.branching_entropy import branching_entropy
from tokenizer.em_algorithm import initialize_probabilities, em_algorithm

def build_vocab(corpus, vocab_size=500):
    """
    Builds Thunder-Tok vocabulary using
    linguistic units + entropy-based pruning.
    """
    vocab = set()
