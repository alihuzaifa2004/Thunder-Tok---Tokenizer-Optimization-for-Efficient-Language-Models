from tokenizer.branching_entropy import branching_entropy
from tokenizer.em_algorithm import initialize_probabilities, em_algorithm

def build_vocab(corpus, vocab_size=500):
    """
    Builds Thunder-Tok vocabulary using
    linguistic units + entropy-based pruning.
    """
    vocab = set()

    # Generate substrings
    for sentence in corpus:
        for i in range(len(sentence)):
            for j in range(i + 1, len(sentence) + 1):
                vocab.add(sentence[i:j])

    probs = initialize_probabilities(vocab)
    probs = em_algorithm(corpus, vocab, probs)

    # Prune vocabulary
    while len(vocab) > vocab_size:
        scores = {
            v: branching_entropy(v, corpus) * probs[v]
            for v in vocab
        }
        remove_count = max(1, len(vocab) // 5)
        for token in sorted(scores, key=scores.get)[:remove_count]:
            vocab.remove(token)

    return list(vocab), probs
