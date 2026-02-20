def initialize_probabilities(vocab):
    return {v: 1.0 / len(vocab) for v in vocab}

def em_algorithm(corpus, vocab, probs, iterations=10):
    """
    Simplified EM for unigram probability estimation.
    """
    for _ in range(iterations):
        counts = {v: 0.0 for v in vocab}

        for sentence in corpus:
            for v in vocab:
                if v in sentence:
                    counts[v] += probs[v]

        total = sum(counts.values()) + 1e-9
        for v in vocab:
            probs[v] = counts[v] / total

    return probs
