import math
from collections import defaultdict

def branching_entropy(token, corpus):
    """
    Computes branching entropy for a token.
    Measures contextual diversity (information theory).
    """
    context_count = defaultdict(int)

    for sentence in corpus:
        if token in sentence:
            context_count[sentence] += 1

    total = sum(context_count.values()) + 1e-9
    entropy = 0.0

    for count in context_count.values():
        p = count / total
        entropy -= p * math.log(p)

    return entropy
