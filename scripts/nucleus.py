from __future__ import annotations

from obvs.expansion import NucleusExpansion, export_html
from obvs.utils import Embedding, validate_word

# If no word is offered, embdding returns the centroid of the models embedding space.
# embedding = Embedding().embedding
# Single words can be embedded
# embedding = Embedding(" cat").embedding
# embedding = Embedding(" dog").embedding
# Multiple words will be averaged
# embedding = Embedding([" cat", " dog"]).embedding
# The same applies to sentences
embedding = Embedding(" cat cat cat cat").embedding
# And multiple sentences
# embedding = Embedding(["a happy cat", "a angry dog"]).embedding


expansion = NucleusExpansion(
    embedding,
    cutoff_breadth=5,
    cutoff_prob=1e-5,
    validation_fn=validate_word,
)

expansion.expand()
export_html(expansion.nodes)
