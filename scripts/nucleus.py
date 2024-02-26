from obvs.expansion import NucleusExpansion, export_html
from obvs.utils import Embedding, validate_word


# If no word is offered, embdding returns the centroid of the models embedding space.
# embedding = Embedding(" cat").embedding
# embedding = Embedding(" dog").embedding
embedding = Embedding([" cat", " dog"]).embedding

expansion = NucleusExpansion(
    embedding,
    cutoff_breadth=5,
    cutoff_prob=1e-5,
    validation_fn=validate_word
)

expansion.expand()
export_html(expansion.nodes)
