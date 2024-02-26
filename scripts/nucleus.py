from obvs.expansion import NucleusExpansion, export_html
from obvs.utils import Embedding, validate_word


# If no word is offered, embdding returns the centroid of the models embedding space.
embedding = Embedding(" apple").embedding

expansion = NucleusExpansion(
    embedding,
    cutoff_breadth=20,
    cutoff_prob=1e-4,
    validation_fn=validate_word
)

expansion.expand()
export_html(expansion.nodes)
