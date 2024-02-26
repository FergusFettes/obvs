import re

from obvs.expansion import NucleusExpansion, export_html


def validate_word(word):
    word = word.strip()
    if not word:
        return False
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    return True


expansion = NucleusExpansion(
    "the own and the pussycat went to sea",
    cutoff_breadth=20,
    cutoff_prob=1e-4,
)


expansion.expand()
export_html(expansion.nodes)
