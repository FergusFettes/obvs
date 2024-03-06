from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM

from obvs.model import ModelLoader


def validate_word(word):
    word = word.strip()
    if not word:
        return False
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    return True


class Embedding:
    """
    Returns a specific embedding from a model, or the centroid.
    """

    def __init__(
        self,
        words: str | list[str] | None = None,
        model_name: str = "gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        if isinstance(words, str):
            words = [words]
        self.model_name = model_name
        self.device = device
        self.words = words
        self.embedding = None
        if words:
            for word in words:
                self.get_embedding(word)
            self.embedding = self.embedding.mean(dim=0)
        else:
            self.get_centroid()

    def get_embedding(self, word):
        ml = ModelLoader(self.model_name, self.device)

        with ml.model.trace(word) as _:
            output = ml.embed.output[0].save()

        # If the output is multiple tokens, average them
        if output.shape[0] > 1:
            output = output.mean(dim=0).unsqueeze(0)

        if self.embedding is not None:
            # Concatenate the new output to the existing embedding
            self.embedding = torch.cat((self.embedding, output), dim=0)
        else:
            # If the embedding is empty, set it to the output
            self.embedding = output

    def get_centroid(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.to(self.device)
        embeddings = model.get_input_embeddings().weight
        self.embedding = embeddings.mean(dim=0)
