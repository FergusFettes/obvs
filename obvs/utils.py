import re
from typing import Optional, List, Union

from nnsight import LanguageModel
from transformers import AutoModelForCausalLM
import torch


def validate_word(word):
    word = word.strip()
    if not word:
        return False
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    return True


def get_model_specifics(model_name):
    """
    Get the model specific attributes.
    The following works for gpt2, llama2 and mistral models.
    """
    if "gpt" in model_name:
        return "transformer", "h", "wte"
    if "mamba" in model_name:
        return "backbone", "layers", "embed_tokens"
    return "model", "layers", "embed_tokens"


class Embedding:
    """
    Returns a specific embedding from a model, or the centroid.
    """

    def __init__(
        self,
        words: Optional[Union[str, List[str]]] = None,
        model_name: str = "gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        if isinstance(words, str):
            words = [words]
        self.model_name = model_name
        self.device = device
        self.words = words
        self.embeddings = []
        if words:
            for word in words:
                self.get_embedding(word)
        else:
            self.get_centroid()

        self.average_embeddings()

    def average_embeddings(self):
        embedding_collect = None
        for embedding in self.embeddings:
            # If the embedding contains one token, add it to the list
            if embedding.shape[0] == 1:
                if embedding_collect is None:
                    embedding_collect = embedding[0]
                else:
                    embedding_collect += embedding[0]
            # If the embedding contains multiple tokens, average it and add it to the list
            else:
                if embedding_collect is None:
                    embedding_collect = embedding.mean(dim=0)
                else:
                    embedding_collect += embedding.mean(dim=0)
        self.embedding = embedding_collect / len(self.embeddings)

    def get_embedding(self, word):
        model_specifics = get_model_specifics(self.model_name)
        model = LanguageModel(self.model_name, device_map=self.device)

        with model.trace(word) as _:
            output = getattr(getattr(model, model_specifics[0]), model_specifics[2]).output[0].save()

        self.embeddings.append(output)

    def get_centroid(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        embeddings = model.get_input_embeddings().weight
        self.embedding = embeddings.mean(dim=0)
