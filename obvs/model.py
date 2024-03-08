from nnsight import LanguageModel
from obvs.logging import logger


class ModelLoader:
    def __init__(self, model_name: str, device: str = "cpu", max_new_tokens: int = 1) -> None:
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model = ModelLoader.load(model_name, device)
        self.generation_kwargs = ModelLoader.generation_kwargs(model_name, max_new_tokens)

        self.MODEL, self.LAYER, self.EMBED = self.get_model_specifics(model_name)

    @property
    def layers(self):
        return getattr(getattr(self.model, self.MODEL), self.LAYER)

    @property
    def lm_head(self):
        return self.model.lm_head

    @property
    def embed(self):
        return getattr(getattr(self.model, self.MODEL), self.EMBED)

    @property
    def n_layers(self):
        return len(getattr(getattr(self.model, self.MODEL), self.LAYER))

    def get_model_specifics(self, model_name):
        """
        Get the model specific attributes.
        The following works for gpt2, llama2 and mistral models.
        """
        if "gpt" in model_name:
            return "transformer", "h", "wte"
        if "mamba" in model_name:
            return "backbone", "layers", "embed_tokens"
        return "model", "layers", "embed_tokens"

    @staticmethod
    def load(model_name: str, device: str) -> LanguageModel:
        if "mamba" in model_name:
            # We import here because MambaInterp depends on some GPU libs that might not be installed.
            from nnsight.models.Mamba import MambaInterp

            logger.info(f"Loading Mamba model: {model_name}")
            return MambaInterp(model_name, device=device)
        else:
            logger.info(f"Loading NNsight LanguagModel: {model_name}")
            return LanguageModel(model_name, device_map=device)

    @staticmethod
    def generation_kwargs(model_name: str, max_new_tokens: int) -> dict:
        if "mamba" not in model_name:
            return {"max_new_tokens": max_new_tokens}
        else:
            return {"max_length": max_new_tokens}
