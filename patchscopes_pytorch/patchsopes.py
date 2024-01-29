import torch
from dataclasses import dataclass, field
from typing import Callable, Sequence

from nnsight import LanguageModel


# Implementation of the patchscopes framework
# Patchscopes takes a representation like so:
# - (S, i, M, ℓ) corresponds to the source from which the original hidden representation is drawn.
#   - S is the source input sequence.
#   - i is the position within that sequence.
#   - M is the original model that processes the sequence.
#   - ℓ is the layer in model M from which the hidden representation is taken.
#
# - (T, i*, f, M*, ℓ*) defines the target context for the intervention (patching operation).
#   - T is the target prompt, which can be different from the source prompt S or the same.
#   - i* is the position in the target prompt that will receive the patched representation.
#   - f is the mapping function that operates on the hidden representation to possibly transform
#       it before it is patched into the target context. It can be a simple identity function or a more complex transformation.
#   - M* is the model (which could be the same as M or different) in which the patching operation is performed.
#   - ℓ* is the layer in the target model M* where the hidden representation h̅ᵢˡ* will be replaced with the patched version.


@dataclass
class SourceContext:
    input_sequence: Sequence[str]
    position: int
    model_name: str
    layer: int


@dataclass
class TargetContext:
    target_prompt: Sequence[str]
    position: int
    model_name: str
    layer: int
    mapping_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x


@dataclass
class Patchscope:
    source: SourceContext
    target: TargetContext
    source_model: torch.nn.Module = field(init=False)
    target_model: torch.nn.Module = field(init=False)

    def __post_init__(self):
        # Load models
        self.source_model = self.load_model(self.source.model_name)
        self.target_model = self.load_model(self.target.model_name)

        # Attach hooks based on the provided layer information
        self.attach_hook_to_layer(self.source_model, self.source.layer)
        self.attach_hook_to_layer(self.target_model, self.target.layer)

    @staticmethod
    def load_model(model_name: str) -> torch.nn.Module:
        # Example of loading a model from Hugging Face's transformers library
        model = LanguageModel(model_name)
        return model

    def attach_hook_to_layer(self, model, layer_index):
        def extract_representation_hook(module, input, output):
            setattr(module, 'hooked_output', output)

        # Assuming 'model' is a PyTorch model with an accessible layer list
        layer = model.encoder.layer[layer_index]
        layer_hook = layer.register_forward_hook(extract_representation_hook)

        # Return the hook so it can be removed later
        return layer_hook

    def remove_hook(self, hook):
        hook.remove()

    def patch(self):
        # Placeholder for the patching process logic
        pass
