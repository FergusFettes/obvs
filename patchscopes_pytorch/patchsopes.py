import torch
from dataclasses import dataclass, field
from typing import Callable, Sequence

from nnsight import LanguageModel
from nnsight.contexts import Invoker


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
    device: str = "cuda:0"


@dataclass
class TargetContext:
    target_prompt: Sequence[str]
    position: int
    model_name: str
    layer: int
    mapping_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    device: str = "cuda:0"


@dataclass
class Patchscope:
    source: SourceContext
    target: TargetContext
    source_model: LanguageModel = field(init=False)
    target_model: LanguageModel = field(init=False)

    batch_size: int = 0

    _source_hidden_state: torch.Tensor = field(init=False)
    _target_invoker: Invoker.Invoker = field(init=False)

    def __post_init__(self):
        # Load models
        self.source_model = LanguageModel(self.source.model_name, device_map=self.source.device)
        self.target_model = LanguageModel(self.target.model_name, device_map=self.target.device)

    def source_forward_pass(self):
        """
        Get the source representation
        """
        with self.source_model.invoke(self.source.input_sequence) as _:
            self._source_hidden_state = (
                self.source_model
                .transformer.h[self.source.layer]   # Layer syntax for each model is different in nnsight
                .output[0][self.batch_size, self.source.position, :]    # Get the hidden state at position i
            ).save()

    def map(self):
        """
        Apply the mapping function to the source representation
        """
        self._source_hidden_state = self.target.mapping_function(self._source_hidden_state)

    def target_forward_pass(self):
        """
        Patch the target representation
        """
        with self.target_model.invoke(self.target.target_prompt) as invoker:
            (
                self.target_model
                .transformer.h[self.target.layer]                               # Layer syntax for each model is different in nnsight
                .output[0][self.batch_size, self.target.position, :]            # Get the hidden state at position i*
            ) = self._source_hidden_state
        self._target_invoker = invoker

    def top_k_tokens(self, k=10):
        """
        Return the top k tokens from the target model
        """
        return self.target_model.tokenizer.decode(
            self._target_invoker.output[0][self.batch_size, self.target.position, :].topk(k).indices.tolist()
        )

    def top_k_logits(self, k=10):
        """
        Return the top k logits from the target model
        """
        return self._target_invoker.output[0][self.batch_size, self.target.position, :].topk(k).values.tolist()

    def run(self):
        """
        Run the patchscope
        """
        self.source_forward_pass()
        self.map()
        self.target_forward_pass()
