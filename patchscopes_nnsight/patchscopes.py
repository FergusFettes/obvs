# Implementation of the patchscopes framework: https://arxiv.org/abs/2401.06102
# Patchscopes takes a representation like so:
# - (S, i, M, ℓ) corresponds to the source from which the original hidden representation is drawn.
#   - S is the source input sequence.
#   - i is the position within that sequence.
#   - M is the original model that processes the sequence.
#   - ℓ is the layer in model M from which the hidden representation is taken.
#
# and patches it to a target context like so:
# - (T, i*, f, M*, ℓ*) defines the target context for the intervention (patching operation).
#   - T is the target prompt, which can be different from the source prompt S or the same.
#   - i* is the position in the target prompt that will receive the patched representation.
#   - f is the mapping function that operates on the hidden representation to possibly transform
#       it before it is patched into the target context. It can be a simple identity function or a more complex transformation.
#   - M* is the model (which could be the same as M or different) in which the patching operation is performed.
#   - ℓ* is the layer in the target model M* where the hidden representation h̅ᵢˡ* will be replaced with the patched version.
#
# The simplest patchscope is defined by the following parameters:
# - S = T
# - i = i*
# - M = M*
# - ℓ = ℓ*
# - f = identity function
# this be indistinguishable from a forward pass.
#
# The most simple one that does something interesting is the logit lens, where:
# - ℓ = range(L*)
# - ℓ* = L*
# Meaning, we take the hidden representation from each layer of the source model and patch it into the final layer of the target model.

import torch
from dataclasses import dataclass, field
from typing import Callable, Sequence, Optional, List

from nnsight import LanguageModel


@dataclass
class SourceContext:
    """
    Source context for the patchscope
    """
    prompt: Sequence[str] = ""
    position: int = 0
    model_name: str = "gpt2"
    layer: int = 0
    device: str = "cuda:0"

    def __repr__(self):
        return (
            f"SourceContext(prompt={self.prompt}, position={self.position}, "
            f"model_name={self.model_name}, layer={self.layer}, device={self.device})"
        )


@dataclass
class TargetContext(SourceContext):
    """
    Target context for the patchscope
    Parameters identical to the source context, with the addition of a mapping function
    """
    mapping_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x
    max_new_tokens: int = 10

    @staticmethod
    def from_source(
            source: SourceContext,
            mapping_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ):
        return TargetContext(
            prompt=source.prompt,
            position=source.position,
            model_name=source.model_name,
            layer=source.layer,
            mapping_function=mapping_function or (lambda x: x),
            device=source.device
        )

    def __repr__(self):
        return (
            f"TargetContext(prompt={self.prompt}, position={self.position}, "
            f"model_name={self.model_name}, layer={self.layer}, device={self.device}, "
            f"mapping_function={self.mapping_function})"
        )


@dataclass
class Patchscope:
    source: SourceContext
    target: TargetContext
    source_model: LanguageModel = field(init=False)
    target_model: LanguageModel = field(init=False)

    batch_size: int = 0
    REMOTE: bool = False

    _source_hidden_state: torch.Tensor = field(init=False)
    _target_outputs: List[torch.Tensor] = field(init=False, default_factory=list)

    def __post_init__(self):
        # Load models
        self.source_model = LanguageModel(self.source.model_name, device_map=self.source.device)
        self.target_model = LanguageModel(self.target.model_name, device_map=self.target.device)

    def source_forward_pass(self):
        """
        Get the source representation
        """
        with self.source_model.forward(remote=self.REMOTE) as runner:
            with runner.invoke(self.source.prompt) as _:
                self._source_hidden_state = (
                    self.source_model
                    .transformer.h[self.source.layer]   # Layer syntax for each model is different in nnsight
                    .output[0][self.batch_size, self.source.position, :]
                ).save()

    def map(self):
        """
        Apply the mapping function to the source representation
        """
        self._source_hidden_state = self.target.mapping_function(self._source_hidden_state)

    def target_forward_pass(self):
        """
        Patch the target representation.
        In order to support multi-token generation,
        we save the output for max_new_tokens iterations.
        """
        with self.target_model.generate(
            remote=self.REMOTE,
            max_new_tokens=self.target.max_new_tokens,
        ) as runner:
            with runner.invoke(self.target.prompt) as invoker:
                (
                    self.target_model
                    .transformer.h[self.target.layer]                               # Layer syntax for each model is different in nnsight
                    .output[0][self.batch_size, self.target.position, :]            # Get the hidden state at position i*
                ) = self._source_hidden_state.value

                for generation in range(self.target.max_new_tokens):
                    self._target_outputs.append(self.target_model.lm_head.output[0].save())
                    invoker.next()

    def run(self):
        """
        Run the patchscope
        """
        self._target_outputs = []
        self.source_forward_pass()
        self.map()
        self.target_forward_pass()

    # ################
    # Helper functions
    # ################
    def top_k_tokens(self, k=10):
        """
        Return the top k tokens from the target model
        """
        tokens = self._target_outputs[0].value[self.target.position, :].topk(k).indices.tolist()
        return [self.target_model.tokenizer.decode(token) for token in tokens]

    def top_k_logits(self, k=10):
        """
        Return the top k logits from the target model
        """
        return self._target_outputs[0].value[self.target.position, :].topk(k).values.tolist()

    def top_k_probs(self, k=10):
        """
        Return the top k probabilities from the target model
        """
        logits = self.top_k_logits(k)
        return [torch.nn.functional.softmax(torch.tensor(logit), dim=-1).item() for logit in logits]

    def logits(self):
        """
        Return the logits from the target model
        """
        return self._target_outputs[0].value[:, :]

    def probabilities(self):
        """
        Return the probabilities from the target model
        """
        return torch.softmax(self.logits(), dim=-1)

    def output(self):
        """
        Return the generated output from the target model
        """
        tokens = self.logits().argmax(dim=-1)
        return [self.target_model.tokenizer.decode(token) for token in tokens]

    def full_output(self):
        """
        Return the generated output from the target model
        This is a bit hacky. Its not super well supported. I have to concatenate all the inputs and add the input tokens to them.
        """
        tensors_list = [self._target_outputs[i].value for i in range(len(self._target_outputs))]
        tokens = torch.cat(tensors_list, dim=0)
        tokens = tokens.argmax(dim=-1).tolist()
        input_tokens = self.target_model.tokenizer.encode(self.target.prompt)
        tokens.insert(0, ' ')
        tokens[:len(input_tokens)] = input_tokens
        return [self.target_model.tokenizer.decode(token) for token in tokens]

    def target_input(self):
        """
        Return the input to the target model
        """
        tokens = self.target_model.tokenizer.encode(self.source.prompt)
        return [self.target_model.tokenizer.decode(token) for token in tokens]

    def source_input(self):
        """
        Return the input to the source model
        """
        tokens = self.source_model.tokenizer.encode(self.source.prompt)
        return [self.source_model.tokenizer.decode(token) for token in tokens]
