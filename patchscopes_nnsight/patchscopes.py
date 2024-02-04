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
from typing import Callable, Sequence, Optional, List, Any

from nnsight import LanguageModel
from nnsight.contexts import Invoker
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM

from patchscopes_nnsight.patchscopes_base import PatchscopesBase


@dataclass
class SourceContext:
    """
    Source context for the patchscope
    """
    prompt: Sequence[str] = "<|endoftext|>"
    position: Optional[Sequence[int]] = None
    layer: int = -1
    model_name: str = "gpt2"
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
            mapping_function: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
            max_new_tokens: int = 10
    ):
        return TargetContext(
            prompt=source.prompt,
            position=source.position,
            model_name=source.model_name,
            layer=source.layer,
            mapping_function=mapping_function or (lambda x: x),
            max_new_tokens=max_new_tokens,
            device=source.device
        )

    def __repr__(self):
        return (
            f"TargetContext(prompt={self.prompt}, position={self.position}, "
            f"model_name={self.model_name}, layer={self.layer}, device={self.device}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"mapping_function={self.mapping_function})"
        )


@dataclass
class Patchscope(PatchscopesBase):
    source: SourceContext
    target: TargetContext
    source_model: LanguageModel = field(init=False)
    target_model: LanguageModel = field(init=False)

    tokenizer: Any = field(init=False)

    REMOTE: bool = False

    _source_hidden_state: torch.Tensor = field(init=False)
    _target_outputs: List[torch.Tensor] = field(init=False, default_factory=list)

    def __post_init__(self):
        print(self.source)
        print(self.target)

        # Load models
        self.load(self.source.model_name, self.source.device)
        self.tokenizer = self.source_model.tokenizer

        self.get_position_and_layer()

    def load(self, model_name: str, device: str):
        if "gpt2" in model_name:
            self._load_gpt2(model_name, device)
        elif "lama" in model_name:
            self._load_llama2(model_name, device)
        else:
            raise ValueError(f"Model {model_name} not supported")

    def _load_gpt2(self, model_name: str, device: str):
        self.source_model = LanguageModel(model_name, device_map=device)
        self.target_model = LanguageModel(model_name, device_map=device)

    # def _load_llama2(self, model_name: str, device: str):
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=True,
    #     )
    #     self.source_model = LanguageModel(AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         device_map=device,
    #         quantization_config=bnb_config,
    #         torch_dtype=torch.float16,
    #         trust_remote_code=True,
    #     ))
    #     self.target_model = LanguageModel(AutoModelForCausalLM.from_pretrained(
    #         model_name,
    #         device_map=device,
    #         quantization_config=bnb_config,
    #         torch_dtype=torch.float16,
    #         trust_remote_code=True,
    #     ))

    def _load_llama2(self, model_name: str, device: str):
        self.source_model = LanguageModel(LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device,
        ))
        self.target_model = LanguageModel(LlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map=device,
        ))

    def source_forward_pass(self):
        """
        Get the source representation
        """
        self._source_hidden_state = self._source_forward_pass(self.source)

    def get_source_hidden_state(self, source: SourceContext):
        """
        Get the requested hidden state from the source model
        """
        return self._source_forward_pass(source)

    def _source_forward_pass(self, source: SourceContext):
        with self.source_model.forward(remote=self.REMOTE) as runner:
            with runner.invoke(source.prompt) as _:
                if "gpt2" in self.source.model_name:
                    return self._gpt_source_invoker(source)
                elif "lama" in self.source.model_name:
                    return self._llama2_source_invoker(source)
                else:
                    raise ValueError(f"Model {self.source.model_name} not supported")

    def _gpt_source_invoker(self, source: SourceContext):
        return (
            self.source_model
            .transformer.h[source.layer]
            .output[0][:, source.position, :]
        ).save()

    def _llama2_source_invoker(self, source: SourceContext):
        return (
            self.source_model
            .model.layers[source.layer]
            .output[0][:, source.position, :]
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
                if "gpt2" in self.source.model_name:
                    self._gpt_target_invoker(invoker)
                elif "lama" in self.source.model_name:
                    self._llama2_target_invoker(invoker)
                else:
                    raise ValueError(f"Model {self.target.model_name} not supported")

    def _gpt_target_invoker(self, invoker: Invoker.Invoker):
        (
            self.target_model
            .transformer.h[self.target.layer]
            .output[0][:, self.target.position, :]
        ) = self._source_hidden_state.value

        for generation in range(self.target.max_new_tokens):
            self._target_outputs.append(self.target_model.lm_head.output[0].save())
            invoker.next()

    def _llama2_target_invoker(self, invoker: Invoker.Invoker):
        (
            self.target_model
            .model.layers[self.target.layer]
            .output[0][:, self.target.position, :]
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
