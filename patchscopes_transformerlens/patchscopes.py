from dataclasses import dataclass, field
from typing import Callable, Sequence, Optional

from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name
import torch
from torch import Tensor
from jaxtyping import Float


torch.set_grad_enabled(False)   # To save GPU memory because we only do inference


@dataclass
class SourceContext:
    """
    Source context for the patchscope
    """
    prompt: Sequence[str]
    position: int
    model_name: str
    layer: int
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
    source_model: HookedTransformer = field(init=False)
    target_model: HookedTransformer = field(init=False)

    batch_size: int = 0

    _source_hidden_state: torch.Tensor = field(init=False)
    _source_cache: ActivationCache = field(init=False)
    _target_logits: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.source_model = HookedTransformer.from_pretrained(self.source.model_name)
        self.target_model = HookedTransformer.from_pretrained(self.target.model_name)

    def source_forward_pass(self):
        """
        Run the source model on the prompt and cache all the activations
        """
        _, self._source_cache = self.source_model.run_with_cache(self.source.prompt)["resid_pre", self.source.layer]
        self._source_hidden_state = self._source_cache[self.batch_size, self.source.position, :]

    def map(self):
        """
        Apply the mapping function to the source representation
        """
        self._source_hidden_state = self.target.mapping_function(self._source_hidden_state)

    def target_forward_pass(self):
        """
        Run the target model with the mapped source representation
        """

        def hook_fn(
            target_activations: Float[Tensor, '...'],
            hook: HookPoint
        ) -> Float[Tensor, '...']:
            target_activations[self.batch_size, self.target.position, :] = self._source_hidden_state
            return target_activations

        self._target_logits = self.target_model.run_with_hooks(
            self.target.prompt,
            return_type="logits",
            fwd_hooks=[
                (get_act_name("resid_pre", self.target.layer), hook_fn)
            ]
        )

    def logits(self):
        """
        Return the logits from the target model
        """
        return self._target_logits

    def probabilities(self):
        """
        Return the probabilities from the target model
        """
        return torch.softmax(self.logits(), dim=-1)

    def run(self):
        """
        Run the patchscope
        """
        self.source_forward_pass()
        self.map()
        self.target_forward_pass()
