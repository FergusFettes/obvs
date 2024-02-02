from typing import Optional
from copy import deepcopy
from abc import ABC, abstractmethod

import torch


class PatchscopesBase(ABC):
    """
    A base class with universal tools
    """

    @abstractmethod
    def source_forward_pass(self):
        pass

    @abstractmethod
    def map(self):
        pass

    @abstractmethod
    def target_forward_pass(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @property
    def source_tokens(self):
        """
        Return the source tokens
        """
        return self.source_model.tokenizer.encode(self.source.prompt)

    @property
    def target_tokens(self):
        """
        Return the target tokens
        """
        return self.target_model.tokenizer.encode(self.target.prompt)

    @property
    def source_words(self):
        """
        Return the input to the source model
        """
        return [self.source_model.tokenizer.decode(token) for token in self.source_tokens]

    @property
    def target_words(self):
        """
        Return the input to the target model
        """
        return [self.target_model.tokenizer.decode(token) for token in self.target_tokens]

    def get_position_and_layer(self):
        if self.source.position is None:
            # If no position or layer is specified, take them all
            self.source.position = range(len(self.source_tokens))
            # self.source.layer = range(len(self.source_model.transformer.h))

        if self.target.position is None:
            self.target.position = range(len(self.target_tokens))
            # self.target.layer = range(self.target_model.config.n_layer)

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

    def full_output_words(self):
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

    def full_output(self):
        """
        Return the generated output from the target model
        This is a bit hacky. Its not super well supported. I have to concatenate all the inputs and add the input tokens to them.
        """
        return "".join(self.full_output_words())

    def find_in_source(self, substring):
        """
        Find the position of the target tokens in the source tokens
        """
        if substring not in self.source.prompt:
            raise ValueError(f"{substring} not in {self.source.prompt}")
        tokens = self.source_model.tokenizer.encode(substring)
        return self.source_tokens.index(tokens[0])

    def find_in_target(self, substring):
        """
        Find the position of the target tokens in the source tokens
        """
        if substring not in self.target.prompt:
            raise ValueError(f"{substring} not in {self.target.prompt}")
        tokens = self.target_model.tokenizer.encode(substring)
        return self.target_tokens.index(tokens[0])

    @property
    def n_layers(self):
        return len(self.target_model.transformer.h)

    def get_activation_pair(self, string_a: str, string_b: Optional[str] = None):
        """
        Get the activations for two strings for activation steering
        """
        tokens_a = self.target_model.tokenizer.encode(string_a)
        # If string_b is not provided, use spaces
        if string_b is None:
            string_b = " "
        tokens_b = self.target_model.tokenizer.encode(string_b)

        # Pad the shortest string with spaces
        if len(tokens_a) > len(tokens_b):
            tokens_b = tokens_b + self.target_model.tokenizer.encode(" ") * (len(tokens_a) - len(tokens_b))
        elif len(tokens_a) < len(tokens_b):
            tokens_a = tokens_a + self.target_model.tokenizer.encode(" ") * (len(tokens_b) - len(tokens_a))

        assert len(tokens_a) == len(tokens_b)
        position = range(len(tokens_a))

        source = deepcopy(self.source)
        source.prompt = self.target_model.tokenizer.decode(tokens_a)
        source.position = position
        activations_a = self.get_source_hidden_state(source)

        source.prompt = self.target_model.tokenizer.decode(tokens_b)
        activations_b = self.get_source_hidden_state(source)

        return activations_a, activations_b
