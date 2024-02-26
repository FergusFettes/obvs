"""
Nucleus expansion module.

For expanding all the paths of a completion out to some cumulative probability.

The basic usage is to expand a prompt out to some cumulative probability.

You can also expand the prompt at multiple positions.

You can also expand to a fixed depth.

You can provide a validation function to validate the expansion, if eg. you only want
whole words.

You can also provide a list of words to include in the expansion.
"""

import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import re
from typing import Union, Sequence, Dict
from tqdm import tqdm

from obvs.patchscope import Patchscope
from obvs.logging import logger
from obvs.utils import get_model_specifics

from nnsight import LanguageModel

import torch


def validate_word(word):
    word = word.strip()
    if not word:
        return False
    if not re.match(r"^[a-zA-Z']+$", word):
        return False
    return True


@dataclass_json
@dataclass
class Node:
    """
    Node class.

    A node in the expansion graph.
    """

    id: int
    text: str
    probability: float
    parent: int
    depth: int

    def get_children(self, nodes: Dict[int, "Node"]):
        return [node for id, node in nodes.items() if node.parent == self.id]


class NucleusExpansion:
    """
    Nucleus expansion class.

    Expand a prompt.
    """
    DEFAULT_PROMPT = "A typical definition of X would be '"

    def __init__(
        self,
        prompt: Union[str, list[int], torch.Tensor, Patchscope],
        model_name: str = "gpt2",
        position: Sequence[int] = [-1],
        cutoff_prob=1e-2,
        cutoff_depth=1e6,
        cutoff_breadth=10,                  # This is the topk
        validation_fn=lambda x: True,
        includes=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the NucleusExpansion object.

        Args:
            prompt (str): The prompt to expand.
            cutoff_prob (float): The cumulative probability to expand to.
            cutoff_depth (int): The maximum depth to expand to.
            validation_fn (function): A function to validate the expansion.
            includes (list): A list of words to include in the expansion.
        """
        self.use_embedding = False
        self.model_name = model_name
        self._prompt, self.model = self._parse_prompt(prompt, device)
        if self.use_embedding:
            self.get_target_position()
        self.cutoff_prob = cutoff_prob
        self.cutoff_depth = cutoff_depth
        self.cutoff_breadth = cutoff_breadth
        self.validation_fn = validation_fn
        self.includes = includes or []
        self.nodes = {0: Node(0, self.prompt, 1.0, None, 0)}
        self.model_specifics = get_model_specifics(self.model_name)

        self.progress_bar = tqdm(desc="Processing", unit="iter")

    @property
    def prompt(self):
        return self._prompt

    @prompt.setter
    def prompt(self, value):
        """
        We will use this to tokenize prompt if the user sets them later.
        """
        self._prompt = value
        self.reset()

    def _parse_prompt(self, prompt, device):
        # We should be able to expand a list of tokens directly.
        # Fow now, lets just do the word expansion.
        if isinstance(prompt, list):
            logger.info("Tokens not yet supported!")
            raise NotImplementedError
            # return [prompt]
        # If its a patchscope, we need to expand the patched representation.
        if isinstance(prompt, Patchscope):
            logger.info("Pathchscopes not yet supported!")
            raise NotImplementedError
            # return prompt.target.prompt, prompt.target_model
        if isinstance(prompt, torch.Tensor):
            self.use_embedding = True
            self.embedding = prompt
            return self.DEFAULT_PROMPT, LanguageModel(self.model_name, device_map=device)

        # We will actually pretty quickly want to conver this to tokens. But first lets get it working with text.
        return prompt, LanguageModel(self.model_name, device_map=device)

    def expand(self):
        """
        Expand the prompt.

        Returns:
            list: The expanded prompt as nodes.
        """
        tokens = self.model.tokenizer.encode(self.prompt)
        self.loop(tokens, 0, 0)
        return self.nodes

    def loop(self, prompt_tokens, node_id, depth):
        if depth >= self.cutoff_depth:
            return
        self.progress_bar.update(1)
        output, prompt = self.forward_pass(prompt_tokens)

        cumulative_prob = self.nodes[node_id].probability
        tokens = self.get_next_tokens(output, cumulative_prob)

        for prob, token in tokens:
            word = self.model.tokenizer.decode(token)
            if not self.validation_fn(word):
                logger.info(f"Skipping invalid word: {word}")
                continue

            logger.info(f"prompt: {prompt} -> {word}:\t{prob:.4f}\t({cumulative_prob * prob:.2e}/{self.cutoff_prob})")

            id = len(self.nodes) + 1
            self.nodes[id] = Node(id, word.strip(), prob * cumulative_prob, node_id, depth + 1)
            self.loop(prompt_tokens + [token], id, depth + 1)

    def forward_pass(self, prompt_tokens):
        prompt = self.model.tokenizer.decode(prompt_tokens)
        with self.model.trace(prompt) as _:
            if self.use_embedding:
                output = getattr(getattr(self.model, self.model_specifics[0]), self.model_specifics[2]).output
                output.t[self.token_position] = self.embedding
            return self.model.lm_head.output.t[-1].save(), prompt

    def get_next_tokens(self, output, cumulative_prob):
        # Apply softmax, filter out low probability tokens, then get the top k
        probs = torch.softmax(output.value, dim=-1)
        topk = probs.topk(self.cutoff_breadth)
        return [
            (prob.item(), token.item())
            for prob, token in zip(topk.values[0], topk.indices[0])
            if (cumulative_prob * prob) > self.cutoff_prob
        ]

    def get_target_position(self):
        """
        Get the position of the target token in the prompt.

        Args:
            prompt_tokens (list): The prompt tokens.

        Returns:
            int: The position of the target token.
        """
        tokens = self.model.tokenizer.encode(self.prompt)
        try:
            x = self.model.tokenizer.encode(" X")
            self.token_position = tokens.index(x[0])
        except ValueError:
            x = self.model.tokenizer.encode("X")
            self.token_position = tokens.index(x[0])

    def includes(self, output):
        """
        Only include the included words list if their probability is greater than some small value.
        """
        min_prob = 1e-3
        return [word for word in output if word in self.includes and output[word] > min_prob]

    def reset(self):
        self.nodes = {0: Node(0, self.prompt, 1.0, None, 0)}
        self.progress_bar.reset()


def export_json(nodes):
    # Make the nodes into a dict
    nodes = {node.id: node.to_dict() for node in nodes.values()}
    data = json.dumps(nodes, ensure_ascii=False)

    with open("graph.json", "w") as file:
        file.write(data)
    return data


def export_html(nodes):
    """
    Embed the data into the HTML template and save the result.

    :param nodes:
    """
    # Serialize the dictionary to a JSON-formatted string, ensuring that it does not escape non-ASCII characters
    data = export_json(nodes)

    # Escape sequences for JSON embedded in HTML/JavaScript
    data = (
        data
        .replace('\\', '\\\\')  # Escape backslashes
        .replace('`', '\\`')    # Escape backticks to allow use in JavaScript template literals
        .replace('\n', '\\n')   # Escape newlines
        .replace('\r', '\\r')   # Escape carriage returns
        .replace('\b', '\\b')   # Escape backspaces
        .replace('\f', '\\f')   # Escape formfeeds
        .replace('\t', '\\t')   # Escape tabs
    )

    # Convert the json to javascript
    data = f"JSON.parse(`{data}`)"

    # Read the HTML template
    with open("./obvs/expansion.html", 'r') as file:
        html_content = file.read()

    # Embed the tree data into the HTML
    html_content = html_content.replace('var rawJsonData = {};', f'var rawJsonData = {data};')

    # Save the modified HTML to the output path
    with open("./graph.html", 'w') as file:
        file.write(html_content)

    print("HTML file saved to ./graph.html")

    # Use subprocess to open the HTML file in the default web browser
    import subprocess
    subprocess.run(["open", "graph.html"])
