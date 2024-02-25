import time
from scripts.token_identity import setup, run_over_all_layers

from obvspython.vis import create_heatmap
from obvspython.logging import logger

from nnsight import LanguageModel
import numpy as np


word = " boat"
prompt = "if its on the road, its a car. if its in the air, its a plane. if its on the sea, its a"
device = "cpu"
model_name = "gpt2"


target_tokens, patchscope = setup(prompt, model_name, device, word)

assert patchscope.source_model == patchscope.target_model


values = np.zeros((4, 4))
start = time.time()
source_layers, target_layers, values, outputs = run_over_all_layers(patchscope, target_tokens, values)
batched_time = time.time() - start

logger.info(f"Elapsed time for batch: {batched_time:.2f}s. Layers: {source_layers}, {target_layers}")

fig = create_heatmap(source_layers, target_layers, values, title=f"Token Identity: Surprisal by Layer {model_name} {prompt}")
fig.show()


model = LanguageModel(model_name, device_map="cpu")

patchscope.target_model = model

assert patchscope.source_model != patchscope.target_model

values = np.zeros((4, 4))
start = time.time()
source_layers, target_layers, values, outputs = run_over_all_layers(patchscope, target_tokens, values)
unbatched_time = time.time() - start

logger.info(f"Elapsed time for unbatched: {unbatched_time:.2f}s. Layers: {source_layers}, {target_layers}")

fig = create_heatmap(source_layers, target_layers, values, title=f"Token Identity: Surprisal by Layer {model_name} {prompt}")
fig.show()

diff = batched_time - unbatched_time
logger.info(f"Batched time: {batched_time:.2f}s, Unbatched time: {unbatched_time:.2f}s, Difference: {diff:.2f}s")
