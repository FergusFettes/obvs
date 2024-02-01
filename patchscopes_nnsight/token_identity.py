from patchscopes_nnsight.patchscopes import SourceContext, TargetContext, Patchscope

remote = True

# We are testing when the concept of 'Mary' is loaded into the final token.
prompt = "John was walking with Mary. John said to"
source_context = SourceContext(
    prompt=prompt,
    position=-1,
    layer=0,
    device="cpu",
    model_name="gpt2-xl" if remote else "gpt2",
)

target_context = TargetContext.from_source(source_context)

# To do this, we patch the final token into the target prompt at the 'x' position
target_context.prompt = "cat is cat; 135 is 135; hello is hello; black is black; shoe is shoe; x is"

# Create the patchscope
patchscope = Patchscope(source=source_context, target=target_context)
patchscope.REMOTE = remote

# We need to find the position of x and set the target position
start_position = patchscope.find_in_target(" x")
# Patch the item before the target so it thinks that x is the next token
target_context.position = start_position


# Prepare to gather data
john_token = patchscope.target_model.tokenizer.encode(" John")
mary_token = patchscope.target_model.tokenizer.encode(" Mary")

# Run the patchscope for each target layer
john_probs = []
mary_probs = []
top_k = []
top_k_probs = []

# Run the parchscope for each target layer
patchscope.target.layer = 20
step_size = patchscope.n_layers // 5
for j in range(patchscope.n_layers - 5, patchscope.n_layers):
    print(f"Source Layer {j}")
    patchscope.source.layer = j
    patchscope.run()
    probs = patchscope.probabilities()[start_position]
    john_probs.append(probs[john_token].item())
    mary_probs.append(probs[mary_token].item())
    top_k.append(patchscope.top_k_tokens(5))
    top_k_probs.append(patchscope.top_k_probs(5))
    print("".join(patchscope.full_output()))

output = patchscope._target_outputs[0].value.argmax(dim=-1).tolist()
decoded = [patchscope.target_model.tokenizer.decode(token) for token in output]

print(top_k)
print(john_probs)
print(mary_probs)

for target, output in zip(patchscope.target_words, patchscope.output()):
    print(f"({target} -> {output})")

print(patchscope.full_output())

# Plot the john and mary probs by layer as lines with plotly
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(1, 12)),
    y=john_probs,
    name="John"
))

fig.add_trace(go.Scatter(
    x=list(range(1, 12)),
    y=mary_probs,
    name="Mary"
))

fig.update_layout(
    title="John and Mary Probs by Layer",
    xaxis_title="Layer",
    yaxis_title="Logit"
)

fig.show()
