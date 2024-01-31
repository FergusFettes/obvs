from patchscopes_nnsight.patchscopes import SourceContext, TargetContext, Patchscope

prompt = "John and Mary are walking together. John said to"

remote = True

# Setup source and target context with the simplest configuration
source_context = SourceContext(
    prompt=prompt,  # Example input text
    model_name="gpt2-xl" if remote else "gpt2",
    position=-1,  # Last token (assuming single input)
    layer=0,  # 10th layer (logit lense actually tests each layer, we'll start with one.)
    device="cpu"
)


target_context = TargetContext.from_source(source_context)
target_context.layer = -1  # Last layer (logit lens)

print(source_context)
print(target_context)

patchscope = Patchscope(source=source_context, target=target_context)
patchscope.REMOTE = remote

# Layers of model
layers = len(patchscope.target_model.transformer.h)
step_size = layers // 10


john_token = patchscope.target_model.tokenizer.encode(" John")
mary_token = patchscope.target_model.tokenizer.encode(" Mary")
john_probs = []
mary_probs = []
top_k = []
top_k_probs = []
for i in range(1, layers - 1, step_size):
    print(f"Layer {i}")

    patchscope.source.layer = i
    patchscope.run()

    probs = patchscope.probabilities()[-1]
    john_probs.append(probs[john_token].item())
    mary_probs.append(probs[mary_token].item())

    top_k.append(patchscope.top_k_tokens(5))
    top_k_probs.append(patchscope.top_k_probs(5))


print(top_k)
# print(top_k_probs)
print(john_probs)
print(mary_probs)

print(
    list(zip(
        patchscope.target_input(),
        patchscope.output()
    ))
)

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
    title="John and Mary Logits by Layer",
    xaxis_title="Layer",
    yaxis_title="Logit"
)

fig.show()
