from patchscopes_pytorch.patchsopes import SourceContext, TargetContext, Patchscope

prompt = "Today is opposites day. The grass is blue. The sea is green. Our protagonist goes outside and looks around at the birds and the beas. She looks up at the sky, and marvels at its hue, a beautiful bright green. The suns color is"

# Setup source and target context with the simplest configuration
source_context = SourceContext(
    input_sequence=[prompt],  # Example input text
    model_name="gpt2",
    position=-1,  # Last token (assuming single input)
    layer=0,  # 10th layer (logit lense actually tests each layer, we'll start with one.)
    device="cpu"
)

target_context = TargetContext(
    target_prompt=source_context.input_sequence,
    model_name=source_context.model_name,
    position=source_context.position,
    device=source_context.device,

    layer=-1,  # Last layer (logit lens)
)

patchscope = Patchscope(source=source_context, target=target_context)
green_token = patchscope.target_model.tokenizer.encode(" green")
blue_token = patchscope.target_model.tokenizer.encode(" blue")
green_probs = []
blue_probs = []
top_k = []
for i in range(1, 11):
    print(f"Layer {i}")

    source_context.layer = i
    patchscope = Patchscope(source=source_context, target=target_context)
    patchscope.run()

    logits = patchscope.logits()
    green_probs.append(logits[green_token].item())
    blue_probs.append(logits[blue_token].item())

    top_k.append(patchscope.top_k_tokens(5))


print(green_probs)
print(blue_probs)
print(top_k)
