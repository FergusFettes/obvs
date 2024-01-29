from patchscopes_pytorch.patchsopes import SourceContext, TargetContext, Patchscope


# Setup source and target context with the simplest configuration
source_context = SourceContext(
    input_sequence=["On opposites day, the grass is blue and the sky is"],  # Example input text
    model_name="gpt2",
    position=-1,  # Last token (assuming single input)
    layer=10,  # 10th layer (logit lense actually tests each layer, we'll start with one.)
    device="cpu"
)

target_context = TargetContext(
    target_prompt=["On opposites day, the grass is blue and the sky is"],  # Same as source
    model_name="gpt2",
    position=-1,  # Same position
    layer=-1,  # Last layer (logit lens)
    device="cpu"
)

# # Run it over all the layers
# for layer in range(12):
#     source_context.layer = layer
#     patchscope = Patchscope(source=source_context, target=target_context)
#     patched_output = patchscope.run()

patchscope = Patchscope(source=source_context, target=target_context)
patched_output = patchscope.run()
