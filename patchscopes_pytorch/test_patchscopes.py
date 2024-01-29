from patchscopes_pytorch.patchsopes import SourceContext, TargetContext, Patchscope


# Setup source and target context with the simplest configuration
source_context = SourceContext(
    input_sequence=["Hello, how are you?"],  # Example input text
    model_name="gpt2",
    position=-1,  # Last token (assuming single input)
    layer=10,  # 10th layer (logit lense actually tests each layer, we'll start with one.)
)

target_context = TargetContext(
    target_prompt=["Hello, how are you?"],  # Same as source
    model_name="gpt2",
    position=-1,  # Same position
    layer=-1  # Last layer (logit lens)
)

# Initialize Patchscope with the source and target contexts
patchscope = Patchscope(source=source_context, target=target_context)

# Perform the patch operation
patched_output = patchscope.patch()

# For demonstration, show the shape of the patched output
print(patched_output.shape)  # Should correspond to the shape of model's last layer output

