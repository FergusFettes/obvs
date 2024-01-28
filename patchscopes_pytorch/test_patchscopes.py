from patchscopes_pytorch import SourceContext, TargetContext, Patchscope


# Define your simple identity mapping function
def identity(x):
    return x


# Setup source and target context with the simplest configuration
source_context = SourceContext(
    input_sequence=["Hello, how are you?"],  # Example input text
    position=-1,  # Last token (assuming single input)
    layer=-2  # Second-to-last layer
)

target_context = TargetContext(
    target_prompt=["Hello, how are you?"],  # Same as source
    position=-1,  # Same position
    mapping_function=identity,  # Identity function, no change
    layer=-1  # Last layer (logit lens)
)

# Initialize Patchscope with the source and target contexts
patchscope = Patchscope(source=source_context, target=target_context)

# Perform the patch operation
patched_output = patchscope.patch()

# For demonstration, show the shape of the patched output
print(patched_output.shape)  # Should correspond to the shape of model's last layer output

