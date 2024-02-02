from patchscopes_nnsight.patchscopes import SourceContext, TargetContext, Patchscope

# Define the concept of 'love' and 'hate' representations,
# patch them into the target context.
# The modification to Patchscopes formalism is such that:
# For 'love' at position i:
# - The hidden representation from the source is combined with the 'love' vector.
# For 'hate' at position i (where i is aligned with the concept like in the token identity scenario):
# - The hidden representation from the source is subtracted by the 'hate' vector.
# The intervention happens at layer 6
# with the forward pass continuing post-intervention.
# We define 'love' and 'hate' vectors, and steering coefficients α and β
# to control the strength of the modifications.

remote = False

# The source context with the original prompt.
source_context = SourceContext(
    prompt="How do I feel about you? Let me make this crystal clear.",
    position=None,      # Position None will automatically be set to take all tokens.
    layer=6,            # Starting layer for the source representation.
    device="cuda:0" if remote else "cpu",
    model_name="gpt2-xl" if remote else "gpt2",
)

target_context = TargetContext.from_source(source_context)
target_context.max_new_tokens = 30


def activation_steering_mapping(h, added_vector, subtracted_vector, α: float = 5., β: float = 6.):
    """
    Applies activation steering, adding vectors to
    the corresponding 'y' positions within the larger tensor 'h'.

    Args:
        h (torch.Tensor): The hidden representations with shape [1, sequence_length, hidden_size].
        α (float): Coefficient to control the impact of the added activation.
        β (float): Coefficient to control the impact of the subtracted activation.
        added_vector (torch.Tensor): Padded added activation vector with shape [1, y_len, hidden_size].
        subtracted_vector (torch.Tensor): Padded subtracted activation vector with shape [1, y_len, hidden_size].

    Returns:
        torch.Tensor: The modified hidden representations with activations applied.
    """
    # Clone the tensor to prevent in-place modification of the computation graph
    updated_h = h.clone()

    # Check that love_vector and subtracted_vector have been appropriately padded
    if added_vector.shape != subtracted_vector.shape:
        raise ValueError("The shapes of love_vector and subtracted_vector must be the same.")

    # Apply the activation steering at each y position
    # If α and β are scalars, they apply uniformly across all y positions.
    for i in range(added_vector.size(1)):
        # Adds the 'love' and 'hate' vectors to the y positions in the hidden representation
        updated_h[:, i, :] += α * added_vector[:, i, :] - β * subtracted_vector[:, i, :]

    return updated_h


# Now we create the Patchscope with the source and target contexts.
patchscope = Patchscope(source=source_context, target=target_context)
patchscope.REMOTE = remote

add_vector, subtract_vector = patchscope.get_activation_pair("sadness", "happiness")

patchscope.target.mapping_function = lambda h: activation_steering_mapping(h, add_vector, subtract_vector)

patchscope.run()
print(patchscope.full_output())


# Switch them around
patchscope.target.mapping_function = lambda h: activation_steering_mapping(
    h, subtract_vector, add_vector
)

patchscope.run()
print(patchscope.full_output())
