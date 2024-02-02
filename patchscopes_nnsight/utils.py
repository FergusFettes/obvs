def activation_steering_mapping(h, added_vector, subtracted_vector, coef: float = 5.):
    """
    Applies activation steering, adding vectors to
    the corresponding 'y' positions within the larger tensor 'h'.

    Args:
        h (torch.Tensor): The hidden representations with shape [1, sequence_length, hidden_size].
        added_vector (torch.Tensor): Padded added activation vector with shape [1, y_len, hidden_size].
        subtracted_vector (torch.Tensor): Padded subtracted activation vector with shape [1, y_len, hidden_size].
        coef (float): Coefficient to control the impact of the added activation.

    Returns:
        torch.Tensor: The modified hidden representations with activations applied.
    """
    # Clone the tensor to prevent in-place modification of the computation graph
    updated_h = h.clone()

    # Check that love_vector and subtracted_vector have been appropriately padded
    if added_vector.shape != subtracted_vector.shape:
        raise ValueError("The shapes of added_vector and subtracted_vector must be the same.")

    # Check that the length of updated_h can fit the added_vector and subtracted_vector
    if added_vector.shape[1] > updated_h.shape[1]:
        raise ValueError("The length of added_vector is greater than the length of updated_h.")

    # Apply the activation steering at each y position
    # If α and β are scalars, they apply uniformly across all y positions.
    for i in range(added_vector.size(1)):
        # Adds the 'love' and 'hate' vectors to the y positions in the hidden representation
        updated_h[:, i, :] += coef * added_vector[:, i, :] - coef * subtracted_vector[:, i, :]

    return updated_h
