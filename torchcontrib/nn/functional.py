def film(input, gamma, beta):
    r"""Applies Feature-wise Linear Modulation to the incoming data.
     See :class:`~torchcontrib.nn.FiLM` for details.
    """
    if input.dim() < 2:
        raise ValueError("film expects input to be at least 2-dimensional, but "
                         "got input of size {}".format(tuple(input.size())))
    if gamma.dim() != 2 and gamma.size(0) == input.size(0) and gamma.size(1) == input.size(1):
        raise ValueError("film expects gamma to be a 2-dimensional tensor of "
                         "the same shape as the first two dimensions of input"
                         "gamma of size {} and input of size {}"
                         .format(tuple(gamma.size()), tuple(input.size())))
    if beta.dim() != 2 and beta.size(0) == input.size(0) and beta.size(1) == input.size(1):
        raise ValueError("film expects beta to be a 2-dimensional tensor of "
                         "the same shape as the first two dimensions of input"
                         "beta of size {} and input of size {}"
                         .format(tuple(beta.size()), tuple(input.size())))
    view_shape = list(input.size())
    for i in range(2, len(view_shape)):
        view_shape[i] = 1
    return gamma.view(view_shape) * input + beta.view(view_shape)
