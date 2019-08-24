from torch.nn import Module

# from pytorch pull request "Pixelshuffle for volumetric convolutions"
# https://github.com/pytorch/pytorch/pull/5051

def pixel_shuffle_3d(input, upscale_factor):
    r"""Rearranges elements in a tensor of shape ``[*, C*r^3, H, W, D]`` to a
    tensor of shape ``[C, H*r, W*r, D*r]``.
    See :class:`~torch.nn.PixelShuffle3D` for details.
    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by
    Examples::
        >>> ps = nn.PixelShuffle3D(2)
        >>> input = autograd.Variable(torch.Tensor(1, 8, 4, 4, 4))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 8, 8, 8])
    """
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, upscale_factor, upscale_factor, upscale_factor,
        in_height, in_width, in_depth)

    shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width, out_depth)


class PixelShuffle3D(Module):
    r"""Rearranges elements in a Tensor of shape :math:`(*, C * r^3, H, W, D]` to a
    tensor of shape :math:`(C, H * r, W * r, D * r)`.
    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.
    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details
    Args:
        upscale_factor (int): factor to increase spatial resolution by
    Shape:
        - Input: :math:`(N, C * {upscale\_factor}^3, H, W, D)`
        - Output: :math:`(N, C, H * {upscale\_factor}, W * {upscale\_factor}, D * * {upscale\_factor})`
    Examples::
        >>> ps = nn.PixelShuffle(2)
        >>> input = autograd.Variable(torch.Tensor(1, 8, 4, 4, 4))
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 8, 8, 8])
    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_shuffle_3d(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
