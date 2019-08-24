import torch
from torch.nn import Module

# from pytorch pull request "Generic version of pixel_shuffle - 1d, 2d, ..., nd"
# this PR is outdated and has a bug, that has been fixed here
# https://github.com/pytorch/pytorch/pull/6340

def pixel_shuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
    tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Variable): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples::
        # 1D example
        >>> input = torch.Tensor(1, 4, 8)
        >>> output = F.pixel_shuffle(input, 2)
        >>> print(output.size())
        torch.Size([1, 2, 16])

        # 2D example
        >>> input = torch.Tensor(1, 9, 8, 8)
        >>> output = F.pixel_shuffle(input, 3)
        >>> print(output.size())
        torch.Size([1, 1, 24, 24])

        # 3D example
        >>> input = torch.Tensor(1, 8, 16, 16, 16)
        >>> output = F.pixel_shuffle(input, 2)
        >>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])
    """
    batch_size, channels, *input_sizes = list(input.size())
    dim = len(input_sizes)

    channels //= (upscale_factor ** dim)
    output_sizes = [input_size * upscale_factor for input_size in input_sizes]

    input_view = input.contiguous().view(batch_size, channels,
            *([upscale_factor] * dim), *input_sizes)

    axes = torch.arange(2, 2 + 2 * dim).reshape(2, dim)
    axes = axes[[1, 0]]
    axes = axes.transpose(0, 1).reshape(-1)
    shuffle_out = input_view.permute(0, 1, *axes).contiguous()

    return shuffle_out.view(batch_size, channels, *output_sizes)


class PixelShuffle(Module):
    r"""Rearranges elements in a Tensor of shape :math:`(N, C, d_{1}, d_{2}, ..., d_{n})` to a
    tensor of shape :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`.
    Where :math:`n` is the dimensionality of the data.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    Input Tensor must have at least 3 dimensions, e.g. :math:`(N, C, d_{1})` for 1D data,
    but Tensors with any number of dimensions after :math:`(N, C, ...)` (where N is mini-batch size,
    and C is channels) are supported.

    Look at the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(N, C, d_{1}, d_{2}, ..., d_{n})`
        - Output: :math:`(N, C/(r^n), d_{1}*r, d_{2}*r, ..., d_{n}*r)`
        Where :math:`n` is the dimensionality of the data, e.g. :math:`n-1` for 1D audio,
        :math:`n=2` for 2D images, etc.

    Examples::

        # 1D example
        >>> ps = nn.PixelShuffle(2)
        >>> input = torch.Tensor(1, 4, 8)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 2, 16])

        # 2D example
        >>> ps = nn.PixelShuffle(3)
        >>> input = torch.Tensor(1, 9, 8, 8)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 24, 24])

        # 3D example
        >>> ps = nn.PixelShuffle(2)
        >>> input = torch.Tensor(1, 8, 16, 16, 16)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 32, 32, 32])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_shuffle(input, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)
