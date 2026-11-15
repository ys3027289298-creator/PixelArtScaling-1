"""
=======
ScaleNx
=======

------------------------------------------------------------------------------
Scale2x, Scale3x, Scale2xSFX and Scale3xSFX image rescaling for Python >=3.10.
------------------------------------------------------------------------------

:Abstract: Current module comprise **Scale2x**, **Scale3x**, **Scale2xSFX**
    and **Scale3xSFX** image rescaling functions, implemented in pure Python.

    Scale2x, Scale3x and Scale4x algorithms `[1]`_ were initially developed by
    `Andrea Mazzoleni`_ in 2001 AD for the sole purpose of upscaling
    low resolution screen output without introducing intermediate colors
    (*i.e.* blur) for old DOS games emulators `[2]`_ and
    similar narrow niche software.

    Later on *ca.* 2014 AD significantly improved Scale2xSFX and Scale3xSFX
    algorithms `[3]`_, providing better diagonals rendering, were introduced
    for the same very specific purpose.

    While screen scalers, based on algorithms above, are numerous,
    general purpose image rescaling wants still seem to be unsupplied.

    Due to severe demand for general purpose ScaleNx library,
    and apparent lack thereof, current general purpose pure Python
    `ScaleNx`_ implementation was developed.

Usage
-----

::

    from scalenx import scaleNx
    result_image = scaleNx(source_image, n, sfx)

where:

- **``source_image``**: source image 3D nested list;
        coordinate system match Photoshop, *i.e.* origin is top left corner,
        channels order is LA or RGBA from 0 to top;
- **``n``**: choice between Scale2* and Scale3* methods:
    - ``n=2``: Scale2x or Scale2xSFX;
    - ``n=3``: Scale3x or Scale3xSFX;

- **``sfx``**: choice between original ScaleNx and improved ScaleNxSFX methods:
    - ``sfx=False``: Scale2x or Scale3x;
    - ``sfx=True``: Scale2xSFX or Scale3xSFX.

.. note:: Function name **``scaleNx``** must contain capital **N**
    to avoid confusion with legacy file/module names.

Compatibility info
------------------

Current implementation is proven to work with CPython 3.10 and above,
and supposed to do so with any other Python version
understanding or ignoring type hints *etc.*

Copyright and redistribution
----------------------------

Current Python `ScaleNx`_ implementation is developed by
Ilya Razmanov (hereinafter referred to as "the Developer"), based on
algorithm descriptions `[1]`_, `[3]`_ by corresponding originators.

Changes introduced by the Developer for the purpose of
speed-up are entirely on his conscience.

Current implementation may be freely used, redistributed and
improved at will by anyone.
Sharing useful modifications with the Developer and lesser species
is your sacred duty to the Universe.

References
----------

`[1]`_. Original description of Scale2x and Scale3x algorithms by `Andrea Mazzoleni`_.

`[2]`_. DOSBox - DOS emulator, using Scale2x and Scale3x screen upscaling.

`[3]`_. Original Scale2xSFX and Scale3xSFX proposal, archived copy.

.. _[1]: https://www.scale2x.it/algorithm

.. _[2]: https://www.dosbox.com/

.. _[3]: https://web.archive.org/web/20160527015550/https://libretro.com/forums/archive/index.php?t-1655.html

.. _Andrea Mazzoleni: https://www.scale2x.it/authors

Resources
---------

The Developer's site: `The Toad's Slimy Mudhole`_;

.. _The Toad's Slimy Mudhole: https://dnyarri.github.io

`ScaleNx`_ explanations and illustrations page for current implementation;

.. _ScaleNx: https://dnyarri.github.io/scalenx.html

ScaleNx source repositories: `ScaleNx@Github`_, `ScaleNx@Gitflic`_;

.. _ScaleNx@Github: https://github.com/Dnyarri/PixelArtScaling

.. _ScaleNx@Gitflic: https://gitflic.ru/project/dnyarri/pixelartscaling

`Changelog`_ for current implementation.

.. _Changelog: https://github.com/Dnyarri/PixelArtScaling/blob/main/CHANGELOG.md

"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2024-2026 Ilya Razmanov'
__credits__ = 'Ilya Razmanov'
__license__ = 'unlicense'
__version__ = '2026.8.6.312'
__maintainer__ = 'Ilya Razmanov'
__email__ = 'ilyarazmanov@gmail.com'
__status__ = 'Production'
__all__ = [
    'scaleNx',
    'smooth_image',
    'smooth_then_scale',
    'gradient_mean',
    'jagged_pixels',
    'diff_pixels',
    'box_smooth',
    'gaussian_smooth',
]

import math
from typing import Literal

from .scalenx import scale2x, scale3x
from .scalenxsfx import scale2x as scale2xsfx
from .scalenxsfx import scale3x as scale3xsfx


def scaleNx(source_image: list[list[list[int]]], n: Literal[2, 3], sfx: bool) -> list[list[list[int]]]:
    """ScaleNx image rescaling, configurable via ``n`` and ``sfx`` options.
    ----

    :param source_image: source image 3D nested list;
        coordinate system match Photoshop, *i.e.* origin is top left corner,
        channels order is LA or RGBA from 0 to top;
    :type source_image: list[list[list[int]]]
    :param int n: ``2`` or ``3``, choice between Scale2* and Scale3* methods;
    :param bool sfx: choice between ScaleNx and ScaleNxSFX methods.
    :raises ValueError: Attempt to use nonexistent method ``n``.
    :return: rescaled image of the same type as ``source_image``.
    :rtype: list[list[list[int]]]

    """
    if sfx:
        if n == 2:
            return scale2xsfx(source_image)
        elif n == 3:
            return scale3xsfx(source_image)
        else:
            raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')
    else:
        if n == 2:
            return scale2x(source_image)
        elif n == 3:
            return scale3x(source_image)
        else:
            raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')


""" ╔════════════════════════════════════════════════════════╗
    ║ Light smoothing preprocessing and smooth-scale pipeline ║
    ╚════════════════════════════════════════════════════════╝ """

#: Supported smoothing kernel names.
KERNELS = ('box', 'gaussian')


def _check_radius(radius: int) -> int:
    """Validate smoothing radius, returning it as a positive int."""
    if type(radius) is not int:
        raise TypeError(f'Smoothing radius must be an int, got `{type(radius).__name__}`.')
    if radius <= 0:
        raise ValueError(f'Smoothing radius must be a positive integer, got `{radius}`.')
    return radius


def _check_kernel(kernel: str) -> str:
    """Validate smoothing kernel name."""
    if kernel not in KERNELS:
        raise ValueError(f'Smoothing kernel must be one of {KERNELS!r}, got `{kernel}`.')
    return kernel


def _kernel_weights(kernel: str, radius: int) -> list[float]:
    """Build 1D normalized kernel weights of length ``2 * radius + 1``."""
    if kernel == 'box':
        side = 2 * radius + 1
        return [1.0 / side] * side
    # Gaussian kernel, sigma = radius / 2, centered at 0
    sigma = radius / 2.0
    raw = [math.exp(-(offset ** 2) / (2.0 * sigma * sigma)) for offset in range(-radius, radius + 1)]
    total = sum(raw)
    return [weight / total for weight in raw]


def _convolve_1d(image: list[list[list[int]]], weights: list[float], radius: int, horizontal: bool) -> list[list[list[float]]]:
    """Separable convolution single pass, edge pixels use replicated border."""
    Y, X = (len(image), len(image[0]))
    channels = len(image[0][0])
    result: list[list[list[float]]] = [[[0.0] * channels for _ in range(X)] for _ in range(Y)]

    for y in range(Y):
        for x in range(X):
            acc = [0.0] * channels
            for step, weight in enumerate(weights):
                offset = step - radius
                if horizontal:
                    sample_y = y
                    sample_x = min(max(x + offset, 0), X - 1)
                else:
                    sample_y = min(max(y + offset, 0), Y - 1)
                    sample_x = x
                pixel = image[sample_y][sample_x]
                for channel in range(channels):
                    acc[channel] += weight * pixel[channel]
            result[y][x] = acc
    return result


def _round_pixel(pixel: list[float]) -> list[int]:
    """Round float channel values to the nearest int, no value clamping."""
    return [int(round(value)) for value in pixel]


def smooth_image(source_image: list[list[list[int]]], kernel: str = 'box', radius: int = 1) -> list[list[list[int]]]:
    """Light smoothing preprocessing using a box or Gaussian kernel.
    ----

    Applies a normalized separable convolution (horizontal then vertical
    pass) with replicated border handling. Output keeps the same dimensions,
    channel order and structure as ``source_image``.

    :param source_image: source image 3D nested list;
        coordinate system match Photoshop, *i.e.* origin is top left corner,
        channels order is LA or RGBA from 0 to top;
    :type source_image: list[list[list[int]]]
    :param str kernel: ``'box'`` (uniform box blur) or ``'gaussian'``
        (Gaussian blur with sigma = radius / 2);
    :param int radius: positive kernel radius; the kernel window size is
        ``2 * radius + 1`` pixels along each axis.
    :raises ValueError: Unknown ``kernel`` name or non-positive ``radius``.
    :raises TypeError: ``radius`` is not an int.
    :return: smoothed image of the same structure as ``source_image``.
    :rtype: list[list[list[int]]]

    """
    _check_kernel(kernel)
    _check_radius(radius)

    weights = _kernel_weights(kernel, radius)
    horizontal = _convolve_1d(source_image, weights, radius, horizontal=True)
    convolved = _convolve_1d(horizontal, weights, radius, horizontal=False)

    return [[_round_pixel(pixel) for pixel in row] for row in convolved]


def box_smooth(source_image: list[list[list[int]]], radius: int = 1) -> list[list[list[int]]]:
    """Box kernel smoothing, thin wrapper around :func:`smooth_image`."""
    return smooth_image(source_image, kernel='box', radius=radius)


def gaussian_smooth(source_image: list[list[list[int]]], radius: int = 1) -> list[list[list[int]]]:
    """Gaussian kernel smoothing, thin wrapper around :func:`smooth_image`."""
    return smooth_image(source_image, kernel='gaussian', radius=radius)


def smooth_then_scale(
    source_image: list[list[list[int]]],
    kernel: str = 'box',
    radius: int = 1,
    n: Literal[2, 3] = 2,
    sfx: bool = False,
    preview: bool = False,
) -> list[list[list[int]]]:
    """Smoothing preprocessing chained with the existing ScaleNx scaling.
    ----

    The source image is first smoothed via :func:`smooth_image` using
    ``kernel`` and ``radius``; unless ``preview`` is set, the smoothed image
    is then passed unchanged into :func:`scaleNx`. The scaling step reuses
    the original ScaleNx functions, so per-pixel scaling behavior is
    identical to calling :func:`scaleNx` directly.

    :param source_image: source image 3D nested list;
    :type source_image: list[list[list[int]]]
    :param str kernel: ``'box'`` or ``'gaussian'`` smoothing kernel;
    :param int radius: positive smoothing kernel radius;
    :param int n: ``2`` or ``3``, passed through to :func:`scaleNx`;
    :param bool sfx: SFX variant flag, passed through to :func:`scaleNx`;
    :param bool preview: when ``True``, return only the smoothed image
        at source resolution (smoothing-only preview, no scaling).
    :raises ValueError: Unknown ``kernel`` name or non-positive ``radius``.
    :raises TypeError: ``radius`` is not an int.
    :return: smoothed and (unless ``preview``) rescaled image.
    :rtype: list[list[list[int]]]

    """
    _check_kernel(kernel)
    _check_radius(radius)

    smoothed = smooth_image(source_image, kernel=kernel, radius=radius)
    if preview:
        return smoothed
    return scaleNx(smoothed, n, sfx)


def gradient_mean(image: list[list[list[int]]]) -> float:
    """Mean absolute gradient magnitude over all image pixels.
    ----

    The gradient at each pixel is the Euclidean magnitude of the mean
    absolute right and down neighbor differences across all channels,
    using replicated borders at the edges. Lower values indicate softer
    transitions; smoothing is expected to reduce this metric.

    :param image: image 3D nested list of int channel values.
    :return: mean gradient magnitude as a float.
    :rtype: float

    """
    Y, X = (len(image), len(image[0]))
    channels = len(image[0][0])
    total = 0.0

    for y in range(Y):
        for x in range(X):
            pixel = image[y][x]
            right = image[y][min(x + 1, X - 1)]
            down = image[min(y + 1, Y - 1)][x]
            dx = sum(abs(right[channel] - pixel[channel]) for channel in range(channels)) / channels
            dy = sum(abs(down[channel] - pixel[channel]) for channel in range(channels)) / channels
            total += math.hypot(dx, dy)

    return total / (Y * X)


def jagged_pixels(image: list[list[list[int]]], threshold: int = 32) -> int:
    """Count pixels exposed to a hard neighbor transition ("jaggies").
    ----
    A pixel is counted when its mean absolute difference against any of its
    4-connected neighbors (replicated border) exceeds ``threshold`` on at
    least one neighbor. The metric is resolution-dependent, so comparisons
    must be made between images of equal size.

    :param image: image 3D nested list of int channel values.
    :param int threshold: positive mean per-channel difference threshold.
    :raises ValueError: non-positive ``threshold``.
    :return: number of jagged pixels in the image.
    :rtype: int

    """
    if threshold <= 0:
        raise ValueError(f'Jagged threshold must be positive, got `{threshold}`.')

    Y, X = (len(image), len(image[0]))
    channels = len(image[0][0])
    count = 0

    for y in range(Y):
        for x in range(X):
            pixel = image[y][x]
            jagged = False
            for neighbor_y, neighbor_x in (
                (max(y - 1, 0), x),
                (min(y + 1, Y - 1), x),
                (y, max(x - 1, 0)),
                (y, min(x + 1, X - 1)),
            ):
                neighbor = image[neighbor_y][neighbor_x]
                difference = sum(abs(neighbor[channel] - pixel[channel]) for channel in range(channels)) / channels
                if difference > threshold:
                    jagged = True
                    break
            if jagged:
                count += 1

    return count


def diff_pixels(image_a: list[list[list[int]]], image_b: list[list[list[int]]]) -> int:
    """Count pixels differing between two equally sized images."""
    if len(image_a) != len(image_b) or len(image_a[0]) != len(image_b[0]):
        raise ValueError('Images must have equal dimensions to compare pixel by pixel.')

    return sum(
        1
        for row_a, row_b in zip(image_a, image_b)
        for pixel_a, pixel_b in zip(row_a, row_b)
        if pixel_a != pixel_b
    )
