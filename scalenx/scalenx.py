#!/usr/bin/env python3

"""
=======
ScaleNx
=======

-----------------------------------
Scale2x and Scale3x image rescaling
-----------------------------------

:Abstract: Current module comprise **Scale2x** and **Scale3x** `[1]`_
    image rescaling functions, implemented in pure Python.

Usage
-----

Scale2x::

    scaled_image = scalenx.scale2x(source_image)

Scale3x::

    scaled_image = scalenx.scale3x(source_image)

where:

- ``source_image``: input image as list of lists (rows) of lists (pixels)
        of int (channel values);
- ``scaled_image``: output image as list of lists (rows) of lists (pixels)
        of int (channel values).

References
----------

`[1]`_. Original description of Scale2x and Scale3x algorithms by `Andrea Mazzoleni`_.

.. _[1]: https://www.scale2x.it/algorithm

.. _Andrea Mazzoleni: https://www.scale2x.it/authors

----
The Developer site: `The Toad's Slimy Mudhole`_

.. _The Toad's Slimy Mudhole: https://dnyarri.github.io

`ScaleNx`_ explanations and illustrations page for current
ScaleNx Python implementation.

.. _ScaleNx: https://dnyarri.github.io/scalenx.html

ScaleNx source repositories: `ScaleNx@Github`_, `ScaleNx@Gitflic`_.

.. _ScaleNx@Github: https://github.com/Dnyarri/PixelArtScaling

.. _ScaleNx@Gitflic: https://gitflic.ru/project/dnyarri/pixelartscaling

`Changelog`_ for current implementation:

.. _Changelog: https://github.com/Dnyarri/PixelArtScaling/blob/main/CHANGELOG.md

"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2024-2026 Ilya Razmanov'
__credits__ = ['Andrea Mazzoleni', 'Ilya Razmanov']
__license__ = 'unlicense'
__version__ = '2026.8.6.312'
__maintainer__ = 'Ilya Razmanov'
__email__ = 'ilyarazmanov@gmail.com'
__status__ = 'Production'

from functools import lru_cache
from typing import Literal

""" ╔════════════════════════════════════════════╗
    ║ Scaling image nested list to 2x image list ║
    ╚════════════════════════════════════════════╝ """


def scale2x(image3d: list[list[list[int]]]) -> list[list[list[int]]]:
    """Scale2x image rescale.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values); coordinate system match Photoshop, *i.e.*
        origin is top left corner, channels order is LA or RGBA from 0 to top;
    :type image3d: list[list[list[int]]]
    :return: 3D nested list of the same structure as input,
        rescaled in X and Y directions twice using Scale2x.
    :rtype: list[list[list[int]]]

    """

    # ↓ determining source image size from list
    Y, X = (len(image3d), len(image3d[0]))

    # ↓ starting new image list
    scaled_image: list[list[list[int]]] = []

    def _dva(A: list[int], B: list[int], C: list[int], D: list[int], E: list[int]):
        """Scale2x conditional tree function."""

        r1 = r2 = r3 = r4 = E

        if A != D and C != B:
            if A == C:
                r1 = C
            if A == B:
                r2 = B
            if D == C:
                r3 = C
            if D == B:
                r4 = B
        return r1, r2, r3, r4

    """ Source around default pixel E
        ┌───┬───┬───┐
        │   │ A │   │
        ├───┼───┼───┤
        │ C │ E │ B │
        ├───┼───┼───┤
        │   │ D │   │
        └───┴───┴───┘

        Result
        ┌────┬────┐
        │ r1 │ r2 │
        ├────┼────┤
        │ r3 │ r4 │
        └────┴────┘
    """
    for y in range(Y):
        """ ┌───────────────────────┐
            │ First pixel in a row. │
            │ "Repeat edge" mode.   │
            └───────────────────────┘ """
        A = image3d[max(y - 1, 0)][0]
        B = image3d[y][min(1, X - 1)]
        C = E = image3d[y][0]
        D = image3d[min(y + 1, Y - 1)][0]

        r1, r2, r3, r4 = _dva(A, B, C, D, E)

        row_rez = [r1, r2]
        row_dvo = [r3, r4]

        """ ┌───────────────────────────────────────────┐
            │ Next pixels in a row (below).             │
            │ Reusing pixels from previous kernel.      │
            │ Only rightmost pixels are read from list. │
            └───────────────────────────────────────────┘ """
        for x in range(1, X):
            C = E
            E = B
            A = image3d[max(y - 1, 0)][x]
            B = image3d[y][min(x + 1, X - 1)]
            D = image3d[min(y + 1, Y - 1)][x]

            r1, r2, r3, r4 = _dva(A, B, C, D, E)

            row_rez.extend((r1, r2))
            row_dvo.extend((r3, r4))

        scaled_image.append(row_rez)
        scaled_image.append(row_dvo)

    return scaled_image


# ↑ rescaling two times finished


""" ╔════════════════════════════════════════════╗
    ║ Scaling image nested list to 3x image list ║
    ╚════════════════════════════════════════════╝ """


def scale3x(image3d: list[list[list[int]]]) -> list[list[list[int]]]:
    """Scale3x image rescale.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values); coordinate system match Photoshop, *i.e.*
        origin is top left corner, channels order is LA or RGBA from 0 to top;
    :type image3d: list[list[list[int]]]
    :return: 3D nested list of the same structure as input,
        rescaled in X and Y directions thrice using Scale3x.
    :rtype: list[list[list[int]]]

    """

    # ↓ determining source image size from list
    Y, X = (len(image3d), len(image3d[0]))

    # ↓ starting new image list
    scaled_image: list[list[list[int]]] = []

    def _tri(A: list[int], B: list[int], C: list[int], D: list[int], E: list[int], F: list[int], G: list[int], H: list[int], I: list[int]):
        """Scale3x conditional tree function."""

        r1 = r2 = r3 = r4 = r5 = r6 = r7 = r8 = r9 = E

        if B != H and D != F:
            if D == B:
                r1 = D
            if (D == B and E != C) or (B == F and E != A):
                r2 = B
            if B == F:
                r3 = F
            if (D == B and E != G) or (D == H and E != A):
                r4 = D
            # central pixel r5 = E set already
            if (B == F and E != I) or (H == F and E != C):
                r6 = F
            if D == H:
                r7 = D
            if (D == H and E != I) or (H == F and E != G):
                r8 = H
            if H == F:
                r9 = F
        return r1, r2, r3, r4, r5, r6, r7, r8, r9

    if type(image3d[0][0]) is tuple:
        _tri = lru_cache(maxsize=512)(_tri)

    """ Source around default pixel E
        ┌───┬───┬───┐
        │ A │ B │ C │
        ├───┼───┼───┤
        │ D │ E │ F │
        ├───┼───┼───┤
        │ G │ H │ I │
        └───┴───┴───┘

        Result
        ┌────┬────┬────┐
        │ r1 │ r2 │ r3 │
        ├────┼────┼────┤
        │ r4 │ r5 │ r6 │
        ├────┼────┼────┤
        │ r7 │ r8 │ r9 │
        └────┴────┴────┘
    """
    for y in range(Y):
        """ ┌───────────────────────┐
            │ First pixel in a row. │
            │ "Repeat edge" mode.   │
            └───────────────────────┘ """
        A = B = image3d[max(y - 1, 0)][0]
        C = image3d[max(y - 1, 0)][min(1, X - 1)]
        D = E = image3d[y][0]
        F = image3d[y][min(1, X - 1)]
        G = H = image3d[min(y + 1, Y - 1)][0]
        I = image3d[min(y + 1, Y - 1)][min(1, X - 1)]

        r1, r2, r3, r4, r5, r6, r7, r8, r9 = _tri(A, B, C, D, E, F, G, H, I)

        row_rez = [r1, r2, r3]
        row_dvo = [r4, r5, r6]
        row_tre = [r7, r8, r9]

        """ ┌───────────────────────────────────────────┐
            │ Next pixels in a row (below).             │
            │ Reusing pixels from previous kernel.      │
            │ Only rightmost pixels are read from list. │
            └───────────────────────────────────────────┘ """
        for x in range(1, X):
            A = B
            B = C
            C = image3d[max(y - 1, 0)][min(x + 1, X - 1)]

            D = E
            E = F
            F = image3d[y][min(x + 1, X - 1)]

            G = H
            H = I
            I = image3d[min(y + 1, Y - 1)][min(x + 1, X - 1)]

            r1, r2, r3, r4, r5, r6, r7, r8, r9 = _tri(A, B, C, D, E, F, G, H, I)

            row_rez.extend((r1, r2, r3))
            row_dvo.extend((r4, r5, r6))
            row_tre.extend((r7, r8, r9))

        scaled_image.append(row_rez)
        scaled_image.append(row_dvo)
        scaled_image.append(row_tre)

    """if type(image3d[0][0]) is tuple:
        print(_tri.cache_info())"""

    return scaled_image


# ↑ rescaling three times finished


""" ╔══════════════════════════════════════════════════════════════╗
    ║ Three-stage enhancement pipeline for low-resolution material ║
    ║ Stage order is fixed: (1) smoothing/denoising,               ║
    ║ (2) integer ScaleNx upscale, (3) edge sharpening;            ║
    ║ each outer stage can be switched off independently.          ║
    ╚══════════════════════════════════════════════════════════════╝ """


def _maxval_of(image3d: list[list[list[int]]]) -> int:
    """Detect per-channel maximum from actual pixel values,
    matching 1/8/16 bpc ranges (``1``, ``255``, ``65535``)
    used by PyPNG/PyPNM representations."""

    peak = 0
    for row in image3d:
        for pixel in row:
            pixel_max = max(pixel)
            if pixel_max > peak:
                peak = pixel_max
    if peak <= 1:
        return 1
    if peak <= 255:
        return 255
    if peak <= 65535:
        return 65535
    return peak


def _round_half_up(delta: float) -> int:
    """Round float ``delta`` half away from zero, keeping sign."""

    if delta >= 0.0:
        return int(delta + 0.5)
    return -int(-delta + 0.5)


def _box_blur(image3d: list[list[list[int]]], radius: int) -> list[list[list[int]]]:
    """Separable integer box blur with "repeat edge" boundary mode:
    horizontal pass first, vertical pass second."""

    Y, X = len(image3d), len(image3d[0])
    Z = len(image3d[0][0])
    window = 2 * radius + 1

    # ↓ Horizontal pass, running sum per channel
    horizontal: list[list[list[int]]] = [[[0] * Z for _ in range(X)] for _ in range(Y)]
    for y in range(Y):
        source_row = image3d[y]
        target_row = horizontal[y]
        for z in range(Z):
            # ↓ First window: indices [0..radius], edge pixel 0 repeated "radius" times
            running = source_row[0][z] * radius
            running += sum(source_row[min(x + 1, X - 1)][z] for x in range(radius))
            for x in range(X):
                target_row[x][z] = running // window
                running += source_row[min(x + radius + 1, X - 1)][z]
                running -= source_row[max(x - radius, 0)][z]

    # ↓ Vertical pass, same running-sum approach
    blurred: list[list[list[int]]] = [[[0] * Z for _ in range(X)] for _ in range(Y)]
    for x in range(X):
        for z in range(Z):
            running = horizontal[0][x][z] * radius
            running += sum(horizontal[min(y + 1, Y - 1)][x][z] for y in range(radius))
            for y in range(Y):
                blurred[y][x][z] = running // window
                running += horizontal[min(y + radius + 1, Y - 1)][x][z]
                running -= horizontal[max(y - radius, 0)][x][z]

    return blurred


def _check_radius_strength(radius: int, strength: float, stage: str) -> None:
    """Shared validation for smoothing/sharpening stage parameters."""

    if isinstance(radius, bool) or not isinstance(radius, int) or not (1 <= radius <= 25):
        raise ValueError(f'Invalid {stage} radius=`{radius}`: integer in range 1..25 expected.')
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not (0.0 < float(strength) <= 1.0):
        raise ValueError(f'Invalid {stage} strength=`{strength}`: number in range (0.0, 1.0] expected.')


def smooth_denoise(image3d: list[list[list[int]]], radius: int = 1, strength: float = 0.5) -> list[list[list[int]]]:
    """Stage 1: mild smoothing/denoising.

    Box-averaged image is blended with the source as
    ``out = source + strength * (blurred - source)``,
    so ``strength`` approaching 0 leaves the image nearly untouched
    and ``strength=1`` yields the full box blur.

    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :param int radius: blur radius in pixels, integer in range 1..25;
    :param float strength: blending strength in range (0.0, 1.0].
    :raises ValueError: Empty image or invalid radius/strength.
    :return: smoothed image of the same dimensions and pixel
        container type (list or tuple) as input.
    :rtype: list[list[list[int]]]

    """

    if not image3d or not image3d[0]:
        raise ValueError('Empty image cannot be smoothed.')
    _check_radius_strength(radius, strength, 'smoothing')

    maxval = _maxval_of(image3d)
    blurred = _box_blur(image3d, radius)
    tuple_pixels = type(image3d[0][0]) is tuple

    smoothed: list[list[list[int]]] = []
    for y, row in enumerate(image3d):
        out_row = []
        for x, pixel in enumerate(row):
            blurred_pixel = blurred[y][x]
            out_pixel = []
            for z, value in enumerate(pixel):
                delta = strength * (blurred_pixel[z] - value)
                out_value = value + _round_half_up(delta)
                out_pixel.append(min(max(out_value, 0), maxval))
            out_row.append(tuple(out_pixel) if tuple_pixels else out_pixel)
        smoothed.append(out_row)

    return smoothed


def sharpen_edges(image3d: list[list[list[int]]], radius: int = 1, strength: float = 0.5) -> list[list[list[int]]]:
    """Stage 3: edge sharpening (unsharp mask).

    High-frequency component is estimated as
    ``detail = source - box_blur(source)`` and added back as
    ``out = source + strength * detail``, clamped to the channel
    range detected from the image itself.

    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :param int radius: blur radius used for detail extraction,
        integer in range 1..25;
    :param float strength: sharpening strength in range (0.0, 1.0].
    :raises ValueError: Empty image or invalid radius/strength.
    :return: sharpened image of the same dimensions and pixel
        container type (list or tuple) as input.
    :rtype: list[list[list[int]]]

    """

    if not image3d or not image3d[0]:
        raise ValueError('Empty image cannot be sharpened.')
    _check_radius_strength(radius, strength, 'sharpening')

    maxval = _maxval_of(image3d)
    blurred = _box_blur(image3d, radius)
    tuple_pixels = type(image3d[0][0]) is tuple

    sharpened: list[list[list[int]]] = []
    for y, row in enumerate(image3d):
        out_row = []
        for x, pixel in enumerate(row):
            blurred_pixel = blurred[y][x]
            out_pixel = []
            for z, value in enumerate(pixel):
                delta = strength * (value - blurred_pixel[z])
                out_value = value + _round_half_up(delta)
                out_pixel.append(min(max(out_value, 0), maxval))
            out_row.append(tuple(out_pixel) if tuple_pixels else out_pixel)
        sharpened.append(out_row)

    return sharpened


def enhance_upscale(
    image3d: list[list[list[int]]],
    n: Literal[2, 3] = 2,
    use_smooth: bool = True,
    smooth_radius: int = 1,
    smooth_strength: float = 0.5,
    use_sharpen: bool = True,
    sharpen_radius: int = 1,
    sharpen_strength: float = 0.5,
) -> list[list[list[int]]]:
    """Three-stage enhancement upscale pipeline.

    Stage order is fixed and cannot be reordered:

    1. mild smoothing/denoising (:func:`smooth_denoise`),
       skipped when ``use_smooth`` is falsy;
    2. integer Scale2x/Scale3x upscale via :func:`scale2x`
       or :func:`scale3x`, always applied;
    3. edge sharpening (:func:`sharpen_edges`),
       skipped when ``use_sharpen`` is falsy.

    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :param int n: ``2`` or ``3``, upscale factor;
    :param bool use_smooth: enable stage 1 smoothing;
    :param int smooth_radius: smoothing radius, integer in range 1..25;
    :param float smooth_strength: smoothing strength, (0.0, 1.0];
    :param bool use_sharpen: enable stage 3 sharpening;
    :param int sharpen_radius: sharpening radius, integer in range 1..25;
    :param float sharpen_strength: sharpening strength, (0.0, 1.0].
    :raises ValueError: Empty image, invalid ``n`` or invalid
        radius/strength of either stage.
    :return: enhanced and rescaled image. With both outer stages
        switched off the result is pixel-identical to the plain
        :func:`scale2x`/:func:`scale3x` output.
    :rtype: list[list[list[int]]]

    """

    if not image3d or not image3d[0]:
        raise ValueError('Empty image cannot be enhanced.')
    if n != 2 and n != 3:
        raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')
    # ↓ Parameters are validated regardless of stage toggles, garbage in is always an error
    _check_radius_strength(smooth_radius, smooth_strength, 'smoothing')
    _check_radius_strength(sharpen_radius, sharpen_strength, 'sharpening')

    result = image3d

    # ↓ Stage 1: mild smoothing/denoising
    if use_smooth:
        result = smooth_denoise(result, smooth_radius, smooth_strength)

    # ↓ Stage 2: integer upscale, original per-pixel ScaleNx behavior
    if n == 2:
        result = scale2x(result)
    else:
        result = scale3x(result)

    # ↓ Stage 3: edge sharpening to compensate smoothing/upscale softness
    if use_sharpen:
        result = sharpen_edges(result, sharpen_radius, sharpen_strength)

    return result


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import scalenx

        help(scalenx)
