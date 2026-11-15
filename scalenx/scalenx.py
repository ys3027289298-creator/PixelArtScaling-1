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
    ║ Pre-scaling brightness normalization: quantile clip-and-stretch ║
    ║ followed by gamma correction, both independently switchable.    ║
    ╚══════════════════════════════════════════════════════════════╝ """


def _percentile(sorted_values: list[int], percent: float) -> int:
    """Nearest-rank percentile of an already sorted list."""
    count = len(sorted_values)
    rank = int(percent * count / 100.0)
    if rank < 0:
        rank = 0
    elif rank >= count:
        rank = count - 1
    return sorted_values[rank]


def _is_real_number(value) -> bool:
    """True for numeric values excluding booleans, NaN and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in (float('inf'), float('-inf'))


def normalize_brightness(
    image3d: list[list[list[int]]],
    low_quantile: float = 0.5,
    high_quantile: float = 99.5,
    gamma: float = 1.0,
    value_max: int | None = None,
    use_stretch: bool = True,
    use_gamma: bool = True,
) -> dict:
    """Pre-scaling brightness normalization.
    ----

    Two independently switchable steps are chained:

    1. **Quantile clip-and-stretch** (``use_stretch``). All non-alpha
       channels of the whole image are pooled together; values below the
       ``low_quantile`` and above the ``high_quantile`` are clipped, and the
       range between the two quantiles is linearly stretched onto the full
       ``0..value_max`` range.
    2. **Gamma correction** (``use_gamma``). Each stretched value ``v`` is
       mapped as ``value_max * (v / value_max) ** gamma``. With
       ``gamma < 1.0`` the image gets brighter, with ``gamma > 1.0`` darker.

    Alpha channel (second channel in LA, fourth channel in RGBA) is never
    altered. The source image is never modified; a new image is returned.

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :type image3d: list[list[list[int]]]
    :param float low_quantile: lower clipping quantile in percent,
        ``0 <= low_quantile < high_quantile <= 100``;
    :param float high_quantile: upper clipping quantile in percent,
        ``0 <= low_quantile < high_quantile <= 100``;
    :param float gamma: gamma exponent, must be a finite number ``> 0``;
    :param value_max: maximum channel value (``255`` for 8-bit,
        ``65535`` for 16-bit). When ``None`` it is inferred from the data;
    :type value_max: int or None
    :param bool use_stretch: enables the quantile clip-and-stretch step;
    :param bool use_gamma: enables the gamma correction step;
    :raises ValueError: any parameter is out of its valid range or the
        image structure is invalid.
    :return: dictionary with keys:

        - ``status`` (``str``): ``'ok'``, ``'all_black'``, ``'all_white'``
          or ``'uniform'`` (constant non-extreme image; stretching skipped);
        - ``image``: normalized 3D nested list of the same structure
          as ``image3d``;
        - ``metrics`` (``dict``):

          - ``quantiles_before``: ``(low, high)`` quantiles of the pooled
            non-alpha channels before stretching;
          - ``quantiles_after``: same quantiles after stretching
            (``None`` when the stretch step is off or skipped);
          - ``contrast_gain``: population standard deviation after
            stretching minus before it (``None`` when off or skipped);
          - ``gamma_diff_pixels``: number of pixels differing after the
            gamma step versus before it (``None`` when gamma is off);
          - ``reversed_order_diff_pixels``: number of pixels differing
            between the normal pipeline (stretch then gamma) and the
            reversed pipeline (gamma then stretch); ``None`` unless both
            steps are enabled.

    :rtype: dict

    """

    # ↓ parameter validation
    if not isinstance(image3d, list) or not image3d or not isinstance(image3d[0], list) or not image3d[0]:
        raise ValueError('image3d must be a non-empty 3D nested list of pixels.')
    if not all(isinstance(row, list) and row for row in image3d):
        raise ValueError('image3d must be a non-empty 3D nested list of pixels.')
    channels = len(image3d[0][0])
    if not isinstance(image3d[0][0], (list, tuple)) or channels < 1:
        raise ValueError('Each pixel must be a non-empty list or tuple of int channel values.')
    for row in image3d:
        for pixel in row:
            if not isinstance(pixel, (list, tuple)) or len(pixel) != channels or not all(isinstance(channel, int) for channel in pixel):
                raise ValueError('All pixels must be equally sized sequences of int channel values.')

    if not isinstance(use_stretch, bool) or not isinstance(use_gamma, bool):
        raise ValueError('use_stretch and use_gamma must be bool.')

    if value_max is None:
        inferred_max = max(max(pixel) for row in image3d for pixel in row)
        if inferred_max <= 1:
            value_max = 1
        elif inferred_max <= 255:
            value_max = 255
        elif inferred_max <= 65535:
            value_max = 65535
        else:
            raise ValueError('Cannot infer value_max: channel value exceeds 65535, pass value_max explicitly.')
    elif not isinstance(value_max, int) or isinstance(value_max, bool) or value_max < 1:
        raise ValueError('value_max must be a positive int or None.')

    for row in image3d:
        for pixel in row:
            if min(pixel) < 0 or max(pixel) > value_max:
                raise ValueError(f'All channel values must lie within 0..{value_max}.')

    if not _is_real_number(low_quantile) or not _is_real_number(high_quantile):
        raise ValueError('Quantiles must be finite int or float numbers.')
    if not (0.0 <= low_quantile < high_quantile <= 100.0):
        raise ValueError('Require 0 <= low_quantile < high_quantile <= 100.')

    if not _is_real_number(gamma) or gamma <= 0:
        raise ValueError('gamma must be a finite number greater than 0.')

    # ↓ indices of channels actually processed (alpha channel is skipped)
    processed_indices = tuple(index for index in range(channels) if not (channels in (2, 4) and index == channels - 1))

    # ↓ pooling all non-alpha channel values of the whole image
    pooled = [pixel[index] for row in image3d for pixel in row for index in processed_indices]
    pooled_sorted = sorted(pooled)

    quantiles_before = (_percentile(pooled_sorted, low_quantile), _percentile(pooled_sorted, high_quantile))

    # ↓ explicit all-black / all-white states, returned unchanged
    if all(value == 0 for value in pooled):
        return {
            'status': 'all_black',
            'image': [[list(pixel) for pixel in row] for row in image3d],
            'metrics': {
                'quantiles_before': quantiles_before,
                'quantiles_after': None,
                'contrast_gain': None,
                'gamma_diff_pixels': None,
                'reversed_order_diff_pixels': None,
            },
        }
    if all(value == value_max for value in pooled):
        return {
            'status': 'all_white',
            'image': [[list(pixel) for pixel in row] for row in image3d],
            'metrics': {
                'quantiles_before': quantiles_before,
                'quantiles_after': None,
                'contrast_gain': None,
                'gamma_diff_pixels': None,
                'reversed_order_diff_pixels': None,
            },
        }

    def _clone(image):
        """Deep copy of an image of lists of lists of ints."""
        return [[list(pixel) for pixel in row] for row in image]

    # ↓ constant non-extreme image: stretch would divide by zero, skip it
    if pooled_sorted[0] == pooled_sorted[-1]:
        result_image = _clone(image3d)
        if use_gamma:
            for row in result_image:
                for pixel in row:
                    for index in processed_indices:
                        pixel[index] = int(round(value_max * (pixel[index] / value_max) ** gamma))
        return {
            'status': 'uniform',
            'image': result_image,
            'metrics': {
                'quantiles_before': quantiles_before,
                'quantiles_after': None,
                'contrast_gain': None,
                'gamma_diff_pixels': None,
                'reversed_order_diff_pixels': None,
            },
        }

    def _stretch_value(value: int) -> int:
        """Clip to quantile range and linearly stretch onto 0..value_max."""
        clipped = low
        if value > low:
            clipped = high if value > high else value
        stretched = (clipped - low) * value_max / (high - low)
        if stretched < 0:
            stretched = 0
        elif stretched > value_max:
            stretched = value_max
        return int(round(stretched))

    def _gamma_value(value: int) -> int:
        """Gamma-correct one channel value."""
        corrected = value_max * (value / value_max) ** gamma
        if corrected < 0:
            corrected = 0
        elif corrected > value_max:
            corrected = value_max
        return int(round(corrected))

    def _std(values: list[int]) -> float:
        """Population standard deviation."""
        mean = sum(values) / len(values)
        return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5

    def _count_different_pixels(image_a, image_b) -> int:
        """Count pixels differing in any processed channel."""
        count = 0
        for row_a, row_b in zip(image_a, image_b):
            for pixel_a, pixel_b in zip(row_a, row_b):
                if any(pixel_a[index] != pixel_b[index] for index in processed_indices):
                    count += 1
        return count

    low, high = quantiles_before
    # ↓ degenerate quantile collision (e.g. tiny images) falls back to min/max
    if low == high:
        low, high = pooled_sorted[0], pooled_sorted[-1]

    # ↓ step 1: quantile clip-and-stretch
    if use_stretch:
        stretched_image = _clone(image3d)
        stretched_pool = []
        for row in stretched_image:
            for pixel in row:
                for index in processed_indices:
                    pixel[index] = _stretch_value(pixel[index])
                    stretched_pool.append(pixel[index])
        stretched_sorted = sorted(stretched_pool)
        quantiles_after = (
            _percentile(stretched_sorted, low_quantile),
            _percentile(stretched_sorted, high_quantile),
        )
        contrast_gain = _std(stretched_pool) - _std(pooled)
    else:
        stretched_image = _clone(image3d)
        quantiles_after = None
        contrast_gain = None

    # ↓ step 2: gamma correction
    if use_gamma:
        pre_gamma_image = _clone(stretched_image)
        result_image = _clone(stretched_image)
        for row in result_image:
            for pixel in row:
                for index in processed_indices:
                    pixel[index] = _gamma_value(pixel[index])
        gamma_diff_pixels = _count_different_pixels(pre_gamma_image, result_image)
    else:
        result_image = stretched_image
        gamma_diff_pixels = None

    # ↓ reference pipeline with the two steps in reversed order
    if use_stretch and use_gamma:
        reversed_image = _clone(image3d)
        for row in reversed_image:
            for pixel in row:
                for index in processed_indices:
                    pixel[index] = _stretch_value(_gamma_value(pixel[index]))
        reversed_order_diff_pixels = _count_different_pixels(result_image, reversed_image)
    else:
        reversed_order_diff_pixels = None

    return {
        'status': 'ok',
        'image': result_image,
        'metrics': {
            'quantiles_before': quantiles_before,
            'quantiles_after': quantiles_after,
            'contrast_gain': contrast_gain,
            'gamma_diff_pixels': gamma_diff_pixels,
            'reversed_order_diff_pixels': reversed_order_diff_pixels,
        },
    }


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import scalenx

        help(scalenx)
