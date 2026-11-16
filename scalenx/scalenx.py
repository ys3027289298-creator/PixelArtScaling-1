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
from math import isfinite, sqrt
from time import perf_counter

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


""" ╔═══════════════════════════════════════════════════════╗
    ║ Colour-count limited palette dithering (error         ║
    ║ diffusion), for output devices with fixed colour set  ║
    ╚═══════════════════════════════════════════════════════╝ """

# ↓ Diffusion kernels as (dx, dy, weight) triples. Positive dx means
#   "ahead in the current row", positive dy means "next rows".
DITHER_KERNELS: dict[str, tuple[tuple[int, int, float], ...]] = {
    'floyd-steinberg': (
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16),
    ),
    'jarvis-judice-ninke': (
        (1, 0, 7 / 48),
        (2, 0, 5 / 48),
        (-2, 1, 3 / 48),
        (-1, 1, 5 / 48),
        (0, 1, 7 / 48),
        (1, 1, 5 / 48),
        (2, 1, 3 / 48),
        (-2, 2, 1 / 48),
        (-1, 2, 3 / 48),
        (0, 2, 5 / 48),
        (1, 2, 3 / 48),
        (2, 2, 1 / 48),
    ),
    'stucki': (
        (1, 0, 8 / 42),
        (2, 0, 4 / 42),
        (-2, 1, 2 / 42),
        (-1, 1, 4 / 42),
        (0, 1, 8 / 42),
        (1, 1, 4 / 42),
        (2, 1, 2 / 42),
        (-2, 2, 1 / 42),
        (-1, 2, 2 / 42),
        (0, 2, 4 / 42),
        (1, 2, 2 / 42),
        (2, 2, 1 / 42),
    ),
    'atkinson': (
        (1, 0, 1 / 8),
        (2, 0, 1 / 8),
        (-1, 1, 1 / 8),
        (0, 1, 1 / 8),
        (1, 1, 1 / 8),
        (0, 2, 1 / 8),
    ),
    'burkes': (
        (1, 0, 8 / 32),
        (2, 0, 4 / 32),
        (-2, 1, 2 / 32),
        (-1, 1, 4 / 32),
        (0, 1, 8 / 32),
        (1, 1, 4 / 32),
        (2, 1, 2 / 32),
    ),
}

# ↓ Number of channels participating in colour matching.
#   Alpha (if present, i.e. LA and RGBA) is not a colour channel.
_COLOR_CHANNELS = {1: 1, 2: 1, 3: 3, 4: 3}


def _check_image(image3d: list[list[list[int]]]) -> tuple[int, int, int, int]:
    """Check image geometry, return ``(Y, X, Z, color_channels)``."""

    if not isinstance(image3d, list) or len(image3d) == 0 or not isinstance(image3d[0], list) or len(image3d[0]) == 0:
        raise ValueError('Image must be a non-empty list of non-empty row lists.')
    Y, X = len(image3d), len(image3d[0])
    Z = len(image3d[0][0]) if isinstance(image3d[0][0], (list, tuple)) else 0
    if Z not in _COLOR_CHANNELS:
        raise ValueError('Image pixels must be lists or tuples of 1 to 4 int channel values.')
    for row in image3d:
        if not isinstance(row, list) or len(row) != X:
            raise ValueError('Image rows must be lists of equal length (ragged image given).')
        for pixel in row:
            if not isinstance(pixel, (list, tuple)) or len(pixel) != Z or any(isinstance(channel, bool) or not isinstance(channel, int) for channel in pixel):
                raise ValueError(f'Image pixels must be lists or tuples of {Z} int channel values.')
    return Y, X, Z, _COLOR_CHANNELS[Z]


def _check_palette(palette, Z: int) -> list[tuple[int, ...]]:
    """Check palette entries, return palette as a list of tuples."""

    if not isinstance(palette, (list, tuple)) or len(palette) == 0:
        raise ValueError('Palette is empty: at least one colour entry is required.')
    checked: list[tuple[int, ...]] = []
    for entry in palette:
        if not isinstance(entry, (list, tuple)) or len(entry) != Z or any(isinstance(channel, bool) or not isinstance(channel, int) for channel in entry):
            raise ValueError(f'Every palette entry must be a list or tuple of {Z} int channel values.')
        checked.append(tuple(entry))
    return checked


def _resolve_kernel(kernel) -> tuple[tuple[int, int, float], ...]:
    """Return diffusion kernel as a tuple of ``(dx, dy, weight)`` triples."""

    if isinstance(kernel, str):
        key = kernel.strip().lower().replace('_', '-').replace(' ', '-')
        if key not in DITHER_KERNELS:
            raise ValueError(f'Unknown diffusion kernel `{kernel}`. Available: {", ".join(DITHER_KERNELS)}.')
        entries = DITHER_KERNELS[key]
    elif isinstance(kernel, (list, tuple)) and len(kernel) > 0:
        entries = kernel
    else:
        raise ValueError('Diffusion kernel must be a kernel name or a non-empty sequence of (dx, dy, weight) triples.')

    checked: list[tuple[int, int, float]] = []
    for item in entries:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError('Every diffusion kernel item must be an (dx, dy, weight) triple.')
        dx, dy, weight = item
        if isinstance(dx, bool) or isinstance(dy, bool) or not isinstance(dx, int) or not isinstance(dy, int):
            raise ValueError('Diffusion kernel offsets must be ints.')
        if dy < 0 or (dy == 0 and dx <= 0):
            raise ValueError('Diffusion kernel offsets must point to not yet processed pixels (dy > 0, or dy == 0 and dx > 0).')
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not isfinite(weight) or weight < 0:
            raise ValueError('Diffusion kernel weights must be finite non-negative numbers.')
        checked.append((dx, dy, float(weight)))
    if sum(weight for _, _, weight in checked) <= 0:
        raise ValueError('Diffusion kernel must distribute a positive total weight.')
    return tuple(checked)


def _check_limits(palette: list[tuple[int, ...]], max_colors: int | None) -> int:
    """Check colour count limit against palette, return effective limit."""

    if max_colors is None:
        return len(palette)
    if isinstance(max_colors, bool) or not isinstance(max_colors, int) or max_colors < 1:
        raise ValueError(f'Colour limit must be a positive integer, got `{max_colors}`.')
    if len(palette) > max_colors:
        raise ValueError(f'Palette has {len(palette)} colours, exceeding colour limit {max_colors}.')
    return max_colors


def _nearest_index(color: list[float] | tuple[int, ...], palette: list[tuple[int, ...]], channels: int) -> int:
    """Index of palette colour nearest to ``color`` in squared RGB distance."""

    best_index = 0
    best_distance = None
    for index, entry in enumerate(palette):
        distance = 0.0
        for channel in range(channels):
            delta = color[channel] - entry[channel]
            distance += delta * delta
        if best_distance is None or distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _check_strength(strength) -> float:
    """Check diffusion strength, return it as float."""

    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not isfinite(strength) or strength < 0:
        raise ValueError(f'Diffusion strength must be a finite non-negative number, got `{strength}`.')
    return float(strength)


def _check_serpentine(serpentine) -> bool:
    """Check serpentine switch."""

    if not isinstance(serpentine, bool):
        raise ValueError(f'Serpentine switch must be a bool, got `{serpentine}`.')
    return serpentine


def _check_block_size(block_size) -> int:
    """Check error measurement block size."""

    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError(f'Error measurement block size must be a positive integer, got `{block_size}`.')
    return block_size


def quantize_nearest(image3d: list[list[list[int]]], palette, max_colors: int | None = None) -> list[list[list[int]]]:
    """Nearest-neighbour palette quantisation without dithering.
    ----

    Every pixel is replaced with the nearest palette entry, measured
    as squared Euclidean distance over colour channels (alpha ignored).
    Output contains palette colours only, so is limited to
    ``len(palette)`` colours.

    :param image3d: source image as list of lists (rows) of lists or tuples
        (pixels) of int (channel values), 1 to 4 channels per pixel;
    :param palette: non-empty list of palette entries, every entry
        has the same number of channels as image pixels;
    :param max_colors: maximum number of colours allowed on target device;
        ``None`` means ``len(palette)``;
    :raises ValueError: on invalid image, empty palette, or palette size
        exceeding ``max_colors``;
    :return: quantised image as list of lists of lists of int.

    """

    Y, X, Z, channels = _check_image(image3d)
    checked = _check_palette(palette, Z)
    _check_limits(checked, max_colors)

    return [[list(checked[_nearest_index(pixel, checked, channels)]) for pixel in row] for row in image3d]


def dither_to_palette(
    image3d: list[list[list[int]]],
    palette,
    max_colors: int | None = None,
    kernel: str | list[tuple[int, int, float]] | tuple[tuple[int, int, float], ...] = 'floyd-steinberg',
    strength: float = 1.0,
    serpentine: bool = True,
) -> list[list[list[int]]]:
    """Dither image to a fixed palette with error diffusion.
    ----

    Every pixel is matched to the nearest palette entry (squared
    Euclidean distance over colour channels, alpha ignored), and the
    matching error is diffused to not yet processed neighbours using
    the selected kernel and strength. This keeps the colour count
    within the target device limit while distributing quantisation
    error as high-frequency texture instead of flat bands.

    Alpha channel, if present (LA, RGBA), does not take part in
    matching, and the output pixel is the full palette entry.

    :param image3d: source image as list of lists (rows) of lists or tuples
        (pixels) of int (channel values), 1 to 4 channels per pixel;
    :param palette: non-empty list of palette entries, every entry
        has the same number of channels as image pixels;
    :param max_colors: maximum number of colours allowed on target device;
        ``None`` means ``len(palette)``;
    :param kernel: diffusion kernel, either a name from :data:`DITHER_KERNELS`
        or a sequence of ``(dx, dy, weight)`` triples pointing to
        not yet processed pixels;
    :param strength: error diffusion strength multiplier, non-negative;
        ``1.0`` is the standard kernel amount, ``0.0`` is equivalent
        to :func:`quantize_nearest`;
    :param serpentine: alternate scan direction row by row to suppress
        directional worm artifacts;
    :raises ValueError: on invalid image, empty palette, palette size
        exceeding ``max_colors``, invalid kernel, strength or switch;
    :return: dithered image as list of lists of lists of int,
        containing palette colours only.

    """

    Y, X, Z, channels = _check_image(image3d)
    checked = _check_palette(palette, Z)
    _check_limits(checked, max_colors)
    kernel_entries = _resolve_kernel(kernel)
    strength_value = _check_strength(strength)
    serpentine_value = _check_serpentine(serpentine)

    # ↓ working buffer keeps colour channels as floats for error accumulation
    work = [[[float(pixel[channel]) for channel in range(channels)] for pixel in row] for row in image3d]
    indices = [[0] * X for _ in range(Y)]

    for y in range(Y):
        step = 1 if (not serpentine_value or y % 2 == 0) else -1
        xs = range(X) if step == 1 else range(X - 1, -1, -1)
        for x in xs:
            color = work[y][x]
            index = _nearest_index(color, checked, channels)
            indices[y][x] = index
            for channel in range(channels):
                error = (color[channel] - checked[index][channel]) * strength_value
                if error == 0.0:
                    continue
                for dx, dy, weight in kernel_entries:
                    nx = x + dx * step
                    ny = y + dy
                    if 0 <= nx < X and ny < Y:
                        work[ny][nx][channel] += error * weight

    return [[list(checked[indices[y][x]]) for x in range(X)] for y in range(Y)]


def _error_metrics(source: list[list[list[int]]], result: list[list[list[int]]], channels: int, block_size: int) -> dict[str, float]:
    """Absolute error metrics, per pixel and per ``block_size`` block."""

    Y, X = len(source), len(source[0])
    total = 0.0
    total_squared = 0.0
    maximum = 0
    for y in range(Y):
        for x in range(X):
            for channel in range(channels):
                delta = abs(source[y][x][channel] - result[y][x][channel])
                total += delta
                total_squared += delta * delta
                if delta > maximum:
                    maximum = delta
    samples = Y * X * channels

    # ↓ block-averaged error: reflects perceived error once dithering
    #   texture is averaged by eye or by the target device
    block_total = 0.0
    block_count = 0
    for by in range(0, Y, block_size):
        for bx in range(0, X, block_size):
            block_pixels = (min(by + block_size, Y) - by) * (min(bx + block_size, X) - bx)
            for channel in range(channels):
                source_sum = 0
                result_sum = 0
                for y in range(by, min(by + block_size, Y)):
                    for x in range(bx, min(bx + block_size, X)):
                        source_sum += source[y][x][channel]
                        result_sum += result[y][x][channel]
                block_total += abs(source_sum - result_sum) / block_pixels
                block_count += 1

    return {
        'mean_abs_error': total / samples,
        'rmse': sqrt(total_squared / samples),
        'max_abs_error': float(maximum),
        'block_mean_abs_error': block_total / block_count,
        'block_size': float(block_size),
    }


def dither_report(
    image3d: list[list[list[int]]],
    palette,
    max_colors: int | None = None,
    kernel: str | list[tuple[int, int, float]] | tuple[tuple[int, int, float], ...] = 'floyd-steinberg',
    strength: float = 1.0,
    serpentine: bool = True,
    block_size: int = 4,
) -> dict:
    """Dither image and report error versus nearest-neighbour quantisation.
    ----

    :param image3d: source image as list of lists (rows) of lists or tuples
        (pixels) of int (channel values), 1 to 4 channels per pixel;
    :param palette: non-empty list of palette entries;
    :param max_colors: maximum number of colours allowed on target device;
        ``None`` means ``len(palette)``;
    :param kernel: diffusion kernel name or sequence, see :func:`dither_to_palette`;
    :param strength: error diffusion strength, see :func:`dither_to_palette`;
    :param serpentine: alternate scan direction row by row;
    :param block_size: side of square blocks used for block-averaged
        error measurement;
    :raises ValueError: on invalid image, empty palette, palette size
        exceeding ``max_colors``, invalid kernel, strength, switch
        or block size;
    :return: dict with source/output colour counts, palette and limit,
        dither timing, and error metrics for dithered and
        nearest-neighbour quantised images.

    """

    Y, X, Z, channels = _check_image(image3d)
    checked = _check_palette(palette, Z)
    limit = _check_limits(checked, max_colors)
    block = _check_block_size(block_size)

    start = perf_counter()
    dithered = dither_to_palette(image3d, checked, limit, kernel, strength, serpentine)
    dither_seconds = perf_counter() - start

    start = perf_counter()
    nearest = quantize_nearest(image3d, checked, limit)
    nearest_seconds = perf_counter() - start

    def _color_count(image: list[list[list[int]]]) -> int:
        return len({tuple(pixel[:channels]) for row in image for pixel in row})

    dither_metrics = _error_metrics(image3d, dithered, channels, block)
    nearest_metrics = _error_metrics(image3d, nearest, channels, block)
    improvement = nearest_metrics['mean_abs_error'] - dither_metrics['mean_abs_error']
    block_improvement = nearest_metrics['block_mean_abs_error'] - dither_metrics['block_mean_abs_error']

    return {
        'size': (X, Y),
        'channels': Z,
        'source_colors': _color_count(image3d),
        'output_colors': _color_count(dithered),
        'nearest_colors': _color_count(nearest),
        'palette_size': len(checked),
        'max_colors': limit,
        'kernel': kernel if isinstance(kernel, str) else 'custom',
        'strength': strength,
        'serpentine': serpentine,
        'dither_seconds': dither_seconds,
        'nearest_seconds': nearest_seconds,
        'dither_mean_abs_error': dither_metrics['mean_abs_error'],
        'dither_rmse': dither_metrics['rmse'],
        'dither_max_abs_error': dither_metrics['max_abs_error'],
        'dither_block_mean_abs_error': dither_metrics['block_mean_abs_error'],
        'nearest_mean_abs_error': nearest_metrics['mean_abs_error'],
        'nearest_rmse': nearest_metrics['rmse'],
        'nearest_max_abs_error': nearest_metrics['max_abs_error'],
        'nearest_block_mean_abs_error': nearest_metrics['block_mean_abs_error'],
        'mean_abs_error_improvement': improvement,
        'block_mean_abs_error_improvement': block_improvement,
        'block_size': block,
    }


# ↑ palette dithering finished


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import scalenx

        help(scalenx)
