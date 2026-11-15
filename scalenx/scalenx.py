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


""" ╔════════════════════════════════════════════════════════╗
    ║ Repeated blocks pipeline: split → dedupe → index map → ║
    ║ sort by frequency → rebuild by index                   ║
    ╚════════════════════════════════════════════════════════╝ """


def _validate_geometry(image3d: list[list[list[int]]], block_size: int, pad: int) -> None:
    """Validate image and block geometry, raise ValueError on illegal sizes."""

    if not isinstance(image3d, list) or len(image3d) == 0 or not isinstance(image3d[0], list) or len(image3d[0]) == 0:
        raise ValueError(f'Image must be a non-empty list of non-empty row lists, got {type(image3d)}.')
    X = len(image3d[0])
    if any(not isinstance(row, list) or len(row) != X for row in image3d):
        raise ValueError('Image rows must be lists of equal length (ragged image given).')
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError(f'Block size must be a positive integer, got `{block_size}`.')
    if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
        raise ValueError(f'Edge padding must be a non-negative integer, got `{pad}`.')


def split_blocks(image3d: list[list[list[int]]], block_size: int = 8, pad: int = 1) -> list[list[list[list[int]]]]:
    """Split image into square blocks with edge padding.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :type image3d: list[list[list[int]]]
    :param int block_size: block core size, in pixels; positive integer;
    :param int pad: edge padding width, in pixels; non-negative integer.
        Padding is filled with replicated edge pixels ("repeat edge" mode),
        matching border handling of Scale2x/Scale3x;
    :raises ValueError: on illegal image, block size or padding;
    :return: flat list of blocks in row-major order. Every block is
        a ``(core_y + 2*pad) × (core_x + 2*pad)`` nested list,
        where core is ``block_size`` everywhere except image edges.
    :rtype: list[list[list[list[int]]]]

    """

    _validate_geometry(image3d, block_size, pad)

    Y, X = (len(image3d), len(image3d[0]))

    blocks: list[list[list[list[int]]]] = []
    for by in range(0, Y, block_size):
        core_y = min(block_size, Y - by)
        for bx in range(0, X, block_size):
            core_x = min(block_size, X - bx)
            block: list[list[list[int]]] = []
            for ry in range(-pad, core_y + pad):
                sy = min(max(by + ry, 0), Y - 1)
                row: list[list[int]] = []
                for rx in range(-pad, core_x + pad):
                    sx = min(max(bx + rx, 0), X - 1)
                    row.append(image3d[sy][sx])
                block.append(row)
            blocks.append(block)

    return blocks


# ↑ image split into padded blocks finished


def _block_key(block: list[list[list[int]]]) -> tuple:
    """Convert block to a hashable nested tuple."""

    return tuple(tuple(tuple(pixel) if isinstance(pixel, (list, tuple)) else pixel for pixel in row) for row in block)


def dedupe_blocks(blocks: list[list[list[list[int]]]]) -> tuple[list, list[int], list[int]]:
    """Deduplicate blocks, building an index map.
    ----

    :param blocks: flat list of blocks as returned by :func:`split_blocks`;
    :return: tuple of (unique blocks in first-occurrence order,
        per-block occurrence counts, index map) where index map
        has the same length and order as ``blocks`` and maps
        every source block position onto unique block index.
    :rtype: tuple[list, list[int], list[int]]

    """

    unique: list = []
    counts: list[int] = []
    index_map: list[int] = []
    registry: dict = {}

    for block in blocks:
        key = _block_key(block)
        index = registry.get(key)
        if index is None:
            index = len(unique)
            registry[key] = index
            unique.append(block)
            counts.append(0)
        counts[index] += 1
        index_map.append(index)

    return unique, counts, index_map


# ↑ blocks deduplicated, index map built


def sort_blocks_by_frequency(unique_blocks: list, counts: list[int], index_map: list[int]) -> tuple[list, list[int], list[int]]:
    """Sort unique blocks by occurrence frequency, most common first.
    ----

    :param unique_blocks: unique blocks as returned by :func:`dedupe_blocks`;
    :param counts: corresponding occurrence counts;
    :param index_map: index map as returned by :func:`dedupe_blocks`;
    :return: tuple of (sorted unique blocks, sorted counts, remapped
        index map) with equal-frequency ties resolved in favour
        of earlier first occurrence.
    :rtype: tuple[list, list[int], list[int]]

    """

    order = sorted(range(len(unique_blocks)), key=lambda i: (-counts[i], i))
    remap = {old: new for new, old in enumerate(order)}

    sorted_unique = [unique_blocks[old] for old in order]
    sorted_counts = [counts[old] for old in order]
    remapped_index_map = [remap[index] for index in index_map]

    return sorted_unique, sorted_counts, remapped_index_map


# ↑ most common blocks moved to the top


def rebuild_from_index(
    unique_blocks: list,
    index_map: list[int],
    height: int,
    width: int,
    block_size: int,
    pad: int = 0,
) -> list[list[list[int]]]:
    """Rebuild image from unique blocks using index map.
    ----

    :param unique_blocks: unique blocks, padded by ``pad`` on every side;
    :param list[int] index_map: block indices in row-major order;
    :param int height: target image height, in pixels;
    :param int width: target image width, in pixels;
    :param int block_size: block core size, in pixels; positive integer;
    :param int pad: padding width blocks carry, in pixels;
        padding is cropped away during rebuild;
    :raises ValueError: on illegal sizes or inconsistent index map;
    :return: rebuilt image as 3D nested list.
    :rtype: list[list[list[int]]]

    """

    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        raise ValueError(f'Image height must be a positive integer, got `{height}`.')
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError(f'Image width must be a positive integer, got `{width}`.')
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError(f'Block size must be a positive integer, got `{block_size}`.')
    if isinstance(pad, bool) or not isinstance(pad, int) or pad < 0:
        raise ValueError(f'Edge padding must be a non-negative integer, got `{pad}`.')

    grid_y = (height + block_size - 1) // block_size
    grid_x = (width + block_size - 1) // block_size
    if len(index_map) != grid_y * grid_x:
        raise ValueError(f'Index map length {len(index_map)} does not match block grid {grid_y}x{grid_x}.')
    if any(index < 0 or index >= len(unique_blocks) for index in index_map):
        raise ValueError('Index map refers to nonexistent unique block.')

    image3d: list[list[list[int]]] = []
    for by in range(grid_y):
        core_y = min(block_size, height - by * block_size)
        rows: list[list[list[int]]] = [[] for _ in range(core_y)]
        for bx in range(grid_x):
            block = unique_blocks[index_map[by * grid_x + bx]]
            core_x = min(block_size, width - bx * block_size)
            for ry in range(core_y):
                rows[ry].extend(block[pad + ry][pad : pad + core_x])
        image3d.extend(rows)

    return image3d


# ↑ image rebuilt from unique blocks by index


def scale_blocks(
    image3d: list[list[list[int]]],
    scaler=scale2x,
    factor: int = 2,
    block_size: int = 8,
    pad: int = 1,
) -> list[list[list[int]]]:
    """Scale image using repeated blocks: split → dedupe → index map →
    sort by frequency → scale unique blocks once → rebuild by index.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :param scaler: scaling function, e.g. :func:`scale2x` or :func:`scale3x`;
    :param int factor: scaling factor of ``scaler``; positive integer;
    :param int block_size: block core size, in pixels; positive integer;
    :param int pad: edge padding width, in pixels; non-negative integer.
        With ``pad >= 1`` result is pixel-identical to scaling
        the whole image with ``scaler`` directly;
    :raises ValueError: on illegal image, block size, padding or factor;
    :return: scaled image as 3D nested list.
    :rtype: list[list[list[int]]]

    """

    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError(f'Scaling factor must be a positive integer, got `{factor}`.')

    Y, X = (len(image3d), len(image3d[0]))

    # ↓ split → dedupe → index map → sort by frequency
    blocks = split_blocks(image3d, block_size, pad)
    unique, counts, index_map = dedupe_blocks(blocks)
    unique, counts, index_map = sort_blocks_by_frequency(unique, counts, index_map)

    # ↓ every unique block scaled only once
    scaled_unique = [scaler(block) for block in unique]

    # ↓ scaled image rebuilt by index
    return rebuild_from_index(scaled_unique, index_map, Y * factor, X * factor, block_size * factor, pad * factor)


# ↑ block-based scaling finished


def block_report(
    image3d: list[list[list[int]]],
    block_size: int = 8,
    pad: int = 1,
    target_size: tuple[int, int] = (1024, 1024),
) -> dict:
    """Run repeated blocks pipeline over image and report metrics.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :param int block_size: block core size, in pixels; positive integer;
    :param int pad: edge padding width, in pixels; non-negative integer;
    :param tuple[int, int] target_size: ``(height, width)`` of image
        used for rebuild timing benchmark;
    :raises ValueError: on illegal image, block size, padding or target size;
    :return: dict with metrics:

        - ``'blocks_total'``: total number of blocks;
        - ``'blocks_unique'``: number of unique blocks after dedupe;
        - ``'dedup_rate'``: deduplication rate, 1 - unique/total;
        - ``'rebuild_identical'``: True if image rebuilt by index
          is pixel-identical to source;
        - ``'rebuild_HxW_seconds'``: time of rebuilding
          a ``H × W`` image from index map, seconds.
    :rtype: dict

    """

    target_y, target_x = target_size
    if isinstance(target_y, bool) or not isinstance(target_y, int) or target_y < 1:
        raise ValueError(f'Target height must be a positive integer, got `{target_y}`.')
    if isinstance(target_x, bool) or not isinstance(target_x, int) or target_x < 1:
        raise ValueError(f'Target width must be a positive integer, got `{target_x}`.')

    Y, X = (len(image3d), len(image3d[0]))

    # ↓ split → dedupe → index map → sort by frequency
    blocks = split_blocks(image3d, block_size, pad)
    blocks_total = len(blocks)
    unique, counts, index_map = dedupe_blocks(blocks)
    unique, counts, index_map = sort_blocks_by_frequency(unique, counts, index_map)
    blocks_unique = len(unique)

    # ↓ rebuild by index and check pixel-exact consistency
    rebuilt = rebuild_from_index(unique, index_map, Y, X, block_size, pad)
    rebuild_identical = rebuilt == image3d

    # ↓ rebuild timing benchmark at target size
    full_y, full_x = (block_size + 2 * pad, block_size + 2 * pad)
    bench_blocks = [block for block in unique if len(block) == full_y and len(block[0]) == full_x]
    if not bench_blocks:
        # ↓ no full-size blocks in source; stretching most common block to full size by edge replication
        bench_blocks = [
            [
                [unique[0][min(ry, len(unique[0]) - 1)][min(rx, len(unique[0][0]) - 1)] for rx in range(full_x)]
                for ry in range(full_y)
            ]
        ]
    grid_y = (target_y + block_size - 1) // block_size
    grid_x = (target_x + block_size - 1) // block_size
    bench_index_map = [index % len(bench_blocks) for index in range(grid_y * grid_x)]

    start = perf_counter()
    rebuild_from_index(bench_blocks, bench_index_map, target_y, target_x, block_size, pad)
    bench_seconds = perf_counter() - start

    return {
        'blocks_total': blocks_total,
        'blocks_unique': blocks_unique,
        'dedup_rate': 1.0 - blocks_unique / blocks_total,
        'rebuild_identical': rebuild_identical,
        f'rebuild_{target_y}x{target_x}_seconds': bench_seconds,
    }


# ↑ repeated blocks pipeline metrics finished


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import scalenx

        help(scalenx)
