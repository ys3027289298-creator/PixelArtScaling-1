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


""" ╔════════════════════════════════════════════╗
    ║ xBR-style 2x rescaling with anti-aliasing  ║
    ╚════════════════════════════════════════════╝ """


def xbr2x(image3d: list[list[list[int]]], blend: float = 0.5, keep_palette: bool = False) -> tuple[str, list[list[list[int]]] | None]:
    """xBR-style 2x image rescale with anti-aliased diagonal edges.
    ----

    Edge and corner detection uses the same four-neighbourhood
    (up, right, left, down) decision tree as Scale2x; however,
    instead of hard-copying the "winning" neighbour into output,
    the neighbour colour is mixed into the central pixel colour
    with configurable weight, producing smooth anti-aliased
    transitions along diagonal edges instead of staircase steps.

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values); coordinate system match Photoshop, *i.e.*
        origin is top left corner, channels order is LA or RGBA from 0 to top;
    :type image3d: list[list[list[int]]]
    :param blend: anti-aliasing mix strength, number from 0.0 (no mixing,
        output equal to nearest neighbour) to 1.0 (full neighbour colour,
        output equal to Scale2x);
    :type blend: float
    :param keep_palette: if True, every mixed colour is snapped to the nearest
        colour from the source pixel 3x3 neighbourhood, so that output
        contains source palette colours only;
    :type keep_palette: bool
    :raises TypeError: ``blend`` is not a number, or ``keep_palette`` is not a bool;
    :raises ValueError: ``blend`` is out of [0.0, 1.0] range, or ``image3d``
        is empty or ragged;
    :return: tuple ``(status, image)``. ``status='ok'`` with rescaled image
        for source 3x3 and bigger; ``status='too_small'`` with ``image=None``
        for source smaller than 3x3 in any dimension.
    :rtype: tuple[str, list[list[list[int]]] | None]

    """

    if isinstance(blend, bool) or not isinstance(blend, (int, float)):
        raise TypeError(f'blend must be a number in [0.0, 1.0] range, got {type(blend).__name__}')
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f'blend must be in [0.0, 1.0] range, got {blend}')
    if not isinstance(keep_palette, bool):
        raise TypeError(f'keep_palette must be a bool, got {type(keep_palette).__name__}')
    if not isinstance(image3d, list) or len(image3d) == 0 or not isinstance(image3d[0], list) or len(image3d[0]) == 0:
        raise ValueError('image3d must be a non-empty list of non-empty row lists')

    # ↓ determining source image size from list
    Y, X = (len(image3d), len(image3d[0]))

    if any(len(row) != X for row in image3d):
        raise ValueError('image3d must be rectangular (all rows of the same length)')

    if X < 3 or Y < 3:
        return ('too_small', None)

    weight = float(blend)

    def _mix(base: list[int], other: list[int]) -> list[int]:
        """Anti-aliasing blend of ``other`` neighbour colour into ``base`` colour."""

        return [round(b + (o - b) * weight) for b, o in zip(base, other)]

    def _snap(mixed: list[int], candidates: list[tuple[int, ...]]) -> list[int]:
        """Snap ``mixed`` colour to the nearest of ``candidates`` palette colours."""

        best = candidates[0]
        best_dist = sum((m - c) ** 2 for m, c in zip(mixed, best))
        for cand in candidates[1:]:
            dist = sum((m - c) ** 2 for m, c in zip(mixed, cand))
            if dist < best_dist:
                best = cand
                best_dist = dist
        return list(best)

    """ Source around default pixel E
        ┌───┬───┬───┐
        │ A │ B │ C │
        ├───┼───┼───┤
        │ D │ E │ F │
        ├───┼───┼───┤
        │ G │ H │ I │
        └───┴───┴───┘

        Result
        ┌────┬────┐
        │ r1 │ r2 │
        ├────┼────┤
        │ r3 │ r4 │
        └────┴────┘ "Repeat edge" mode.
    """
    # ↓ starting new image list
    scaled_image: list[list[list[int]]] = []

    for y in range(Y):
        row_up = image3d[max(y - 1, 0)]
        row = image3d[y]
        row_dn = image3d[min(y + 1, Y - 1)]

        row_rez: list[list[int]] = []
        row_dvo: list[list[int]] = []

        for x in range(X):
            xl = max(x - 1, 0)
            xr = min(x + 1, X - 1)

            # ↓ four-neighbourhood edge/corner detection
            B = row_up[x]
            D = row[xl]
            E = row[x]
            F = row[xr]
            H = row_dn[x]

            r1 = r2 = r3 = r4 = E

            if B != H and D != F:
                # ↓ weighted anti-aliasing mix instead of hard copy
                if B == D:
                    r1 = _mix(E, B)
                if B == F:
                    r2 = _mix(E, B)
                if H == D:
                    r3 = _mix(E, H)
                if H == F:
                    r4 = _mix(E, H)

            if keep_palette and (r1 is not E or r2 is not E or r3 is not E or r4 is not E):
                # ↓ snapping mixed colours to nearest source palette colour
                A = row_up[xl]
                C = row_up[xr]
                G = row_dn[xl]
                I = row_dn[xr]
                candidates = list({tuple(p) for p in (A, B, C, D, E, F, G, H, I)})
                if r1 is not E:
                    r1 = _snap(r1, candidates)
                if r2 is not E:
                    r2 = _snap(r2, candidates)
                if r3 is not E:
                    r3 = _snap(r3, candidates)
                if r4 is not E:
                    r4 = _snap(r4, candidates)

            row_rez.extend((r1, r2))
            row_dvo.extend((r3, r4))

        scaled_image.append(row_rez)
        scaled_image.append(row_dvo)

    return ('ok', scaled_image)


# ↑ xBR-style rescaling two times finished


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import scalenx

        help(scalenx)
