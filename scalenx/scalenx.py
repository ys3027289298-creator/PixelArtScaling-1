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


""" ╔════════════════════════════════════════════════════╗
    ║ xBR-style 2x rescaling with anti-aliased diagonals ║
    ╚════════════════════════════════════════════════════╝ """


def scale_xbr(
    image3d: list[list[list[int]]],
    blend_strength: float = 0.5,
    keep_palette: bool = False,
    border: str = 'clamp',
) -> tuple[str, list[list[list[int]]] | None]:
    """xBR-style 2x image rescale with anti-aliased diagonal edges.
    ----

    Unlike classic ScaleNx, which never introduces new colors, current
    method detects diagonal edges (corner judgement on a 3x3 neighbourhood,
    luma-based) and smooths them by weighted blending of the centre pixel
    with the corresponding diagonal neighbour.

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values); coordinate system match Photoshop, *i.e.*
        origin is top left corner, channels order is LA or RGBA from 0 to top;
    :type image3d: list[list[list[int]]]
    :param float blend_strength: base weight of diagonal neighbour in the
        blend, from ``0.0`` (no smoothing, plain 2x) to ``1.0``
        (full replacement); default ``0.5``;
    :param bool keep_palette: when ``True``, every blended pixel is snapped
        to the nearest colour found in the source 3x3 neighbourhood, so the
        result contains only colours present in the source image palette;
    :param str border: edge handling mode, ``'clamp'`` (replicate edge
        pixels) or ``'wrap'`` (toroidal wrap-around);
    :raises ValueError: on invalid ``blend_strength``, ``keep_palette``
        or ``border`` values;
    :return: tuple ``(status, image)``; ``status`` is ``'ok'`` and ``image``
        is the rescaled 3D nested list on success; for source images
        smaller than 3x3 status is ``'too_small'`` and image is ``None``.
    :rtype: tuple[str, list[list[list[int]]] | None]

    """

    # ↓ parameters validation: invalid values are reported as errors
    if isinstance(blend_strength, bool) or not isinstance(blend_strength, (int, float)):
        raise ValueError(f'blend_strength must be a number in [0.0, 1.0] range, got `{blend_strength!r}`')
    if not 0.0 <= blend_strength <= 1.0:
        raise ValueError(f'blend_strength must be within [0.0, 1.0] range, got `{blend_strength}`')
    if not isinstance(keep_palette, bool):
        raise ValueError(f'keep_palette must be bool, got `{keep_palette!r}`')
    if border not in ('clamp', 'wrap'):
        raise ValueError(f'border must be either `clamp` or `wrap`, got `{border!r}`')

    # ↓ determining source image size from list
    Y = len(image3d)
    X = len(image3d[0]) if Y else 0

    # ↓ explicit status for images too small for 3x3 neighbourhood
    if X < 3 or Y < 3:
        return ('too_small', None)

    Z = len(image3d[0][0])  # number of channels

    # ↓ border handling path: edge replicate (clamp) or toroidal (wrap)
    if border == 'wrap':
        ym = [(y - 1) % Y for y in range(Y)]
        yp = [(y + 1) % Y for y in range(Y)]
        xm = [(x - 1) % X for x in range(X)]
        xp = [(x + 1) % X for x in range(X)]
    else:
        ym = [max(y - 1, 0) for y in range(Y)]
        yp = [min(y + 1, Y - 1) for y in range(Y)]
        xm = [max(x - 1, 0) for x in range(X)]
        xp = [min(x + 1, X - 1) for x in range(X)]

    # ↓ luma (x1000) map, used for fast edge direction judgement
    if Z >= 3:
        lum = [[299 * px[0] + 587 * px[1] + 114 * px[2] for px in row] for row in image3d]
    else:
        lum = [[1000 * px[0] for px in row] for row in image3d]

    def _weight(l_diag: int, l_side1: int, l_side2: int, l_e: int) -> float:
        """Corner judgement: return blend weight for diagonal neighbour.

        Edge is considered to cross the corner diagonally when the two
        pixels adjacent to the diagonal are closer to each other than
        the diagonal pixel to the centre. Weight is full blend strength
        when the diagonal colour is connected to at least one adjacent
        side, and half strength for isolated diagonal pixels.
        """
        d_diag_e = abs(l_diag - l_e)
        if abs(l_side1 - l_side2) < d_diag_e:
            if abs(l_diag - l_side1) < d_diag_e or abs(l_diag - l_side2) < d_diag_e:
                return blend_strength
            return blend_strength * 0.5
        return 0.0

    def _mix(p: list[int], q: list[int], w: float, neighbourhood: tuple) -> list[int]:
        """Weighted blend of pixels p and q, optionally snapped to palette."""
        iw = 1.0 - w
        mixed = [int(p[c] * iw + q[c] * w + 0.5) for c in range(Z)]
        if keep_palette:
            best = None
            best_d = None
            for candidate in neighbourhood:
                d = sum(abs(mixed[c] - candidate[c]) for c in range(Z))
                if best_d is None or d < best_d:
                    best_d = d
                    best = candidate
            return list(best)
        return mixed

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
        └────┴────┘
        r1..r4 are E, optionally blended toward A, C, G, I respectively.
    """
    scaled_image: list[list[list[int]]] = []

    for y in range(Y):
        row_u = image3d[ym[y]]
        row_c = image3d[y]
        row_d = image3d[yp[y]]
        lum_u = lum[ym[y]]
        lum_c = lum[y]
        lum_d = lum[yp[y]]

        row_rez: list[list[int]] = []
        row_dvo: list[list[int]] = []

        for x in range(X):
            x_l = xm[x]
            x_r = xp[x]

            A = row_u[x_l]
            B = row_u[x]
            C = row_u[x_r]
            D = row_c[x_l]
            E = row_c[x]
            F = row_c[x_r]
            G = row_d[x_l]
            H = row_d[x]
            I = row_d[x_r]

            la = lum_u[x_l]
            lb = lum_u[x]
            lc = lum_u[x_r]
            ld = lum_c[x_l]
            le = lum_c[x]
            lf = lum_c[x_r]
            lg = lum_d[x_l]
            lh = lum_d[x]
            li = lum_d[x_r]

            w1 = _weight(la, lb, ld, le)
            w2 = _weight(lc, lb, lf, le)
            w3 = _weight(lg, ld, lh, le)
            w4 = _weight(li, lf, lh, le)

            if w1 or w2 or w3 or w4:
                neighbourhood = (A, B, C, D, E, F, G, H, I)
                r1 = _mix(E, A, w1, neighbourhood) if w1 else list(E)
                r2 = _mix(E, C, w2, neighbourhood) if w2 else list(E)
                r3 = _mix(E, G, w3, neighbourhood) if w3 else list(E)
                r4 = _mix(E, I, w4, neighbourhood) if w4 else list(E)
            else:
                r1 = r2 = r3 = r4 = list(E)

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
