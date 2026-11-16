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


""" ╔════════════════════════════════════════════════════════════════╗
    ║ Bilinear and anisotropic (isophote-directed) image upscaling.  ║
    ╚════════════════════════════════════════════════════════════════╝ """


def _channel_count(image3d: list[list[list[int]]]) -> int:
    """Number of channels per pixel (1 for L/I, 2 for LA/IA, 3 or 4 for RGB(A))."""
    return len(image3d[0][0])


def _maxval(image3d: list[list[list[int]]], channels: int) -> int:
    """Largest channel value present in the image; 1-bit images return 1."""
    peak = 0
    for row in image3d:
        for pixel in row:
            for c in range(channels):
                if pixel[c] > peak:
                    peak = pixel[c]
    return peak if peak > 1 else 1


def _clamp(value: float, maxval: int) -> int:
    """Round and clip an interpolated channel value to the valid range."""
    rounded = int(value + 0.5) if value >= 0 else -int(-value + 0.5)
    if rounded < 0:
        return 0
    if rounded > maxval:
        return maxval
    return rounded


def bilinear_scale(image3d: list[list[list[int]]], factor: float) -> list[list[list[int]]]:
    """Isotropic bilinear image rescale.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :type image3d: list[list[list[int]]]
    :param float factor: upscale factor, must be >= 1.
    :raises ValueError: ``factor`` below 1.
    :return: 3D nested list of the same structure as input,
        rescaled in X and Y directions using bilinear interpolation,
        with "repeat edge" boundary mode.
    :rtype: list[list[list[int]]]

    """
    if factor < 1:
        raise ValueError(f'Attempt to use scale factor=`{factor}`.\nOnly factor >= 1 allowed.')

    Y, X = (len(image3d), len(image3d[0]))
    channels = _channel_count(image3d)
    maxval = _maxval(image3d, channels)

    out_X = int(round(X * factor))
    out_Y = int(round(Y * factor))
    # Scale of the coordinate mapping; 1:1 pixels keep their exact value.
    scale_x = (X - 1) / (out_X - 1) if out_X > 1 else 0.0
    scale_y = (Y - 1) / (out_Y - 1) if out_Y > 1 else 0.0

    scaled_image: list[list[list[int]]] = []

    for oy in range(out_Y):
        sy = oy * scale_y
        y0 = int(sy)
        ty = sy - y0
        y1 = min(y0 + 1, Y - 1)
        row_top = image3d[y0]
        row_bottom = image3d[y1]
        out_row: list[list[int]] = []

        for ox in range(out_X):
            sx = ox * scale_x
            x0 = int(sx)
            tx = sx - x0
            x1 = min(x0 + 1, X - 1)

            w00 = (1.0 - tx) * (1.0 - ty)
            w01 = tx * (1.0 - ty)
            w10 = (1.0 - tx) * ty
            w11 = tx * ty

            p00 = row_top[x0]
            p01 = row_top[x1]
            p10 = row_bottom[x0]
            p11 = row_bottom[x1]

            pixel = [
                _clamp(w00 * p00[c] + w01 * p01[c] + w10 * p10[c] + w11 * p11[c], maxval)
                for c in range(channels)
            ]
            out_row.append(pixel)
        scaled_image.append(out_row)

    return scaled_image


def _structure_tensor(image3d: list[list[list[int]]], Y: int, X: int, channels: int, window: int):
    """Per-pixel Sobel structure tensor components (Jxx, Jyy, Jxy).

    Gradients are averaged over the color channels (luminance-style),
    then squared and summed over a (2*window+1) neighborhood, so the
    dominant orientation is robust against single-pixel noise.
    "Repeat edge" boundary mode is used throughout.
    """
    gx = [[0.0] * X for _ in range(Y)]
    gy = [[0.0] * X for _ in range(Y)]

    for y in range(Y):
        ym = y - 1 if y > 0 else 0
        yp = y + 1 if y < Y - 1 else Y - 1
        row_m = image3d[ym]
        row_c = image3d[y]
        row_p = image3d[yp]
        for x in range(X):
            xm = x - 1 if x > 0 else 0
            xp = x + 1 if x < X - 1 else X - 1
            sx = sy = 0.0
            for c in range(channels):
                # Sobel 3x3 operators
                sx += (
                    row_m[xp][c] + 2 * row_c[xp][c] + row_p[xp][c]
                    - row_m[xm][c] - 2 * row_c[xm][c] - row_p[xm][c]
                )
                sy += (
                    row_p[xm][c] + 2 * row_p[x][c] + row_p[xp][c]
                    - row_m[xm][c] - 2 * row_m[x][c] - row_m[xp][c]
                )
            gx[y][x] = sx / channels
            gy[y][x] = sy / channels

    jxx = [[0.0] * X for _ in range(Y)]
    jyy = [[0.0] * X for _ in range(Y)]
    jxy = [[0.0] * X for _ in range(Y)]

    for y in range(Y):
        for x in range(X):
            sxx = syy = sxy = 0.0
            for dy in range(-window, window + 1):
                yy = y + dy
                if yy < 0:
                    yy = 0
                elif yy >= Y:
                    yy = Y - 1
                for dx in range(-window, window + 1):
                    xx = x + dx
                    if xx < 0:
                        xx = 0
                    elif xx >= X:
                        xx = X - 1
                    ex = gx[yy][xx]
                    ey = gy[yy][xx]
                    sxx += ex * ex
                    syy += ey * ey
                    sxy += ex * ey
            jxx[y][x] = sxx
            jyy[y][x] = syy
            jxy[y][x] = sxy

    return jxx, jyy, jxy


def _sample_channel(
    image3d: list[list[list[int]]],
    X: int,
    Y: int,
    px: float,
    py: float,
    ch: int,
) -> float:
    """Bilinear sample of one channel at fractional position (px, py).

    Coordinates outside the image are clamped ("repeat edge" mode).
    """
    if px < 0.0:
        px = 0.0
    elif px > X - 1.0:
        px = X - 1.0
    if py < 0.0:
        py = 0.0
    elif py > Y - 1.0:
        py = Y - 1.0
    ix = int(px)
    iy = int(py)
    fx = px - ix
    fy = py - iy
    ix1 = min(ix + 1, X - 1)
    iy1 = min(iy + 1, Y - 1)
    row0 = image3d[iy]
    row1 = image3d[iy1]
    return (
        (1.0 - fx) * (1.0 - fy) * row0[ix][ch]
        + fx * (1.0 - fy) * row0[ix1][ch]
        + (1.0 - fx) * fy * row1[ix][ch]
        + fx * fy * row1[ix1][ch]
    )


def anisotropic_scale(
    image3d: list[list[list[int]]],
    factor: float,
    window: int = 1,
    strength: float = 1.0,
) -> list[list[list[int]]]:
    """Anisotropic isophote-directed image rescale.
    ----

    The local gradient direction is estimated per source pixel with Sobel
    operators and a structure tensor aggregated over a
    ``2*window+1`` square direction window. Interpolation is then performed
    *along the isophote* (the line tangent to the level curve, i.e.
    perpendicular to the gradient), while variation across the edge is
    preserved: the stronger the local anisotropy, the more the directional
    term dominates over plain bilinear interpolation.

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values);
    :type image3d: list[list[list[int]]]
    :param float factor: upscale factor, must be >= 1;
    :param int window: half-size of the orientation averaging window
        (``window=1`` -> 3x3, ``window=2`` -> 5x5, ...);
    :param float strength: blending strength between 0 (plain bilinear)
        and 1 (full directional correction);
    :raises ValueError: ``factor`` below 1, or invalid ``window``/``strength``.
    :return: 3D nested list of the same structure as input,
        rescaled in X and Y directions using anisotropic interpolation,
        with "repeat edge" boundary mode.
    :rtype: list[list[list[int]]]

    """
    if factor < 1:
        raise ValueError(f'Attempt to use scale factor=`{factor}`.\nOnly factor >= 1 allowed.')
    if not isinstance(window, int) or window < 0:
        raise ValueError(f'Attempt to use window=`{window}`.\nOnly non-negative int allowed.')
    if strength < 0.0 or strength > 1.0:
        raise ValueError(f'Attempt to use strength=`{strength}`.\nOnly values in [0, 1] allowed.')

    Y, X = (len(image3d), len(image3d[0]))
    channels = _channel_count(image3d)
    maxval = _maxval(image3d, channels)

    # No actual rescaling: return a structurally independent copy.
    if factor == 1:
        return [[[v for v in pixel] for pixel in row] for row in image3d]

    out_X = int(round(X * factor))
    out_Y = int(round(Y * factor))
    scale_x = (X - 1) / (out_X - 1)
    scale_y = (Y - 1) / (out_Y - 1)

    # Orientation field: the structure tensor's minor eigenvector points
    # along the isophote (perpendicular to the gradient); coherence measures
    # how strongly one direction dominates. Precomputed per source pixel.
    jxx, jyy, jxy = _structure_tensor(image3d, Y, X, channels, window)
    coherence = [[0.0] * X for _ in range(Y)]
    tangent_x = [[1.0] * X for _ in range(Y)]
    tangent_y = [[0.0] * X for _ in range(Y)]

    for y in range(Y):
        for x in range(X):
            a = jxx[y][x]
            b = jyy[y][x]
            c = jxy[y][x]
            trace = a + b
            delta = ((a - b) * (a - b) + 4.0 * c * c) ** 0.5
            lambda_min = 0.5 * (trace - delta)
            lambda_max = 0.5 * (trace + delta)
            if trace > 0.0:
                coherence[y][x] = strength * (lambda_max - lambda_min) / trace
                tvx = c
                tvy = lambda_min - a
                tnorm = (tvx * tvx + tvy * tvy) ** 0.5
                if tnorm > 1e-12:
                    tangent_x[y][x] = tvx / tnorm
                    tangent_y[y][x] = tvy / tnorm

    scaled_image: list[list[list[int]]] = []

    for oy in range(out_Y):
        sy = oy * scale_y
        y0 = int(sy)
        ty = sy - y0
        y1 = min(y0 + 1, Y - 1)
        row_top = image3d[y0]
        row_bottom = image3d[y1]
        coh_row = coherence[y0]
        tvx_row = tangent_x[y0]
        tvy_row = tangent_y[y0]
        out_row: list[list[int]] = []

        for ox in range(out_X):
            sx = ox * scale_x
            x0 = int(sx)
            tx = sx - x0
            x1 = min(x0 + 1, X - 1)

            p00 = row_top[x0]
            p01 = row_top[x1]
            p10 = row_bottom[x0]
            p11 = row_bottom[x1]

            # ↓ bilinear base value
            w00 = (1.0 - tx) * (1.0 - ty)
            w01 = tx * (1.0 - ty)
            w10 = (1.0 - tx) * ty
            w11 = tx * ty

            k = coh_row[x0]
            tvx = tvx_row[x0]
            tvy = tvy_row[x0]

            pixel: list[int] = []
            for ch in range(channels):
                base = w00 * p00[ch] + w01 * p01[ch] + w10 * p10[ch] + w11 * p11[ch]
                if k <= 0.0:
                    pixel.append(_clamp(base, maxval))
                else:
                    # Directional term: sample one pixel to either side of
                    # the *fractional* output position along the isophote and
                    # average. Anchoring at the exact output position keeps
                    # sub-pixel geometry (no block staircase), while never
                    # mixing neighbors across the edge preserves cross-edge
                    # contrast; along the isophote values stay smooth.
                    # Both samples, hence the blend, stay bounded by the
                    # source value range.
                    isophote = 0.5 * (
                        _sample_channel(image3d, X, Y, sx + tvx, sy + tvy, ch)
                        + _sample_channel(image3d, X, Y, sx - tvx, sy - tvy, ch)
                    )
                    pixel.append(_clamp((1.0 - k) * base + k * isophote, maxval))
            out_row.append(pixel)
        scaled_image.append(out_row)

    return scaled_image


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import scalenx

        help(scalenx)
