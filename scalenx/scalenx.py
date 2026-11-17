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

import json
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path

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
    ║ Batch export support: perceptual fingerprint skip cache ║
    ╚════════════════════════════════════════════════════════╝

Batch export pipeline in three stages:

1. ``fingerprint(image3d, algo)`` computes a 64-bit perceptual
   fingerprint of an image nested list. Algorithm is configurable:
   ``'ahash'`` (8x8 average luma vs mean) or ``'dhash'`` (9x8 luma
   horizontal gradient). Both are pure Python, stdlib only.
2. ``fingerprint_distance(a, b)`` compares two fingerprints,
   returning normalized Hamming distance in [0.0, 1.0].
3. ``SkipCache`` persists fingerprints of previously exported images
   in one JSON file and decides whether a source image changed
   enough (distance > threshold) to be re-exported.

Store safety properties:

- Size upper bound: at most ``max_entries`` keys of at most
  ``MAX_KEY_LEN`` chars each; oldest entries are evicted LRU-style.
  With defaults the store file never exceeds
  ``64 + 1024 * (128 + 24) = 155712`` bytes (~152 KiB);
  see ``SkipCache.size_bound``.
- Corruption fallback: an unreadable or malformed store file is
  renamed to ``<name>.corrupt`` and the cache restarts empty
  (``SkipCache.recovered`` is set to True), so every image is
  re-exported once and the store is rebuilt on next ``save()``.
- Out-of-range threshold: values outside [0.0, 1.0] (including NaN)
  are clamped into range with an explicit ``RuntimeWarning``.
"""

FINGERPRINT_ALGOS = ('ahash', 'dhash')
"""Available perceptual fingerprint algorithms."""

DEFAULT_THRESHOLD = 0.0
"""Default distance threshold: skip only when fingerprints match exactly."""

MAX_ENTRIES = 1024
"""Default maximum number of fingerprint entries kept in the store."""

MAX_KEY_LEN = 128
"""Maximum length of an entry key, characters."""


def _luma_grid(image3d: list[list[list[int]]], gw: int, gh: int) -> list[list[int]]:
    """Reduce image to ``gw`` x ``gh`` grid of average luma (int, x1000)."""

    Y, X = (len(image3d), len(image3d[0]))
    grid: list[list[int]] = []
    for gy in range(gh):
        y0 = gy * Y // gh
        y1 = max((gy + 1) * Y // gh, y0 + 1)
        row: list[int] = []
        for gx in range(gw):
            x0 = gx * X // gw
            x1 = max((gx + 1) * X // gw, x0 + 1)
            total, count = 0, 0
            for y in range(y0, y1):
                for x in range(x0, x1):
                    pixel = image3d[y][x]
                    if len(pixel) >= 3:
                        total += 299 * pixel[0] + 587 * pixel[1] + 114 * pixel[2]
                    else:
                        total += 1000 * pixel[0]
                    count += 1
            row.append(total // count)
        grid.append(row)
    return grid


def fingerprint(image3d: list[list[list[int]]], algo: str = 'dhash') -> int:
    """Compute 64-bit perceptual fingerprint of an image nested list.
    ----

    :param image3d: 3D nested list (image) of lists (rows) of lists (pixels)
        of int (channel values), same structure as used by ``scale2x``;
    :param str algo: fingerprint algorithm, ``'ahash'`` or ``'dhash'``
        (see ``FINGERPRINT_ALGOS``);
    :raises ValueError: unknown fingerprint algorithm;
    :return: 64-bit fingerprint;
    :rtype: int

    """

    if algo == 'ahash':
        grid = _luma_grid(image3d, 8, 8)
        flat = [cell for row in grid for cell in row]
        mean = sum(flat) // 64
        fp = 0
        for cell in flat:
            fp = (fp << 1) | (cell >= mean)
        return fp
    elif algo == 'dhash':
        grid = _luma_grid(image3d, 9, 8)
        fp = 0
        for row in grid:
            for x in range(8):
                fp = (fp << 1) | (row[x] > row[x + 1])
        return fp
    else:
        raise ValueError(f'Unknown fingerprint algorithm `{algo}`\nOnly {FINGERPRINT_ALGOS} allowed.')


def fingerprint_distance(fp_a: int, fp_b: int) -> float:
    """Normalized Hamming distance between two 64-bit fingerprints, [0.0, 1.0]."""

    return bin((fp_a ^ fp_b) & 0xFFFFFFFFFFFFFFFF).count('1') / 64.0


def sanitize_threshold(threshold: float) -> float:
    """Clamp threshold into [0.0, 1.0]; out-of-range values (incl. NaN)
    are clamped with an explicit ``RuntimeWarning``."""

    t = float(threshold)
    if math.isnan(t):
        warnings.warn(f'Threshold {threshold} is NaN; reset to {DEFAULT_THRESHOLD}', RuntimeWarning, stacklevel=2)
        return DEFAULT_THRESHOLD
    if t < 0.0:
        warnings.warn(f'Threshold {t} below 0.0; clamped to 0.0', RuntimeWarning, stacklevel=2)
        return 0.0
    if t > 1.0:
        warnings.warn(f'Threshold {t} above 1.0; clamped to 1.0', RuntimeWarning, stacklevel=2)
        return 1.0
    return t


class SkipCache:
    """Persistent store of perceptual fingerprints for batch export skipping.
    ----

    :param path: path to JSON fingerprint store file;
    :param float threshold: distance threshold in [0.0, 1.0]; image is
        re-exported only when fingerprint distance from the stored one
        is greater than threshold. Out-of-range values are clamped
        with a warning (see ``sanitize_threshold``);
    :param str algo: fingerprint algorithm (see ``FINGERPRINT_ALGOS``);
    :param int max_entries: maximum number of entries kept; oldest
        entries are evicted beyond this limit, bounding the store file
        size to ``size_bound`` bytes.

    Usage::

        cache = SkipCache('export_fingerprints.json', threshold=0.0)
        for name, image3d in batch:
            if cache.changed(name, image3d):
                export(name, scaleNx(image3d, 2, False))
                cache.update(name)
        cache.save()

    """

    def __init__(self, path: str | Path, threshold: float = DEFAULT_THRESHOLD, algo: str = 'dhash', max_entries: int = MAX_ENTRIES):
        if algo not in FINGERPRINT_ALGOS:
            raise ValueError(f'Unknown fingerprint algorithm `{algo}`\nOnly {FINGERPRINT_ALGOS} allowed.')
        if not isinstance(max_entries, int) or isinstance(max_entries, bool) or max_entries < 1:
            raise ValueError(f'max_entries must be a positive int, got `{max_entries}`')
        self.path = Path(path)
        self.threshold = sanitize_threshold(threshold)
        self.algo = algo
        self.max_entries = max_entries
        self.recovered = False
        """True if the store file was corrupted and quarantined on load."""
        self._entries: dict[str, str] = {}
        self._memo: dict[str, int] = {}
        self._load()

    @property
    def size_bound(self) -> int:
        """Upper bound of the store file size, bytes."""

        return 64 + self.max_entries * (MAX_KEY_LEN + 24)

    def __len__(self) -> int:
        return len(self._entries)

    def _check_key(self, key: str) -> None:
        if not isinstance(key, str) or not key or len(key) > MAX_KEY_LEN:
            raise ValueError(f'Key must be a non-empty str of at most {MAX_KEY_LEN} chars, got `{key!r}`')

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            entries = data['entries']
            if not isinstance(entries, dict) or not all(isinstance(k, str) and isinstance(v, str) and len(v) == 16 and int(v, 16) >= 0 for k, v in entries.items()):
                raise ValueError('Malformed entries')
            self._entries = dict(entries)
        except FileNotFoundError:
            pass  # No store yet: normal first run
        except (OSError, ValueError, TypeError, KeyError, UnicodeDecodeError):
            # ↓ Corrupted or malformed store: quarantine it and restart
            #   empty, so that everything is re-exported once (safe default).
            try:
                os.replace(self.path, self.path.with_name(self.path.name + '.corrupt'))
            except OSError:
                pass
            self._entries = {}
            self.recovered = True
        # ↓ Enforcing entry limit even for a hand-grown store file
        while len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))

    def changed(self, key: str, image3d: list[list[list[int]]]) -> bool:
        """Return True if image differs from the stored fingerprint
        by more than ``threshold`` (or was never exported)."""

        self._check_key(key)
        fp = fingerprint(image3d, self.algo)
        self._memo[key] = fp
        old = self._entries.get(key)
        if old is None:
            return True
        return fingerprint_distance(fp, int(old, 16)) > self.threshold

    def update(self, key: str, image3d: list[list[list[int]]] | None = None) -> None:
        """Store fingerprint of successfully exported image.
        Reuses fingerprint memoized by ``changed`` unless ``image3d``
        is given explicitly."""

        self._check_key(key)
        if image3d is not None:
            fp = fingerprint(image3d, self.algo)
        else:
            fp = self._memo.get(key)
            if fp is None:
                raise ValueError(f'No memoized fingerprint for key `{key}`; pass image3d explicitly')
        if key in self._entries:
            del self._entries[key]  # Reinserted below as most recent
        self._entries[key] = f'{fp:016x}'
        while len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))

    def save(self) -> None:
        """Write store to disk atomically, enforcing the size bound."""

        text = ''
        while True:
            payload = {'version': 1, 'algo': self.algo, 'threshold': self.threshold, 'entries': self._entries}
            text = json.dumps(payload, separators=(',', ':'), sort_keys=True)
            if len(text.encode('utf-8')) <= self.size_bound or not self._entries:
                break
            self._entries.pop(next(iter(self._entries)))  # Evict oldest until it fits
        tmp_path = self.path.with_name(self.path.name + '.tmp')
        tmp_path.write_text(text, encoding='utf-8')
        os.replace(tmp_path, self.path)


# ↑ batch export skip cache finished


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import scalenx

        help(scalenx)
