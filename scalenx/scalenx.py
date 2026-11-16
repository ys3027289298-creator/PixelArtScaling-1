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

Batch processing with output buffer pooling::

    pool = scalenx.BufferPool(capacity=4, growth='eager')
    for scaled_image in scalenx.scale2x_batch(source_images, pool):
        ...  # consume scaled_image; buffer is recycled on next iteration

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


def _new_image(rows: int, cols: int) -> list[list[list[int]]]:
    """Allocate a zero-initialized nested-list image buffer of ``rows`` x ``cols`` pixels."""

    return [[None] * cols for _ in range(rows)]


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
    scaled_image = _new_image(2 * Y, 2 * X)

    _scale2x_into(image3d, scaled_image)

    return scaled_image


def _scale2x_into(image3d: list[list[list[int]]], scaled_image: list[list[list[int]]]) -> list[list[list[int]]]:
    """Scale2x rescale of ``image3d`` written into pre-allocated ``scaled_image``.

    ``scaled_image`` must be a nested list of 2*Y rows of 2*X cells each,
    where Y, X are ``image3d`` sizes. Every cell is overwritten.
    Returns the buffer passed in.
    """

    # ↓ determining source image size from list
    Y, X = (len(image3d), len(image3d[0]))

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
        row_rez = scaled_image[2 * y]
        row_dvo = scaled_image[2 * y + 1]

        A = image3d[max(y - 1, 0)][0]
        B = image3d[y][min(1, X - 1)]
        C = E = image3d[y][0]
        D = image3d[min(y + 1, Y - 1)][0]

        r1, r2, r3, r4 = _dva(A, B, C, D, E)

        row_rez[0] = r1
        row_rez[1] = r2
        row_dvo[0] = r3
        row_dvo[1] = r4

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

            row_rez[2 * x] = r1
            row_rez[2 * x + 1] = r2
            row_dvo[2 * x] = r3
            row_dvo[2 * x + 1] = r4

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
    scaled_image = _new_image(3 * Y, 3 * X)

    _scale3x_into(image3d, scaled_image)

    return scaled_image


def _scale3x_into(image3d: list[list[list[int]]], scaled_image: list[list[list[int]]]) -> list[list[list[int]]]:
    """Scale3x rescale of ``image3d`` written into pre-allocated ``scaled_image``.

    ``scaled_image`` must be a nested list of 3*Y rows of 3*X cells each,
    where Y, X are ``image3d`` sizes. Every cell is overwritten.
    Returns the buffer passed in.
    """

    # ↓ determining source image size from list
    Y, X = (len(image3d), len(image3d[0]))

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
        row_rez = scaled_image[3 * y]
        row_dvo = scaled_image[3 * y + 1]
        row_tre = scaled_image[3 * y + 2]

        A = B = image3d[max(y - 1, 0)][0]
        C = image3d[max(y - 1, 0)][min(1, X - 1)]
        D = E = image3d[y][0]
        F = image3d[y][min(1, X - 1)]
        G = H = image3d[min(y + 1, Y - 1)][0]
        I = image3d[min(y + 1, Y - 1)][min(1, X - 1)]

        r1, r2, r3, r4, r5, r6, r7, r8, r9 = _tri(A, B, C, D, E, F, G, H, I)

        row_rez[0] = r1
        row_rez[1] = r2
        row_rez[2] = r3
        row_dvo[0] = r4
        row_dvo[1] = r5
        row_dvo[2] = r6
        row_tre[0] = r7
        row_tre[1] = r8
        row_tre[2] = r9

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

            row_rez[3 * x] = r1
            row_rez[3 * x + 1] = r2
            row_rez[3 * x + 2] = r3
            row_dvo[3 * x] = r4
            row_dvo[3 * x + 1] = r5
            row_dvo[3 * x + 2] = r6
            row_tre[3 * x] = r7
            row_tre[3 * x + 1] = r8
            row_tre[3 * x + 2] = r9

    """if type(image3d[0][0]) is tuple:
        print(_tri.cache_info())"""

    return scaled_image


# ↑ rescaling three times finished


""" ╔════════════════════════════════════════════╗
    ║ Output buffer pool for batch processing    ║
    ╚════════════════════════════════════════════╝ """


class PoolExhaustedError(RuntimeError):
    """Raised when a :class:`BufferPool` configured with ``on_exhaust='raise'``
    cannot provide a buffer because its ``max_buffers`` limit is reached."""


class BufferPool:
    """Pool of pre-allocated output image buffers for batch rescaling.

    Buffers are nested lists of ``rows`` x ``cols`` cells, keyed by target
    shape, created once and reused across batch items to avoid per-image
    allocation overhead.

    :param int capacity: maximum number of *idle* buffers retained per target
        shape; buffers released beyond this are dropped to the garbage
        collector. ``0`` disables reuse (every acquire allocates fresh).
    :param str growth: ``'lazy'`` (default) creates exactly one new buffer on
        a cache miss; ``'eager'`` pre-allocates buffers for the requested
        shape in one go until its idle count reaches ``capacity``.
    :param str on_exhaust: behaviour when a new buffer is needed but the
        ``max_buffers`` limit is already reached (pool exhausted):
        ``'allocate'`` (default) silently falls back to a fresh, untracked
        buffer (results stay correct, it just is not reused);
        ``'raise'`` raises :class:`PoolExhaustedError`.
    :param max_buffers: hard cap on the total number of pool-managed buffers
        (checked-out plus idle). ``None`` (default) means no limit.
    :type max_buffers: int or None
    :param max_buffer_cells: buffers larger than ``rows * cols`` cells are
        never pooled: ``acquire`` returns a fresh untracked buffer and
        ``release`` of it is a no-op. Keeps oversized outputs from being
        retained in memory. ``None`` (default) means no limit.
    :type max_buffer_cells: int or None

    Counters (readable attributes): ``created`` (pool buffers allocated),
    ``reused`` (acquires served from the idle cache), ``fallbacks``
    (untracked buffers handed out), ``dropped`` (buffers discarded on
    release), ``checked_out`` (buffers currently in use).
    """

    def __init__(self, capacity: int = 8, growth: str = 'lazy', on_exhaust: str = 'allocate', max_buffers: int | None = None, max_buffer_cells: int | None = None):
        if capacity < 0:
            raise ValueError(f'capacity must be >= 0, got {capacity}')
        if growth not in ('lazy', 'eager'):
            raise ValueError(f"growth must be 'lazy' or 'eager', got '{growth}'")
        if on_exhaust not in ('allocate', 'raise'):
            raise ValueError(f"on_exhaust must be 'allocate' or 'raise', got '{on_exhaust}'")
        self.capacity = capacity
        self.growth = growth
        self.on_exhaust = on_exhaust
        self.max_buffers = max_buffers
        self.max_buffer_cells = max_buffer_cells
        # ↓ (rows, cols) -> list of idle buffers of that shape
        self._idle: dict[tuple[int, int], list[list]] = {}
        # ↓ id(buffer) -> (rows, cols) for pool-managed checked-out buffers
        self._shapes: dict[int, tuple[int, int]] = {}
        self._managed = 0  # pool-managed buffers currently alive (idle + checked out)
        # ↓ statistics
        self.created = 0
        self.reused = 0
        self.fallbacks = 0
        self.dropped = 0
        self.checked_out = 0

    def _create(self, rows: int, cols: int) -> list:
        """Create one pool-managed buffer, assuming budget was checked."""
        self.created += 1
        self._managed += 1
        return _new_image(rows, cols)

    def _fallback(self, rows: int, cols: int) -> list:
        """Hand out a fresh untracked buffer (not reused, not limited)."""
        self.fallbacks += 1
        return _new_image(rows, cols)

    def acquire(self, rows: int, cols: int) -> list:
        """Get a buffer of ``rows`` x ``cols`` cells, pooled when possible.

        May raise :class:`PoolExhaustedError` if ``on_exhaust='raise'``
        and the ``max_buffers`` limit is reached.
        """
        if rows < 0 or cols < 0:
            raise ValueError(f'Buffer shape must be non-negative, got {rows}x{cols}')

        if self.max_buffer_cells is not None and rows * cols > self.max_buffer_cells:
            # ↓ oversized buffer: explicitly bypass the pool
            return self._fallback(rows, cols)

        shape = (rows, cols)
        idle = self._idle.get(shape)
        if idle:
            self.reused += 1
            self.checked_out += 1
            buffer = idle.pop()
            self._shapes[id(buffer)] = shape
            return buffer

        # ↓ cache miss: new buffer(s) needed
        if self.max_buffers is not None and self._managed >= self.max_buffers:
            # ↓ pool exhausted: explicit behaviour per on_exhaust policy
            if self.on_exhaust == 'raise':
                raise PoolExhaustedError(f'BufferPool exhausted: max_buffers={self.max_buffers} reached, no idle buffer of shape {shape}')
            return self._fallback(rows, cols)

        if self.growth == 'eager' and self.capacity > 0:
            # ↓ pre-allocate up to capacity idle buffers for this shape in one go
            budget = self.capacity if self.max_buffers is None else min(self.capacity, self.max_buffers - self._managed)
            stock = [self._create(rows, cols) for _ in range(budget - 1)]
            self._idle.setdefault(shape, []).extend(stock)

        self.checked_out += 1
        buffer = self._create(rows, cols)
        self._shapes[id(buffer)] = shape
        return buffer

    def release(self, buffer: list) -> None:
        """Return a buffer previously obtained from :meth:`acquire` to the pool.

        Releasing an untracked (fallback) or already-released buffer is
        a no-op.
        """
        shape = self._shapes.pop(id(buffer), None)
        if shape is None:
            return
        self.checked_out -= 1
        idle = self._idle.setdefault(shape, [])
        if len(idle) < self.capacity:
            idle.append(buffer)
        else:
            # ↓ idle cache for this shape is full: drop to garbage collector
            self.dropped += 1
            self._managed -= 1

    def prealloc(self, rows: int, cols: int, count: int) -> int:
        """Pre-allocate up to ``count`` idle buffers of ``rows`` x ``cols``.

        Bounded by ``capacity`` per shape and by ``max_buffers`` overall.
        Returns the number of buffers actually created.
        """
        if self.max_buffer_cells is not None and rows * cols > self.max_buffer_cells:
            return 0
        shape = (rows, cols)
        idle = self._idle.setdefault(shape, [])
        made = 0
        while len(idle) < self.capacity and made < count:
            if self.max_buffers is not None and self._managed >= self.max_buffers:
                break
            idle.append(self._create(rows, cols))
            made += 1
        return made

    def clear(self) -> None:
        """Drop all idle buffers. Checked-out buffers stay valid and may
        still be released afterwards (they will be re-pooled as usual)."""
        dropped = sum(len(buffers) for buffers in self._idle.values())
        self._idle.clear()
        self._managed -= dropped
        self.dropped += dropped


def scale2x_batch(images3d, pool: BufferPool | None = None):
    """Generator rescaling a sequence of images with Scale2x.

    :param images3d: iterable of source images (see :func:`scale2x`);
    :param pool: optional :class:`BufferPool` used to recycle output buffers.
        With ``pool=None`` every image is scaled into a freshly allocated
        list, exactly as ``map(scale2x, images3d)`` would do.

    .. warning:: When a pool is used, the yielded image is a recycled pool
        buffer: it stays valid only until the generator is advanced to the
        next item. Copy it (or keep ``pool=None``) if you need to retain
        results.
    """
    for image3d in images3d:
        if pool is None:
            yield scale2x(image3d)
        else:
            buffer = pool.acquire(2 * len(image3d), 2 * len(image3d[0]))
            try:
                _scale2x_into(image3d, buffer)
                yield buffer
            finally:
                pool.release(buffer)


def scale3x_batch(images3d, pool: BufferPool | None = None):
    """Generator rescaling a sequence of images with Scale3x.

    Parameters and buffer-recycling contract are the same as
    for :func:`scale2x_batch`.
    """
    for image3d in images3d:
        if pool is None:
            yield scale3x(image3d)
        else:
            buffer = pool.acquire(3 * len(image3d), 3 * len(image3d[0]))
            try:
                _scale3x_into(image3d, buffer)
                yield buffer
            finally:
                pool.release(buffer)


# ↓ Dummy stub for standalone execution attempt
if __name__ == '__main__':
    print('Module to be imported, not run as standalone.')
    need_help = input('Would you like to read some help (y/n)?')
    if need_help.startswith(('y', 'Y')):
        import scalenx

        help(scalenx)
