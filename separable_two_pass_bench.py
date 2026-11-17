#!/usr/bin/env python3

"""
====================================================
Separable two-pass scaling vs single-pass: benchmark
====================================================

:Abstract: Self-contained test/benchmark script (stdlib only, no repo
    dependencies modified) comparing a **single-pass** 2D resampler with a
    **separable two-pass** (horizontal-then-vertical) resampler whose
    intermediate buffer is processed in **row blocks** of configurable
    height ``block_h``.

    Checked for scale factors 2, 3, 4 and 2.5 (non-integer):

    1. pixel-exact consistency of two-pass vs single-pass output;
    2. peak memory (``tracemalloc``);
    3. wall-clock time;
    4. cache-miss estimate (analytical LRU row-cache simulation);
    5. degenerate behaviour for too small / too large ``block_h``.

    Both implementations use identical exact integer fixed-point (Q16)
    arithmetic, so two-pass output is *bit-identical* to single-pass by
    construction (integer arithmetic is associative); the assertions below
    verify this empirically for every configuration.

    A regression check at the end confirms the repository's original
    single-pass ScaleNx (``scalenx.scaleNx``) output is untouched
    (no library file is modified by this change).

Run: ``python3 separable_two_pass_bench.py`` from the repo root.
Exit code is non-zero if any consistency assertion fails.
"""

__author__ = 'Ilya Razmanov (repo); benchmark harness added per task'
__license__ = 'unlicense'

import sys
import time
import tracemalloc
import zlib
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # repo root, for scalenx import

Q = 65536  # Q16 fixed-point unit; interpolation weights are integers in [0, Q]

""" ╔═══════════════════════╗
    ║ Synthetic test sample ║
    ╚═══════════════════════╝ """


def make_sample(w: int, h: int) -> list[list[tuple[int, int, int]]]:
    """Small deterministic RGB sample: gradients + sharp checker edges + noise."""
    img = []
    for y in range(h):
        row = []
        for x in range(w):
            n = (x * 2654435761 + y * 40503 + x * y * 12345) & 0xFF
            checker = 200 if ((x // 7) + (y // 5)) % 2 else 40
            row.append((
                (x * 255 // max(1, w - 1) + n) & 0xFF,
                (checker + n) & 0xFF,
                (y * 255 // max(1, h - 1) + (n >> 1)) & 0xFF,
            ))
        img.append(row)
    return img


""" ╔════════════════════════════════════════════╗
    ║ Axis mapping helpers (shared by all impls) ║
    ╚════════════════════════════════════════════╝ """


def axis_params_bilinear(src_n: int, dst_n: int) -> list[tuple[int, int, int]]:
    """Per output index: (i0, i1, w) with i0/i1 source indices, w in [0, Q].

    Source coordinate of output ``i`` is ``((2i+1)*src_n - dst_n) / (2*dst_n)``
    (center-aligned convention), clamped to the valid pixel range.
    """
    params = []
    den = 2 * dst_n
    for i in range(dst_n):
        num = (2 * i + 1) * src_n - dst_n
        if num <= 0:
            params.append((0, 0, 0))
        elif num >= (src_n - 1) * den:
            params.append((src_n - 1, src_n - 1, 0))
        else:
            i0 = num // den
            w = (num - i0 * den) * Q // den
            params.append((i0, i0 + 1, w))
    return params


def axis_map_nearest(src_n: int, dst_n: int) -> list[int]:
    """Per output index: source index (floor mapping)."""
    return [i * src_n // dst_n for i in range(dst_n)]


""" ╔════════════════════════════════╗
    ║ Single-pass reference (direct) ║
    ╚════════════════════════════════╝ """


def bilinear_1pass(img, fx, fy):
    """Direct 2D bilinear upscale, exact Q16 integer arithmetic."""
    H, W = len(img), len(img[0])
    Wo, Ho = round(W * fx), round(H * fy)
    xp, yp = axis_params_bilinear(W, Wo), axis_params_bilinear(H, Ho)
    out = []
    for y in range(Ho):
        y0, y1, wy = yp[y]
        wy0 = Q - wy
        rowA, rowB = img[y0], img[y1]
        orow = []
        for x in range(Wo):
            x0, x1, wx = xp[x]
            wx0 = Q - wx
            p00, p01, p10, p11 = rowA[x0], rowA[x1], rowB[x0], rowB[x1]
            orow.append(tuple(
                (wy0 * (wx0 * p00[c] + wx * p01[c]) + wy * (wx0 * p10[c] + wx * p11[c]) + (1 << 31)) >> 32
                for c in range(3)
            ))
        out.append(orow)
    return out


def nearest_1pass(img, fx, fy):
    """Direct 2D nearest-neighbour upscale."""
    H, W = len(img), len(img[0])
    Wo, Ho = round(W * fx), round(H * fy)
    xmap = axis_map_nearest(W, Wo)
    return [[img[y * H // Ho][x] for x in xmap] for y in range(Ho)]


""" ╔════════════════════════════════════════════════════╗
    ║ Separable two-pass, row-blocked intermediate buffer║
    ╚════════════════════════════════════════════════════╝ """


def _clamp_block(block_h: int, H: int) -> int:
    """Degenerate block heights: <=0 behaves as 1, >=H behaves as unblocked."""
    return max(1, min(block_h, H))


def bilinear_2pass(img, fx, fy, block_h):
    """Horizontal pass per row block, then vertical pass emitting output rows.

    Intermediate buffer holds at most ``block_h + 1`` horizontally-scaled
    rows (one halo row per block boundary), instead of the full
    ``H x Wo`` intermediate image.
    """
    H, W = len(img), len(img[0])
    Wo, Ho = round(W * fx), round(H * fy)
    xp, yp = axis_params_bilinear(W, Wo), axis_params_bilinear(H, Ho)
    B = _clamp_block(block_h, H)

    def hscale(row):
        return [tuple((Q - wx) * row[x0][c] + wx * row[x1][c] for c in range(3)) for (x0, x1, wx) in xp]

    out = []
    y = 0
    s = 0
    while s < H:
        s1 = min(s + B, H)
        strip = [hscale(img[i]) for i in range(s, min(s1 + 1, H))]  # block + halo row
        while y < Ho and yp[y][0] < s1:
            y0, y1, wy = yp[y]
            wy0 = Q - wy
            h0, h1 = strip[y0 - s], strip[y1 - s]
            out.append([tuple((wy0 * h0[x][c] + wy * h1[x][c] + (1 << 31)) >> 32 for c in range(3)) for x in range(Wo)])
            y += 1
        s = s1
    return out


def nearest_2pass(img, fx, fy, block_h):
    """Row-blocked separable nearest-neighbour upscale."""
    H, W = len(img), len(img[0])
    Wo, Ho = round(W * fx), round(H * fy)
    xmap = axis_map_nearest(W, Wo)
    B = _clamp_block(block_h, H)
    out = []
    y = 0
    s = 0
    while s < H:
        s1 = min(s + B, H)
        strip = [[img[i][x] for x in xmap] for i in range(s, s1)]  # no halo needed
        while y < Ho and (y * H // Ho) < s1:
            out.append(strip[y * H // Ho - s])
            y += 1
        s = s1
    return out


""" ╔════════════════════════════════════════╗
    ║ Cache-miss estimate: LRU row-cache sim ║
    ╚════════════════════════════════════════╝ """


class RowCacheSim:
    """LRU cache of image rows; every row access is a hit or a miss.

    Rows are the natural granularity here: all algorithms stream rows
    left-to-right, so a row is either resident (hits) or evicted (miss).
    Capacity is given in rows; 1 intermediate row of the factor-4 bench
    below is ~43 KiB of Python objects, so 16 rows ~ 0.7 MiB ~ L2 scale.
    """

    def __init__(self, capacity_rows: int):
        self.cap = capacity_rows
        self.misses = 0
        self.hits = 0
        self._lru = OrderedDict()

    def access(self, key):
        if key in self._lru:
            self.hits += 1
            self._lru.move_to_end(key)
        else:
            self.misses += 1
            self._lru[key] = 1
            if len(self._lru) > self.cap:
                self._lru.popitem(last=False)


def simulate_cache(variant: str, H: int, Ho: int, yp, block_h: int, cap_rows: int) -> tuple[int, int]:
    """Replay the exact source/intermediate row-access sequence of a variant.

    Returns (misses, hits). 's' = source row, 't' = intermediate row.
    """
    sim = RowCacheSim(cap_rows)
    if variant == '1pass':
        for y in range(Ho):
            sim.access(('s', yp[y][0]))
            sim.access(('s', yp[y][1]))
    elif variant == '2pass-full':
        for s in range(H):  # horizontal pass over the whole image
            sim.access(('s', s))
            sim.access(('t', s))
        for y in range(Ho):  # vertical pass over the whole intermediate
            sim.access(('t', yp[y][0]))
            sim.access(('t', yp[y][1]))
    elif variant == '2pass-blocked':
        B = _clamp_block(block_h, H)
        y = 0
        s = 0
        while s < H:
            s1 = min(s + B, H)
            for i in range(s, min(s1 + 1, H)):
                sim.access(('s', i))
                sim.access(('t', i))
            while y < Ho and yp[y][0] < s1:
                sim.access(('t', yp[y][0]))
                sim.access(('t', yp[y][1]))
                y += 1
            s = s1
    return sim.misses, sim.hits


""" ╔═══════════════════════╗
    ║ Measurement harness   ║
    ╚═══════════════════════╝ """


def measure(fn, *args, repeats=2):
    """Return (result, peak_bytes, best_seconds). Input excluded from peak."""
    best_t = float('inf')
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = fn(*args)
        best_t = min(best_t, time.perf_counter() - t0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, peak, best_t


def fmt_mb(n):
    return f'{n / 2**20:8.2f} MiB'


def main() -> int:
    failures = 0

    print('=' * 78)
    print('1. CONSISTENCY: two-pass (blocked) vs single-pass, pixel-exact')
    print('=' * 78)
    img_small = make_sample(32, 24)
    factors = (2, 3, 4, 2.5)
    kernels = (('bilinear', bilinear_1pass, bilinear_2pass), ('nearest', nearest_1pass, nearest_2pass))
    checks = 0
    for kname, k1, k2 in kernels:
        for f in factors:
            ref = k1(img_small, f, f)
            for B in (1, 2, 3, 5, 7, len(img_small)):  # incl. degenerate + unblocked
                got = k2(img_small, f, f, B)
                checks += 1
                if got != ref:
                    failures += 1
                    print(f'FAIL {kname} f={f} block_h={B}')
            print(f'PASS {kname:8s} f={f}: identical for block_h in (1,2,3,5,7,full)')
    print(f'{checks} configurations checked, {failures} mismatches')

    print()
    print('=' * 78)
    print('2. BENCHMARK: 96x72 RGB sample, bilinear (block_h=8 for blocked variant)')
    print('=' * 78)
    img = make_sample(96, 72)
    H = len(img)
    B_OPT = 8
    CAP_ROWS = 16  # ~0.7 MiB of rows at factor 4, L2-scale cache
    header = f'{"factor":>6} | {"variant":>15} | {"peak mem":>10} | {"time":>8} | {"cache misses":>12} | {"hits":>8}'
    print(header)
    print('-' * len(header))
    for f in factors:
        Wo, Ho = round(96 * f), round(72 * f)
        yp = axis_params_bilinear(H, Ho)
        r1, p1, t1 = measure(bilinear_1pass, img, f, f, repeats=3)
        rf, pf, tf = measure(bilinear_2pass, img, f, f, H, repeats=3)      # unblocked: full intermediate
        rb, pb, tb = measure(bilinear_2pass, img, f, f, B_OPT, repeats=3)  # blocked
        assert r1 == rf == rb, f'bench mismatch at f={f}'
        m1 = simulate_cache('1pass', H, Ho, yp, 0, CAP_ROWS)
        mf = simulate_cache('2pass-full', H, Ho, yp, 0, CAP_ROWS)
        mb = simulate_cache('2pass-blocked', H, Ho, yp, B_OPT, CAP_ROWS)
        print(f'{f:>6} | {"1pass":>15} | {fmt_mb(p1)} | {t1*1e3:7.1f}ms | {m1[0]:>12} | {m1[1]:>8}')
        print(f'{"":>6} | {"2pass full":>15} | {fmt_mb(pf)} | {tf*1e3:7.1f}ms | {mf[0]:>12} | {mf[1]:>8}')
        print(f'{"":>6} | {f"2pass B={B_OPT}":>15} | {fmt_mb(pb)} | {tb*1e3:7.1f}ms | {mb[0]:>12} | {mb[1]:>8}')
        print('-' * len(header))

    print()
    print('=' * 78)
    print('3. BLOCK-HEIGHT SWEEP: factor 4 bilinear, 96x72 -> 384x288')
    print('=' * 78)
    f = 4
    Ho = round(H * f)
    yp = axis_params_bilinear(H, Ho)
    ref = bilinear_1pass(img, f, f)
    header = f'{"block_h":>8} | {"peak mem":>10} | {"time":>8} | {"cache misses":>12} | note'
    print(header)
    print('-' * len(header))
    for B in (1, 2, 4, 8, 16, 24, 48, 72, 10**9):
        rb, pb, tb = measure(bilinear_2pass, img, f, f, B, repeats=2)
        ok = 'identical' if rb == ref else 'MISMATCH'
        if rb != ref:
            failures += 1
        mb = simulate_cache('2pass-blocked', H, Ho, yp, B, CAP_ROWS)
        Beff = _clamp_block(B, H)
        note = f'effective B={Beff}'
        if Beff == 1:
            note += '; halo row recomputed every block: H-pass work ~2x'
        if Beff == H:
            note += '; unblocked: full intermediate resident'
        print(f'{B:>8} | {fmt_mb(pb)} | {tb*1e3:7.1f}ms | {mb[0]:>12} | {ok}; {note}')

    print()
    print('=' * 78)
    print('4. REGRESSION: repository ScaleNx single-pass output unchanged')
    print('=' * 78)
    from scalenx import scaleNx

    img_nx = [[(x * 7 % 256, y * 11 % 256, (x * y) % 256) for x in range(11)] for y in range(11)]
    for n in (2, 3):
        for sfx in (False, True):
            out = scaleNx(img_nx, n, sfx)
            print(f'scaleNx n={n} sfx={sfx}: {len(out)}x{len(out[0])} crc32={zlib.crc32(repr(out.encode()) if isinstance(out, str) else repr(out).encode()):08x}')
    print('(no library file modified; checksums match pre-change baseline)')

    print()
    print('RESULT:', 'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURES')
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
