#!/usr/bin/env python3

"""
=======
seamfix
=======

--------------------------------------------------------------
Seam detection and correction for seamless (tileable) textures
--------------------------------------------------------------

:Abstract: When a seamless texture is upscaled, tile seams may become
    visible. Current module implements a three-stage seam pipeline
    working on the same 3D nested list image format as ``scalenx``:

    1. **Detection** — seam positions are located from per-row and
       per-column mean absolute gradients: any row/column boundary
       whose gradient exceeds ``threshold`` times the image mean
       gradient is considered a seam.
    2. **Transition** — within a configurable band around every seam,
       each pixel is blended with its mirror reflection across the
       seam axis, with the blend weight peaking at the seam and
       fading to zero at the band edge.
    3. **Wrap check** — continuity across the four tile edges
       (left/right and top/bottom wrap-around) is verified; can be
       switched off.

    Detection threshold, transition width and wrap check are
    configurable via :func:`fix_seams` parameters.

    The module never modifies the source image in place and never
    touches ``scalenx`` scaling functions — original scaling output
    stays pixel-identical (verified by a golden hash regression
    in the self-test below).

Usage
-----

::

    fixed_image, report = fix_seams(scaled_image,
                                    threshold=3.0,
                                    width=3,
                                    wrap_check=True)

Running the file as a script generates four small synthetic sample
sets (seamless, vertical seam, horizontal seam, both), scales them
with Scale2x, runs the pipeline on each, and prints a report with
seam count, gradient difference after transition, wrap consistency
and elapsed time per set. Samples with no seams are reported
explicitly.

----
"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2026 Ilya Razmanov'
__license__ = 'unlicense'
__version__ = '2026.11.18'
__status__ = 'Production'

from time import perf_counter

""" ╔════════════════════════════════════════════╗
    ║           Stage 1: seam detection          ║
    ╚════════════════════════════════════════════╝ """


def _axis_gradients(image3d: list[list[list[int]]], axis: str) -> list[float]:
    """Mean absolute gradient across each column (axis='v') or row
    (axis='h') boundary. Returned list has length X-1 or Y-1;
    element ``i`` is the gradient between columns/rows ``i`` and ``i+1``."""

    Y, X = len(image3d), len(image3d[0])
    grads: list[float] = []

    if axis == 'v':
        for x in range(X - 1):
            total = 0.0
            for y in range(Y):
                left, right = image3d[y][x], image3d[y][x + 1]
                total += sum(abs(right[c] - left[c]) for c in range(len(left)))
            grads.append(total / Y)
    else:
        for y in range(Y - 1):
            total = 0.0
            for x in range(X):
                top, bottom = image3d[y][x], image3d[y + 1][x]
                total += sum(abs(bottom[c] - top[c]) for c in range(len(top)))
            grads.append(total / X)

    return grads


def _find_seams(grads: list[float], threshold: float) -> list[tuple[int, int]]:
    """Group gradient boundaries exceeding ``threshold`` times the mean
    gradient into seam bands. Returns list of (start, end) boundary
    index pairs, inclusive."""

    if not grads:
        return []
    mean = sum(grads) / len(grads)
    if mean <= 0.0:
        return []
    limit = threshold * mean

    seams: list[tuple[int, int]] = []
    start = None
    for i, g in enumerate(grads):
        if g > limit:
            if start is None:
                start = i
        elif start is not None:
            seams.append((start, i - 1))
            start = None
    if start is not None:
        seams.append((start, len(grads) - 1))
    return seams


""" ╔════════════════════════════════════════════╗
    ║     Stage 2: mirror blend at seam band     ║
    ╚════════════════════════════════════════════╝ """


def _blend_seam(image3d: list[list[list[int]]], seam: tuple[int, int], axis: str, width: int) -> list[list[list[int]]]:
    """Blend a band of ``width`` pixels to each side of the seam with
    its mirror reflection across the seam axis. Blend weight peaks at
    the seam center and fades linearly to zero at the band edge."""

    Y, X = len(image3d), len(image3d[0])
    result = [[list(pixel) for pixel in row] for row in image3d]

    # ↓ seam axis sits between boundaries seam[0]..seam[1] and their +1 neighbours
    lo = seam[0]  # first boundary index of the band
    hi = seam[1] + 1  # last column/row after the band
    center = (lo + hi) / 2.0

    size = X if axis == 'v' else Y
    for i in range(max(0, lo - width), min(size, hi + width + 1)):
        mirror = int(2 * center - i)
        # ↓ reflect mirror index back into the valid range
        mirror = max(0, min(size - 1, mirror))
        distance = abs(i - center)
        alpha = 0.5 * max(0.0, 1.0 - distance / (width + 0.5))
        if alpha <= 0.0 or mirror == i:
            continue
        if axis == 'v':
            for y in range(Y):
                src, ref = result[y][i], image3d[y][mirror]
                for c in range(len(src)):
                    src[c] = round((1.0 - alpha) * src[c] + alpha * ref[c])
        else:
            for x in range(X):
                src, ref = result[i][x], image3d[mirror][x]
                for c in range(len(src)):
                    src[c] = round((1.0 - alpha) * src[c] + alpha * ref[c])

    return result


""" ╔════════════════════════════════════════════╗
    ║   Stage 3: wrap-around continuity check    ║
    ╚════════════════════════════════════════════╝ """


def wrap_metrics(image3d: list[list[list[int]]]) -> dict[str, float]:
    """Wrap-around continuity across the four tile edges:
    mean absolute gradient between last and first column
    (horizontal wrap) and between last and first row (vertical wrap)."""

    Y, X = len(image3d), len(image3d[0])
    channels = len(image3d[0][0])

    hor = sum(
        abs(image3d[y][X - 1][c] - image3d[y][0][c])
        for y in range(Y)
        for c in range(channels)
    ) / Y
    ver = sum(
        abs(image3d[Y - 1][x][c] - image3d[0][x][c])
        for x in range(X)
        for c in range(channels)
    ) / X

    return {'horizontal_wrap': hor, 'vertical_wrap': ver}


""" ╔════════════════════════════════════════════╗
    ║            Full three-stage pipe           ║
    ╚════════════════════════════════════════════╝ """


def fix_seams(
    image3d: list[list[list[int]]],
    threshold: float = 3.0,
    width: int = 3,
    wrap_check: bool = True,
) -> tuple[list[list[list[int]]], dict]:
    """Detect seams by row/column gradients, smooth them with a mirror
    blend transition, and verify wrap-around continuity.

    :param image3d: source image as 3D nested list (not modified);
    :param threshold: detection threshold, multiplier over the image
        mean gradient (higher = fewer seams detected);
    :param width: transition half-width in pixels to each side
        of a seam;
    :param wrap_check: whether to verify four-edge wrap-around
        continuity;
    :return: ``(fixed_image, report)`` where report holds seam count,
        seam gradient before/after transition, wrap consistency and
        elapsed seconds.

    """

    started = perf_counter()

    v_grads = _axis_gradients(image3d, 'v')
    h_grads = _axis_gradients(image3d, 'h')
    v_seams = _find_seams(v_grads, threshold)
    h_seams = _find_seams(h_grads, threshold)

    def _seam_level(grads: list[float], seams: list[tuple[int, int]]) -> float:
        """Mean gradient at detected seam boundaries."""
        values = [g for seam in seams for g in grads[seam[0]: seam[1] + 1]]
        return sum(values) / len(values) if values else 0.0

    before = _seam_level(v_grads, v_seams) + _seam_level(h_grads, h_seams)

    fixed = image3d
    for seam in v_seams:
        fixed = _blend_seam(fixed, seam, 'v', width)
    for seam in h_seams:
        fixed = _blend_seam(fixed, seam, 'h', width)

    if v_seams or h_seams:
        after = _seam_level(_axis_gradients(fixed, 'v'), v_seams) + _seam_level(_axis_gradients(fixed, 'h'), h_seams)
    else:
        after = 0.0

    report = {
        'vertical_seams': len(v_seams),
        'horizontal_seams': len(h_seams),
        'seam_gradient_before': before,
        'seam_gradient_after': after,
        'wrap_check_enabled': wrap_check,
        'elapsed': perf_counter() - started,
    }
    if wrap_check:
        metrics = wrap_metrics(fixed)
        mean_grad = (sum(v_grads) / max(1, len(v_grads)) + sum(h_grads) / max(1, len(h_grads))) / 2.0
        limit = threshold * mean_grad
        report.update(metrics)
        report['wrap_consistent'] = metrics['horizontal_wrap'] <= limit and metrics['vertical_wrap'] <= limit

    return fixed, report


# ↑ three-stage pipe finished


""" ╔════════════════════════════════════════════╗
    ║  Self-test: original scaling output must   ║
    ║  stay pixel-identical (golden hashes of    ║
    ║  pristine scalenx/scalenxsfx functions)    ║
    ╚════════════════════════════════════════════╝ """

_GOLDEN = {
    'scalenx.scale2x': '1776dfac929383ce',
    'scalenx.scale3x': '2763588d8ae3d347',
    'scalenxsfx.scale2x': '1776dfac929383ce',
    'scalenxsfx.scale3x': '2763588d8ae3d347',
}


def _selftest_scaling_unchanged() -> bool:
    """Hash scale2x/scale3x (plain and SFX) output on a fixed pattern
    and compare against golden hashes of the pristine library."""

    import hashlib
    import random
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent / 'scalenx'))
    import scalenx
    import scalenxsfx

    random.seed(20261118)
    img = [[[random.randrange(256) for _ in range(3)] for _ in range(8)] for _ in range(8)]

    def digest(im: list[list[list[int]]]) -> str:
        h = hashlib.sha256()
        for row in im:
            for px in row:
                h.update(bytes(px))
        return h.hexdigest()[:16]

    ok = True
    for name, fn in [
        ('scalenx.scale2x', scalenx.scale2x),
        ('scalenx.scale3x', scalenx.scale3x),
        ('scalenxsfx.scale2x', scalenxsfx.scale2x),
        ('scalenxsfx.scale3x', scalenxsfx.scale3x),
    ]:
        match = digest(fn(img)) == _GOLDEN[name]
        ok = ok and match
        print(f'  {"PASS" if match else "FAIL"}  {name} 输出逐像素不变')
    return ok


""" ╔════════════════════════════════════════════╗
    ║  Demo: four synthetic sample sets, one run ║
    ╚════════════════════════════════════════════╝ """


def _make_samples(size: int = 24) -> dict[str, list[list[list[int]]]]:
    """Four small deterministic RGB textures: seamless, vertical seam,
    horizontal seam, both seams."""

    import math
    import random

    random.seed(7)
    noise = [[random.randrange(60) for _ in range(size)] for _ in range(size)]

    def seamless(x: int, y: int) -> list[int]:
        # ↓ periodic in both directions: genuinely tileable, no seams
        r = 128 + int(100 * math.sin(2 * math.pi * 3 * x / size) * math.cos(2 * math.pi * 2 * y / size))
        g = 128 + int(80 * math.sin(2 * math.pi * (x + y) / size))
        b = 128 + int(60 * math.cos(2 * math.pi * 2 * x / size))
        return [r, g, b]

    def v_seam(x: int, y: int) -> list[int]:
        base = 210 if x >= size // 2 else 60  # hard vertical break
        v = base + noise[y][x]
        return [v, v, (v * 3) % 256]

    def h_seam(x: int, y: int) -> list[int]:
        base = 210 if y >= size // 2 else 60  # hard horizontal break
        v = base + noise[y][x]
        return [v, (v * 3) % 256, v]

    def both(x: int, y: int) -> list[int]:
        base = 200 if (x >= size // 2) == (y >= size // 2) else 50
        v = base + noise[y][x]
        return [v, v, v]

    makers = {
        'A 无缝贴图 (seamless)': seamless,
        'B 垂直接缝 (vertical)': v_seam,
        'C 水平接缝 (horizontal)': h_seam,
        'D 双向接缝 (both)': both,
    }
    return {name: [[maker(x, y) for x in range(size)] for y in range(size)] for name, maker in makers.items()}


def _demo() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent / 'scalenx'))
    import scalenx

    # ↓ configurable pipeline parameters
    threshold = 3.0  # detection threshold, × mean gradient
    width = 3  # transition half-width, px
    wrap_check = True  # four-edge wrap-around verification

    print('原缩放输出回归校验 (golden hash):')
    if not _selftest_scaling_unchanged():
        print('缩放输出发生变化，终止。')
        raise SystemExit(1)

    print(f'\n接缝处理参数: 检测阈值={threshold}×平均梯度, 过渡宽度=±{width}px, 环绕校验={"开" if wrap_check else "关"}')
    print('流程: 24×24 样例 → Scale2x 放大 → 接缝检测/过渡/环绕校验\n')

    for name, image in _make_samples().items():
        scaled = scalenx.scale2x(image)  # seams become visible after upscale
        fixed, rep = fix_seams(scaled, threshold=threshold, width=width, wrap_check=wrap_check)

        total = rep['vertical_seams'] + rep['horizontal_seams']
        print(f'■ {name}  (放大后 {len(scaled[0])}×{len(scaled)})')
        if total == 0:
            print('    未检测到接缝 (no seams detected) — 已跳过过渡处理')
        else:
            print(f'    接缝数量: 垂直 {rep["vertical_seams"]} + 水平 {rep["horizontal_seams"]} = {total}')
            print(f'    接缝带梯度: 过渡前 {rep["seam_gradient_before"]:.1f} → 过渡后 {rep["seam_gradient_after"]:.1f}'
                  f'  (降低 {rep["seam_gradient_before"] - rep["seam_gradient_after"]:.1f})')
        if rep['wrap_check_enabled']:
            verdict = '连续' if rep['wrap_consistent'] else '不连续'
            print(f'    环绕一致性: 水平 {rep["horizontal_wrap"]:.1f}, 垂直 {rep["vertical_wrap"]:.1f} → {verdict}')
        print(f'    耗时: {rep["elapsed"] * 1000:.2f} ms\n')


# ↓ Dummy stub for standalone execution
if __name__ == '__main__':
    _demo()
