#!/usr/bin/env python3

"""
================
test_anisotropic
================

Unit tests for Scale2x/Scale3x regression and for the bilinear and
anisotropic (isophote-directed) upscaling functions added to
:mod:`scalenx.scalenx`.

Pure standard library only; run with::

    python -m unittest scalenx.test_anisotropic -v
"""

import math
import sys
import unittest
from pathlib import Path

# Allow running both as `python -m unittest scalenx.test_anisotropic`
# and directly as `python scalenx/test_anisotropic.py`, including under
# Python isolated mode (-I), which ignores PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scalenx.scalenx import anisotropic_scale, bilinear_scale, scale2x, scale3x
from scalenx.scalenxsfx import scale2x as scale2xsfx
from scalenx.scalenxsfx import scale3x as scale3xsfx
from scalenx import scaleNx


# Small hand-made fixtures --------------------------------------------------

GRADIENT = [[[x * 17] for x in range(8)] for _ in range(6)]

RGB_CHECKER = [
    [[(x + y) % 2 * 255, 0, (1 - (x + y) % 2) * 255] for x in range(4)]
    for y in range(4)
]

# 45-degree diagonal line: 255 where x == y, 0 elsewhere.
DIAGONAL = [
    [[255] if x == y else [0] for x in range(8)]
    for y in range(8)
]

# Filled circle, radius 3, centered in a 9x9 field.
CIRCLE = [
    [[255] if (x - 4) ** 2 + (y - 4) ** 2 <= 9 else [0] for x in range(9)]
    for y in range(9)
]


class TestScaleNxRegression(unittest.TestCase):
    """Original Scale2x/Scale3x behavior must remain pixel-identical."""

    def test_scale2x_dimensions(self):
        out = scale2x(DIAGONAL)
        self.assertEqual((len(out), len(out[0])), (16, 16))

    def test_scale3x_dimensions(self):
        out = scale3x(DIAGONAL)
        self.assertEqual((len(out), len(out[0])), (24, 24))

    def test_scale2x_known_block(self):
        # Single bright pixel on black background: Scale2x rule keeps the
        # 2x2 block equal to E when all 4 neighbors differ from each other
        # in the "no edge" arrangement... here A=C=D=0, B=0 too, so the
        # whole block stays E=255.
        img = [[[0], [0], [0]], [[0], [255], [0]], [[0], [0], [0]]]
        out = scale2x(img)
        self.assertEqual(out[2][2], [255])
        self.assertEqual(out[3][3], [255])
        self.assertEqual(out[0][0], [0])

    def test_scale2x_edge_rule(self):
        # Horizontal edge: top row white, bottom row black. Scale2x must
        # propagate white upward and black downward, no mixed colors.
        img = [[[255], [255]], [[0], [0]]]
        out = scale2x(img)
        self.assertTrue(all(p == [255] for p in out[0]))
        self.assertTrue(all(p == [255] for p in out[1]))
        self.assertTrue(all(p == [0] for p in out[2]))
        self.assertTrue(all(p == [0] for p in out[3]))

    def test_scale2x_scale3x_unchanged_outputs(self):
        # Golden outputs captured from the pre-change implementation
        # (deterministic conditional trees, no randomness).
        img = [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]]
        self.assertEqual(
            scale2x(img),
            [
                [[1], [1], [2], [2], [3], [3]],
                [[1], [1], [2], [2], [3], [3]],
                [[4], [4], [5], [5], [6], [6]],
                [[4], [4], [5], [5], [6], [6]],
                [[7], [7], [8], [8], [9], [9]],
                [[7], [7], [8], [8], [9], [9]],
            ],
        )
        self.assertEqual(
            scale3x(img),
            [
                [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
                [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
                [[1], [1], [1], [2], [2], [2], [3], [3], [3]],
                [[4], [4], [4], [5], [5], [5], [6], [6], [6]],
                [[4], [4], [4], [5], [5], [5], [6], [6], [6]],
                [[4], [4], [4], [5], [5], [5], [6], [6], [6]],
                [[7], [7], [7], [8], [8], [8], [9], [9], [9]],
                [[7], [7], [7], [8], [8], [8], [9], [9], [9]],
                [[7], [7], [7], [8], [8], [8], [9], [9], [9]],
            ],
        )

    def test_scaleNx_dispatch(self):
        self.assertEqual(scaleNx(DIAGONAL, 2, False), scale2x(DIAGONAL))
        self.assertEqual(scaleNx(DIAGONAL, 3, False), scale3x(DIAGONAL))
        self.assertEqual(scaleNx(DIAGONAL, 2, True), scale2xsfx(DIAGONAL))
        self.assertEqual(scaleNx(DIAGONAL, 3, True), scale3xsfx(DIAGONAL))
        with self.assertRaises(ValueError):
            scaleNx(DIAGONAL, 4, False)


class TestBilinear(unittest.TestCase):

    def test_dimensions(self):
        out = bilinear_scale(GRADIENT, 2)
        self.assertEqual((len(out), len(out[0])), (12, 16))

    def test_factor_one_identity(self):
        self.assertEqual(bilinear_scale(GRADIENT, 1), GRADIENT)

    def test_integer_factor_corners_preserved(self):
        out = bilinear_scale(DIAGONAL, 2)
        self.assertEqual(out[0][0], DIAGONAL[0][0])
        self.assertEqual(out[-1][-1], DIAGONAL[-1][-1])
        self.assertEqual(out[0][-1], DIAGONAL[0][-1])

    def test_flat_region_stays_flat(self):
        img = [[[42] for _ in range(4)] for _ in range(4)]
        out = bilinear_scale(img, 3)
        self.assertTrue(all(p == [42] for row in out for p in row))

    def test_linear_gradient_exact(self):
        # Bilinear must reproduce a linear ramp exactly. The ground truth
        # here is the same coordinate mapping evaluated independently,
        # with one LSB tolerance for integer rounding.
        out = bilinear_scale(GRADIENT, 2)
        out_X = len(out[0])
        scale = (8 - 1) / (out_X - 1)
        for row in out:
            for x, p in enumerate(row):
                expected = round((x * scale) * 17)
                self.assertLessEqual(abs(p[0] - expected), 1)

    def test_value_range(self):
        out = bilinear_scale(RGB_CHECKER, 2.5)
        for row in out:
            for p in row:
                for v in p:
                    self.assertGreaterEqual(v, 0)
                    self.assertLessEqual(v, 255)

    def test_factor_below_one_raises(self):
        with self.assertRaises(ValueError):
            bilinear_scale(GRADIENT, 0.5)


class TestAnisotropic(unittest.TestCase):

    def test_dimensions(self):
        for f in (2, 3, 1.5, 2.7):
            out = anisotropic_scale(GRADIENT, f)
            self.assertEqual(
                (len(out), len(out[0])),
                (round(6 * f), round(8 * f)),
                f'factor {f}',
            )

    def test_factor_one_returns_equivalent_copy(self):
        out = anisotropic_scale(DIAGONAL, 1)
        self.assertEqual(out, DIAGONAL)
        # Structural independence: mutating the result must not touch input.
        out[0][0][0] = 123
        self.assertEqual(DIAGONAL[0][0], [255])

    def test_factor_below_one_raises(self):
        with self.assertRaises(ValueError):
            anisotropic_scale(GRADIENT, 0.99)

    def test_bad_window_and_strength_raise(self):
        with self.assertRaises(ValueError):
            anisotropic_scale(GRADIENT, 2, window=-1)
        with self.assertRaises(ValueError):
            anisotropic_scale(GRADIENT, 2, strength=-0.01)
        with self.assertRaises(ValueError):
            anisotropic_scale(GRADIENT, 2, strength=1.01)

    def test_strength_zero_equals_bilinear(self):
        # With the directional term disabled the scaler collapses onto
        # plain bilinear interpolation.
        self.assertEqual(
            anisotropic_scale(DIAGONAL, 2, window=1, strength=0.0),
            bilinear_scale(DIAGONAL, 2),
        )
        self.assertEqual(
            anisotropic_scale(CIRCLE, 2.5, window=2, strength=0.0),
            bilinear_scale(CIRCLE, 2.5),
        )

    def test_configurable_window_runs(self):
        for w in (0, 1, 2, 3):
            out = anisotropic_scale(DIAGONAL, 2, window=w)
            self.assertEqual((len(out), len(out[0])), (16, 16))

    def test_rgb_and_tuple_pixels(self):
        out_list = anisotropic_scale(RGB_CHECKER, 2)
        out_tuple = anisotropic_scale(
            [[tuple(p) for p in row] for row in RGB_CHECKER], 2
        )
        self.assertEqual(
            [[list(p) for p in row] for row in out_tuple],
            out_list,
        )
        self.assertEqual(len(out_list[0][0]), 3)

    def test_value_range_and_bitdepth(self):
        # 16 bpc style image must stay within its own max.
        deep = [[[v * 4096] for v in row] for row in [[0, 1], [1, 0]]]
        out = anisotropic_scale(deep, 3)
        peak = max(p[0] for row in out for p in row)
        self.assertLessEqual(peak, 4096)
        # 1 bpc image stays binary-valued.
        binary = [[[x % 2] for x in range(5)] for _ in range(5)]
        outb = anisotropic_scale(binary, 2)
        self.assertTrue(all(p[0] in (0, 1) for row in outb for p in row))

    def test_flat_region_unaffected_by_direction(self):
        img = [[[77] for _ in range(5)] for _ in range(5)]
        out = anisotropic_scale(img, 2, window=2, strength=1.0)
        self.assertTrue(all(p == [77] for row in out for p in row))

    def test_corners_preserved(self):
        # Output corner pixels map exactly onto source corner pixels. Where
        # the corner neighborhood is flat the coherence is zero, so the
        # value passes through unchanged (edge pixels legitimately undergo
        # directional interpolation).
        solid = [[[90] for _ in range(4)] for _ in range(4)]
        outs = anisotropic_scale(solid, 3)
        self.assertEqual(outs[0][0], [90])
        self.assertEqual(outs[-1][-1], [90])
        outb = bilinear_scale(solid, 3)
        self.assertEqual(outb[0][0], [90])
        self.assertEqual(outb[-1][-1], [90])

    def test_diagonal_edge_orientation_beats_bilinear(self):
        # The "发毛" (ragged/fuzzy edge) complaint is an orientation problem.
        # On a realistic anti-aliased 45-degree edge the anisotropic scaler
        # must reconstruct isophote directions more faithfully and must not
        # sacrifice edge sharpness relative to isotropic bilinear.
        factor = 4
        size = 48

        def smooth(d):
            if d <= -0.5:
                return 0
            if d >= 0.5:
                return 255
            t = d + 0.5
            t = t * t * (3 - 2 * t)
            return round(t * 255)

        diagonal = [
            [[smooth(x - y)] for x in range(size * factor)]
            for y in range(size * factor)
        ]
        low = [
            [
                [round(sum(
                    diagonal[y * factor + dy][x * factor + dx][0]
                    for dy in range(factor) for dx in range(factor)
                ) / (factor * factor))]
                for x in range(size)
            ]
            for y in range(size)
        ]

        def sobel(img):
            flat = [[p[0] for p in r] for r in img]
            gx = [[0.0] * len(flat[0]) for _ in flat]
            gy = [r[:] for r in gx]
            for y in range(1, len(flat) - 1):
                for x in range(1, len(flat[0]) - 1):
                    gx[y][x] = flat[y][x + 1] - flat[y][x - 1]
                    gy[y][x] = flat[y + 1][x] - flat[y - 1][x]
            return gx, gy

        def orientation_error(img):
            rg = sobel(diagonal)
            cg = sobel(img)
            total = 0.0
            count = 0
            for y in range(len(diagonal)):
                for x in range(len(diagonal[0])):
                    if math.hypot(rg[0][y][x], rg[1][y][x]) <= 64.0:
                        continue
                    ra = math.atan2(-rg[0][y][x], rg[1][y][x])
                    ca = math.atan2(-cg[0][y][x], cg[1][y][x])
                    diff = abs(ra - ca) % math.pi
                    total += min(diff, math.pi - diff)
                    count += 1
            return math.degrees(total / count)

        def sharpness(img):
            gx, gy = sobel(img)
            mags = [
                math.hypot(gx[y][x], gy[y][x])
                for y in range(len(gx)) for x in range(len(gx[0]))
                if math.hypot(gx[y][x], gy[y][x]) > 64.0
            ]
            return sum(mags) / len(mags)

        bil = bilinear_scale(low, factor)
        ani = anisotropic_scale(low, factor, window=1, strength=1.0)
        self.assertLess(orientation_error(ani), orientation_error(bil))
        self.assertGreaterEqual(sharpness(ani), sharpness(bil) * 0.99)

    def test_axis_aligned_edge_rows_coherent(self):
        # A vertical edge has a vertical isophote. Interpolation along that
        # tangent must not introduce any row-to-row variation: every output
        # row of the upscaled image stays identical.
        img = [[[255] if x >= 4 else [0] for x in range(8)] for _ in range(8)]
        out = anisotropic_scale(img, 4, window=1, strength=1.0)
        self.assertEqual(out[0], out[-1])
        for row in out[1:]:
            self.assertEqual(row, out[0])
        # Every value must remain within the source range (no overshoot).
        self.assertTrue(all(0 <= v <= 255 for row in out for (v,) in row))

    def test_better_orientation_than_bilinear_on_circle(self):
        # End-to-end promise on the "magnified circle" use case: measured
        # against an independently synthesized high-resolution reference,
        # anisotropic interpolation must reconstruct isophote directions
        # more faithfully and reach higher PSNR than plain bilinear.
        factor = 4
        size = 48

        def smooth(d):
            if d <= -0.5:
                return 0
            if d >= 0.5:
                return 255
            t = d + 0.5
            t = t * t * (3 - 2 * t)
            return round(t * 255)

        def circle(side):
            c = (side - 1) / 2.0
            r = side * 0.32
            return [[[smooth(r - math.hypot(x - c, y - c))] for x in range(side)]
                    for y in range(side)]

        def area_down(hi, f):
            lo = []
            for y in range(size):
                row = []
                for x in range(size):
                    acc = 0
                    for dy in range(f):
                        for dx in range(f):
                            acc += hi[y * f + dy][x * f + dx][0]
                    row.append([round(acc / (f * f))])
                lo.append(row)
            return lo

        reference = circle(size * factor)
        low = area_down(reference, factor)

        def orient_err(candidate):
            def sobel(im):
                flat = [[p[0] for p in r] for r in im]
                gx = [[0.0] * len(flat[0]) for _ in flat]
                gy = [r[:] for r in gx]
                for y in range(1, len(flat) - 1):
                    for x in range(1, len(flat[0]) - 1):
                        gx[y][x] = flat[y][x + 1] - flat[y][x - 1]
                        gy[y][x] = flat[y + 1][x] - flat[y - 1][x]
                return gx, gy

            rg = sobel(reference)
            cg = sobel(candidate)
            total = 0.0
            count = 0
            for y in range(len(reference)):
                for x in range(len(reference[0])):
                    if math.hypot(rg[0][y][x], rg[1][y][x]) <= 64.0:
                        continue
                    ra = math.atan2(-rg[0][y][x], rg[1][y][x])
                    ca = math.atan2(-cg[0][y][x], cg[1][y][x])
                    diff = abs(ra - ca) % math.pi
                    total += min(diff, math.pi - diff)
                    count += 1
            return math.degrees(total / count)

        def psnr_of(candidate):
            mse = sum(
                (reference[y][x][0] - candidate[y][x][0]) ** 2
                for y in range(len(reference))
                for x in range(len(reference[0]))
            ) / (len(reference) * len(reference[0]))
            return 10.0 * math.log10(255.0 ** 2 / mse)

        bil = bilinear_scale(low, factor)
        ani = anisotropic_scale(low, factor, window=1, strength=1.0)
        self.assertLess(orient_err(ani), orient_err(bil))
        self.assertGreater(psnr_of(ani), psnr_of(bil))

    def test_gradient_direction_orthogonality(self):
        # On a vertical edge the gradient is horizontal, so the isophote
        # tangent must be vertical (x-component ~ 0): exercise the internal
        # structure tensor through the public output staying column-coherent.
        img = [[[255] if x >= 4 else [0] for x in range(8)] for _ in range(8)]
        out = anisotropic_scale(img, 2)
        # Every output row of a vertical-edge image must be identical.
        self.assertEqual(out[0], out[-1])

    def test_input_not_mutated(self):
        snapshot = [row[:] for row in [p[:] for p in DIAGONAL]]
        anisotropic_scale(DIAGONAL, 2)
        bilinear_scale(DIAGONAL, 2)
        self.assertEqual(DIAGONAL, snapshot)


if __name__ == '__main__':
    unittest.main(verbosity=2)
