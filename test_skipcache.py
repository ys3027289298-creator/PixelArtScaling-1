#!/usr/bin/env python3

"""Tests for batch export skip cache (fingerprint + SkipCache)
added to ``scalenx/scalenx.py``, plus a regression check that
``scale2x``/``scale3x`` output stayed pixel-identical after the change.

Run from repository root::

    python3 -m unittest test_skipcache -v
"""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scalenx import scaleNx  # noqa: E402
from scalenx.scalenx import (  # noqa: E402
    DEFAULT_THRESHOLD,
    MAX_KEY_LEN,
    SkipCache,
    fingerprint,
    fingerprint_distance,
    sanitize_threshold,
    scale2x,
    scale3x,
)


def make_image(X: int, Y: int, Z: int, seed: int) -> list[list[list[int]]]:
    """Small deterministic test image, no third-party libraries."""

    state = seed & 0xFFFFFFFF

    def rnd() -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state % 256

    return [[[rnd() for _ in range(Z)] for _ in range(X)] for _ in range(Y)]


class TestFingerprint(unittest.TestCase):
    def test_same_image_same_fingerprint_both_algos(self):
        img = make_image(16, 16, 3, 7)
        for algo in ('ahash', 'dhash'):
            self.assertEqual(fingerprint(img, algo), fingerprint(img, algo))

    def test_fingerprint_is_64bit(self):
        img = make_image(16, 16, 3, 7)
        for algo in ('ahash', 'dhash'):
            fp = fingerprint(img, algo)
            self.assertIsInstance(fp, int)
            self.assertGreaterEqual(fp, 0)
            self.assertLess(fp, 1 << 64)

    def test_different_images_differ(self):
        a = make_image(16, 16, 3, 1)
        b = make_image(16, 16, 3, 2)
        for algo in ('ahash', 'dhash'):
            self.assertGreater(fingerprint_distance(fingerprint(a, algo), fingerprint(b, algo)), 0.0)

    def test_identical_distance_is_zero(self):
        img = make_image(16, 16, 3, 3)
        fp = fingerprint(img)
        self.assertEqual(fingerprint_distance(fp, fp), 0.0)

    def test_grayscale_and_alpha_channels(self):
        for z in (1, 2, 4):
            img = make_image(12, 12, z, 5)
            self.assertIsInstance(fingerprint(img), int)

    def test_unknown_algo_raises(self):
        with self.assertRaises(ValueError):
            fingerprint(make_image(8, 8, 3, 1), algo='md5')


class TestThreshold(unittest.TestCase):
    def test_valid_threshold_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            self.assertEqual(sanitize_threshold(0.25), 0.25)
            self.assertEqual(sanitize_threshold(0.0), 0.0)
            self.assertEqual(sanitize_threshold(1.0), 1.0)

    def test_below_range_clamped_with_warning(self):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(sanitize_threshold(-0.5), 0.0)

    def test_above_range_clamped_with_warning(self):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(sanitize_threshold(1.5), 1.0)

    def test_nan_reset_with_warning(self):
        with self.assertWarns(RuntimeWarning):
            self.assertEqual(sanitize_threshold(float('nan')), DEFAULT_THRESHOLD)


class TestSkipCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store_path = Path(self.tmp.name) / 'fp.json'

    def tearDown(self):
        self.tmp.cleanup()

    def test_unchanged_skipped_changed_detected(self):
        img = make_image(16, 16, 3, 11)
        cache = SkipCache(self.store_path)
        self.assertTrue(cache.changed('a.png', img))  # never exported
        cache.update('a.png')
        self.assertFalse(cache.changed('a.png', img))  # unchanged -> skip
        img2 = make_image(16, 16, 3, 12)
        self.assertTrue(cache.changed('a.png', img2))  # changed -> export

    def test_persistence_across_instances(self):
        img = make_image(16, 16, 3, 21)
        cache = SkipCache(self.store_path)
        cache.changed('a.png', img)
        cache.update('a.png')
        cache.save()
        cache2 = SkipCache(self.store_path)
        self.assertFalse(cache2.recovered)
        self.assertFalse(cache2.changed('a.png', img))

    def test_algo_configurable(self):
        img = make_image(16, 16, 3, 31)
        for algo in ('ahash', 'dhash'):
            cache = SkipCache(Path(self.tmp.name) / f'fp_{algo}.json', algo=algo)
            cache.changed('a.png', img)
            cache.update('a.png')
            cache.save()
            self.assertFalse(SkipCache(cache.path, algo=algo).changed('a.png', img))
        with self.assertRaises(ValueError):
            SkipCache(self.store_path, algo='sha256')

    def test_threshold_out_of_range_clamped(self):
        with self.assertWarns(RuntimeWarning):
            cache = SkipCache(self.store_path, threshold=2.5)
        self.assertEqual(cache.threshold, 1.0)
        with self.assertWarns(RuntimeWarning):
            cache = SkipCache(self.store_path, threshold=-1)
        self.assertEqual(cache.threshold, 0.0)

    def test_perceptual_threshold_skips_tiny_edit(self):
        img = make_image(32, 32, 3, 41)
        edited = [[list(px) for px in row] for row in img]
        edited[5][5][0] = (edited[5][5][0] + 1) % 256  # 1 channel of 1 pixel
        # ↓ Tiny edit: fingerprint unchanged, skipped even at threshold 0
        cache = SkipCache(self.store_path, threshold=0.05)
        cache.changed('a.png', img)
        cache.update('a.png')
        self.assertFalse(cache.changed('a.png', edited))
        # ↓ Visible edit (4x4 block, big delta): flips fingerprint bits,
        #   detected at strict threshold 0
        strict = SkipCache(Path(self.tmp.name) / 'strict.json', threshold=0.0)
        strict.changed('a.png', img)
        strict.update('a.png')
        visible = [[list(px) for px in row] for row in img]
        for y in range(4):
            for x in range(4):
                visible[y][x][0] = (visible[y][x][0] + 128) % 256
        self.assertTrue(strict.changed('a.png', visible))

    def test_corrupt_store_fallback(self):
        img = make_image(16, 16, 3, 51)
        cache = SkipCache(self.store_path)
        cache.changed('a.png', img)
        cache.update('a.png')
        cache.save()
        # ↓ Corrupting the store file with garbage
        self.store_path.write_bytes(b'\x00\xff not json {{{')
        cache2 = SkipCache(self.store_path)
        self.assertTrue(cache2.recovered)
        self.assertEqual(len(cache2), 0)
        # ↓ Fallback behavior: everything is re-exported
        self.assertTrue(cache2.changed('a.png', img))
        # ↓ Corrupted file quarantined, store rebuilds cleanly
        self.assertTrue(self.store_path.with_name('fp.json.corrupt').exists())
        cache2.update('a.png')
        cache2.save()
        cache3 = SkipCache(self.store_path)
        self.assertFalse(cache3.recovered)
        self.assertFalse(cache3.changed('a.png', img))

    def test_malformed_structure_fallback(self):
        self.store_path.write_text(json.dumps({'entries': {'a.png': 'not-a-hash'}}), encoding='utf-8')
        cache = SkipCache(self.store_path)
        self.assertTrue(cache.recovered)
        self.assertEqual(len(cache), 0)

    def test_size_bound_enforced(self):
        cache = SkipCache(self.store_path, max_entries=4)
        self.assertEqual(cache.size_bound, 64 + 4 * (MAX_KEY_LEN + 24))
        for i in range(20):
            img = make_image(8, 8, 3, 100 + i)
            cache.changed(f'image_{i:03d}.png', img)
            cache.update(f'image_{i:03d}.png')
        self.assertLessEqual(len(cache), 4)
        cache.save()
        self.assertLessEqual(self.store_path.stat().st_size, cache.size_bound)
        # ↓ Most recent entries survive eviction
        self.assertFalse(cache.changed('image_019.png', make_image(8, 8, 3, 119)))

    def test_default_size_bound_value(self):
        self.assertEqual(SkipCache(self.store_path).size_bound, 155712)

    def test_key_too_long_raises(self):
        cache = SkipCache(self.store_path)
        with self.assertRaises(ValueError):
            cache.changed('x' * (MAX_KEY_LEN + 1), make_image(8, 8, 3, 1))

    def test_update_without_memo_raises(self):
        cache = SkipCache(self.store_path)
        with self.assertRaises(ValueError):
            cache.update('never-seen.png')


class TestScalingRegression(unittest.TestCase):
    """Scaling output must stay pixel-identical after adding the skip cache."""

    def test_dimensions_and_determinism(self):
        img = make_image(11, 13, 3, 61)
        for n in (2, 3):
            for sfx in (False, True):
                out1 = scaleNx(img, n, sfx)
                out2 = scaleNx(img, n, sfx)
                self.assertEqual(len(out1), n * len(img))
                self.assertEqual(len(out1[0]), n * len(img[0]))
                self.assertEqual(out1, out2)

    def test_pixel_identical_to_git_head(self):
        """Compare scale2x/scale3x output against pristine HEAD version."""
        try:
            src = subprocess.run(
                ['git', 'show', 'HEAD:scalenx/scalenx.py'],
                capture_output=True,
                check=True,
                cwd=Path(__file__).resolve().parent,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            self.skipTest(f'git HEAD not available: {exc}')
        with tempfile.TemporaryDirectory() as tmp:
            pristine = Path(tmp) / 'scalenx_head.py'
            pristine.write_bytes(src)
            spec = importlib.util.spec_from_file_location('scalenx_head', pristine)
            head = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(head)
            for z in (1, 3, 4):
                img = make_image(13, 11, z, 71 + z)
                self.assertEqual(scale2x(img), head.scale2x(img))
                self.assertEqual(scale3x(img), head.scale3x(img))


if __name__ == '__main__':
    unittest.main()
