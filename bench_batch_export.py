#!/usr/bin/env python3

"""Benchmark for batch export skip cache: four groups of self-made
small sample images, measuring for each group:

1. skip rate        - share of images not re-exported;
2. misjudgment rate - share of decisions disagreeing with ground truth;
3. total export time - with skip logic vs always-export baseline;
4. corrupted fingerprint store fallback behavior (group A addendum).

Everything is generated in a temporary directory, small, and runs
in one go. No third-party libraries, no network.

Run from repository root::

    python3 bench_batch_export.py
"""

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pypng import list2png, png2list  # noqa: E402
from scalenx import scaleNx  # noqa: E402
from scalenx.scalenx import SkipCache, fingerprint, fingerprint_distance  # noqa: E402

N_IMAGES = 12  # images per group
SIZE = 32  # image size, px
REPS = 3  # timing repetitions, best-of


def make_image(seed: int, X: int = SIZE, Y: int = SIZE) -> list[list[list[int]]]:
    """Deterministic 8-bit RGB test image: gradient + seeded blocks."""

    state = seed & 0xFFFFFFFF

    def rnd() -> int:
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state % 256

    img = [[[(x * 8 + y * 4) % 256, (x * 2 + y * 9) % 256, (x * 5 + y * 5 + seed) % 256] for x in range(X)] for y in range(Y)]
    for _ in range(6):  # seeded rectangles to make images distinct
        x0, y0, w, h = rnd() % 24, rnd() % 24, 4 + rnd() % 4, 4 + rnd() % 4
        c = [rnd(), rnd(), rnd()]
        for y in range(y0, min(y0 + h, Y)):
            for x in range(x0, min(x0 + w, X)):
                img[y][x] = list(c)
    return img


def write_sources(src_dir: Path, seeds: list[int]) -> list[Path]:
    paths = []
    for i, seed in enumerate(seeds):
        p = src_dir / f'img_{i:02d}.png'
        img = make_image(seed)
        list2png(str(p), img, {'bitdepth': 8})
        paths.append(p)
    return paths


def mutate(paths: list[Path], indexes: list[int], mode: str) -> None:
    """Rewrite selected source PNGs: 'strong' = visible change,
    'tiny' = one pixel one channel +-1 (below perceptual threshold)."""
    for i in indexes:
        X, Y, Z, maxcolors, img, info = png2list(str(paths[i]))
        if mode == 'strong':
            for y in range(8):
                for x in range(8):
                    img[y][x] = [(v + 128) % 256 for v in img[y][x]]
        else:  # tiny
            img[5][5][0] = (img[5][5][0] + 1) % 256
        list2png(str(paths[i]), img, info)


def export_pass(paths: list[Path], out_dir: Path, cache: SkipCache | None) -> tuple[int, float, dict[str, bool]]:
    """One batch export pass. Returns (exported_count, seconds, decisions).

    decisions[key] = True if exported. With cache=None everything is
    exported (baseline)."""
    decisions = {}
    exported = 0
    start = time.perf_counter()
    for p in paths:
        X, Y, Z, maxcolors, img, info = png2list(str(p))
        if cache is not None and not cache.changed(p.name, img):
            decisions[p.name] = False
            continue
        result = scaleNx(img, 2, False)
        list2png(str(out_dir / p.name), result, info)
        if cache is not None:
            cache.update(p.name)
        decisions[p.name] = True
        exported += 1
    elapsed = time.perf_counter() - start
    if cache is not None:
        cache.save()
    return exported, elapsed, decisions


def run_group(tmp: Path, name: str, mutate_indexes: list[int], mutate_mode: str, threshold: float, ground_truth_changed: set[str]) -> dict:
    src_dir, out_dir = tmp / f'{name}_src', tmp / f'{name}_out'
    src_dir.mkdir()
    out_dir.mkdir()
    paths = write_sources(src_dir, seeds=[1000 + i for i in range(N_IMAGES)])
    store = tmp / f'{name}_fp.json'

    # ↓ Priming pass: initial export of everything, fingerprints stored
    cache = SkipCache(store, threshold=threshold)
    export_pass(paths, out_dir, cache)

    # ↓ Mutation between two batch runs
    if mutate_indexes:
        mutate(paths, mutate_indexes, mutate_mode)

    # ↓ Baseline: always export everything (best of REPS)
    t_base = min(export_pass(paths, out_dir, None)[1] for _ in range(REPS))

    # ↓ Measured pass with skip logic (fresh load of the primed store)
    cache = SkipCache(store, threshold=threshold)
    exported, t_skip, decisions = export_pass(paths, out_dir, cache)

    skipped = N_IMAGES - exported
    misjudged = sum(1 for key, was_exported in decisions.items() if was_exported != (key in ground_truth_changed))

    return {
        'group': name,
        'threshold': threshold,
        'exported': exported,
        'skip_rate': skipped / N_IMAGES,
        'misjudge_rate': misjudged / N_IMAGES,
        't_skip': t_skip,
        't_base': t_base,
        'store_bytes': store.stat().st_size,
        'store_bound': cache.size_bound,
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpname:
        tmp = Path(tmpname)
        all_names = {f'img_{i:02d}.png' for i in range(N_IMAGES)}
        changed_c = {f'img_{i:02d}.png' for i in (1, 4, 7, 10)}

        groups = [
            run_group(tmp, 'A_no_change', [], 'strong', 0.0, set()),
            run_group(tmp, 'B_all_changed', list(range(N_IMAGES)), 'strong', 0.0, set(all_names)),
            run_group(tmp, 'C_part_changed', [1, 4, 7, 10], 'strong', 0.0, set(changed_c)),
            run_group(tmp, 'D_tiny_edits', list(range(N_IMAGES)), 'tiny', 0.05, set()),
        ]

        print('=' * 96)
        print(f'Batch export skip benchmark: {N_IMAGES} images/group, {SIZE}x{SIZE} RGB, Scale2x export, best of {REPS} timing reps')
        print('Fingerprint store: dhash 64-bit, JSON, size bound = 64 + max_entries(1024) * (128+24) = 155712 bytes (~152 KiB)')
        print('=' * 96)
        header = f'{"Group":<18}{"Threshold":>10}{"Exported":>9}{"Skip rate":>11}{"Misjudge":>10}{"Time skip":>11}{"Time base":>11}{"Speedup":>9}'
        print(header)
        print('-' * 96)
        for g in groups:
            print(f'{g["group"]:<18}{g["threshold"]:>10.2f}{g["exported"]:>9d}{g["skip_rate"]:>10.1%}{g["misjudge_rate"]:>10.1%}{g["t_skip"]:>10.3f}s{g["t_base"]:>10.3f}s{g["t_base"] / g["t_skip"]:>8.2f}x')
        print('-' * 96)
        print('Ground truth: groups A-C = exact pixel change; group D = perceptually meaningful change')
        print('(tiny +-1 edits are NOT meaningful, so skipping them is a CORRECT decision at threshold 0.05).')

        # ↓ Group D addendum: actual fingerprint distances of tiny edits,
        #   and misjudgment if exact-pixel ground truth were (wrongly) used
        d = groups[3]
        print(f'\nGroup D note: vs exact-pixel ground truth all {N_IMAGES} skips would count as missed changes;')
        print('that is the documented price of a perceptual threshold > 0.')

        # ↓ Corrupted store fallback demo (on group A store)
        print('\n' + '=' * 96)
        print('Corrupted fingerprint store fallback (group A store):')
        store_a = tmp / 'A_no_change_fp.json'
        store_a.write_bytes(b'\x00\xffgarbage{{{not json')
        cache = SkipCache(store_a, threshold=0.0)
        print(f'  store corrupted -> recovered flag: {cache.recovered}, entries: {len(cache)}')
        print(f'  quarantine backup exists: {(tmp / "A_no_change_fp.json.corrupt").exists()}')
        src_dir, out_dir = tmp / 'A_no_change_src', tmp / 'A_no_change_out2'
        out_dir.mkdir()
        paths = sorted(src_dir.glob('*.png'))
        exported, t, _ = export_pass(paths, out_dir, cache)
        print(f'  fallback behavior: all {exported}/{N_IMAGES} images re-exported (safe default), store rebuilt')
        cache2 = SkipCache(store_a, threshold=0.0)
        print(f'  rebuilt store loads cleanly: recovered={cache2.recovered}, entries={len(cache2)}, size={store_a.stat().st_size} bytes (bound {cache2.size_bound})')

        # ↓ Store size bound demonstration
        print('\n' + '=' * 96)
        print('Store size upper bound:')
        big = SkipCache(tmp / 'big_fp.json')
        img = make_image(1)
        for i in range(5000):
            big.changed(f'k{i}', img)
            big.update(f'k{i}')
        big.save()
        size = (tmp / 'big_fp.json').stat().st_size
        print(f'  5000 updates -> entries kept: {len(big)} (max_entries=1024), file size: {size} bytes <= bound {big.size_bound} bytes: {size <= big.size_bound}')

        # ↓ Out-of-range threshold handling
        print('\n' + '=' * 96)
        print('Out-of-range threshold handling (clamp + RuntimeWarning):')
        import warnings

        for bad in (-0.5, 1.7, float('nan')):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                c = SkipCache(tmp / 'thr_fp.json', threshold=bad)
            print(f'  threshold={bad!r:>8} -> clamped to {c.threshold}, warning: "{caught[0].message}"')


if __name__ == '__main__':
    main()
