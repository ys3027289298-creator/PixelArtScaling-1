"""
=======
ScaleNx
=======

------------------------------------------------------------------------------
Scale2x, Scale3x, Scale2xSFX and Scale3xSFX image rescaling for Python >=3.10.
------------------------------------------------------------------------------

:Abstract: Current module comprise **Scale2x**, **Scale3x**, **Scale2xSFX**
    and **Scale3xSFX** image rescaling functions, implemented in pure Python.

    Scale2x, Scale3x and Scale4x algorithms `[1]`_ were initially developed by
    `Andrea Mazzoleni`_ in 2001 AD for the sole purpose of upscaling
    low resolution screen output without introducing intermediate colors
    (*i.e.* blur) for old DOS games emulators `[2]`_ and
    similar narrow niche software.

    Later on *ca.* 2014 AD significantly improved Scale2xSFX and Scale3xSFX
    algorithms `[3]`_, providing better diagonals rendering, were introduced
    for the same very specific purpose.

    While screen scalers, based on algorithms above, are numerous,
    general purpose image rescaling wants still seem to be unsupplied.

    Due to severe demand for general purpose ScaleNx library,
    and apparent lack thereof, current general purpose pure Python
    `ScaleNx`_ implementation was developed.

Usage
-----

::

    from scalenx import scaleNx
    result_image = scaleNx(source_image, n, sfx)

where:

- **``source_image``**: source image 3D nested list;
        coordinate system match Photoshop, *i.e.* origin is top left corner,
        channels order is LA or RGBA from 0 to top;
- **``n``**: choice between Scale2* and Scale3* methods:
    - ``n=2``: Scale2x or Scale2xSFX;
    - ``n=3``: Scale3x or Scale3xSFX;

- **``sfx``**: choice between original ScaleNx and improved ScaleNxSFX methods:
    - ``sfx=False``: Scale2x or Scale3x;
    - ``sfx=True``: Scale2xSFX or Scale3xSFX.

.. note:: Function name **``scaleNx``** must contain capital **N**
    to avoid confusion with legacy file/module names.

Compatibility info
------------------

Current implementation is proven to work with CPython 3.10 and above,
and supposed to do so with any other Python version
understanding or ignoring type hints *etc.*

Copyright and redistribution
----------------------------

Current Python `ScaleNx`_ implementation is developed by
Ilya Razmanov (hereinafter referred to as "the Developer"), based on
algorithm descriptions `[1]`_, `[3]`_ by corresponding originators.

Changes introduced by the Developer for the purpose of
speed-up are entirely on his conscience.

Current implementation may be freely used, redistributed and
improved at will by anyone.
Sharing useful modifications with the Developer and lesser species
is your sacred duty to the Universe.

References
----------

`[1]`_. Original description of Scale2x and Scale3x algorithms by `Andrea Mazzoleni`_.

`[2]`_. DOSBox - DOS emulator, using Scale2x and Scale3x screen upscaling.

`[3]`_. Original Scale2xSFX and Scale3xSFX proposal, archived copy.

.. _[1]: https://www.scale2x.it/algorithm

.. _[2]: https://www.dosbox.com/

.. _[3]: https://web.archive.org/web/20160527015550/https://libretro.com/forums/archive/index.php?t-1655.html

.. _Andrea Mazzoleni: https://www.scale2x.it/authors

Resources
---------

The Developer's site: `The Toad's Slimy Mudhole`_;

.. _The Toad's Slimy Mudhole: https://dnyarri.github.io

`ScaleNx`_ explanations and illustrations page for current implementation;

.. _ScaleNx: https://dnyarri.github.io/scalenx.html

ScaleNx source repositories: `ScaleNx@Github`_, `ScaleNx@Gitflic`_;

.. _ScaleNx@Github: https://github.com/Dnyarri/PixelArtScaling

.. _ScaleNx@Gitflic: https://gitflic.ru/project/dnyarri/pixelartscaling

`Changelog`_ for current implementation.

.. _Changelog: https://github.com/Dnyarri/PixelArtScaling/blob/main/CHANGELOG.md

"""

__author__ = 'Ilya Razmanov'
__copyright__ = '(c) 2024-2026 Ilya Razmanov'
__credits__ = 'Ilya Razmanov'
__license__ = 'unlicense'
__version__ = '2026.8.6.312'
__maintainer__ = 'Ilya Razmanov'
__email__ = 'ilyarazmanov@gmail.com'
__status__ = 'Production'
__all__ = ['scaleNx', 'compare_scalers', 'format_compare_report', 'side_by_side']

import math
from time import perf_counter
from typing import Literal

from .scalenx import scale2x, scale3x
from .scalenxsfx import scale2x as scale2xsfx
from .scalenxsfx import scale3x as scale3xsfx


def scaleNx(source_image: list[list[list[int]]], n: Literal[2, 3], sfx: bool) -> list[list[list[int]]]:
    """ScaleNx image rescaling, configurable via ``n`` and ``sfx`` options.
    ----

    :param source_image: source image 3D nested list;
        coordinate system match Photoshop, *i.e.* origin is top left corner,
        channels order is LA or RGBA from 0 to top;
    :type source_image: list[list[list[int]]]
    :param int n: ``2`` or ``3``, choice between Scale2* and Scale3* methods;
    :param bool sfx: choice between ScaleNx and ScaleNxSFX methods.
    :raises ValueError: Attempt to use nonexistent method ``n``.
    :return: rescaled image of the same type as ``source_image``.
    :rtype: list[list[list[int]]]

    """
    if sfx:
        if n == 2:
            return scale2xsfx(source_image)
        elif n == 3:
            return scale3xsfx(source_image)
        else:
            raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')
    else:
        if n == 2:
            return scale2x(source_image)
        elif n == 3:
            return scale3x(source_image)
        else:
            raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')


""" ╔══════════════════════════════════════════════════════════════╗
    ║ Upscaling algorithm comparison: metrics, weighted scoring,   ║
    ║ and side-by-side output for picking a scaler for artists     ║
    ╚══════════════════════════════════════════════════════════════╝ """

#: Metrics included in every comparison report, in table order.
COMPARE_METRICS = ('psnr', 'sharpness', 'colors', 'time')

#: Metric direction: True when a bigger value is better.
METRIC_HIGHER_IS_BETTER = {'psnr': True, 'sharpness': True, 'colors': False, 'time': False}

#: Default metric weights, overridable through ``compare_scalers(weights=...)``.
DEFAULT_METRIC_WEIGHTS = {'psnr': 1.0, 'sharpness': 1.0, 'colors': 1.0, 'time': 1.0}

#: Built-in ScaleNx methods available by name for comparison.
COMPARE_ALGORITHMS = ('Scale2x', 'Scale3x', 'Scale2xSFX', 'Scale3xSFX')

#: PSNR reported for pixel-identical images, in dB; keeps report values finite.
PSNR_CAP_DB = 99.0


def _check_image3d(image3d: list[list[list[int]]], label: str = 'image3d') -> tuple[int, int, int]:
    """Validate 3D nested list geometry, returning ``(Y, X, Z)``.
    ----

    :param image3d: image as a non-empty list of non-empty row lists of
        non-empty pixel sequences;
    :param str label: name used in error messages;
    :raises ValueError: empty, ragged, or otherwise malformed image;
    :return: image height, width, and channel count;
    :rtype: tuple[int, int, int]

    """

    if not isinstance(image3d, list) or not image3d or not isinstance(image3d[0], list) or not image3d[0]:
        raise ValueError(f'`{label}` must be a non-empty list of non-empty row lists.')
    Y, X = (len(image3d), len(image3d[0]))
    if any(not isinstance(row, list) or len(row) != X for row in image3d):
        raise ValueError(f'`{label}` rows must be lists of equal length (ragged image given).')
    try:
        Z = len(image3d[0][0])
    except TypeError as error:
        raise ValueError(f'`{label}` pixels must be sequences of int channel values.') from error
    if Z < 1:
        raise ValueError(f'`{label}` pixels must contain at least one channel.')
    for row in image3d:
        for pixel in row:
            if len(pixel) != Z:
                raise ValueError(f'`{label}` pixels must all have the same number of channels ({Z}).')
    return (Y, X, Z)


def _maxval_of(image3d: list[list[list[int]]]) -> int:
    """Detect per-channel maximum from actual pixel values, matching the
    1/8/16 bpc ranges (``1``, ``255``, ``65535``) used by PyPNG and PyPNM."""

    peak = 0
    for row in image3d:
        for pixel in row:
            pixel_max = max(pixel)
            if pixel_max > peak:
                peak = pixel_max
    if peak <= 1:
        return 1
    if peak <= 255:
        return 255
    if peak <= 65535:
        return 65535
    return peak


def _nearest_upscale(image3d: list[list[list[int]]], factor: int) -> list[list[list[int]]]:
    """Nearest neighbour upscale, used as the same-factor fidelity reference."""

    Y, X = (len(image3d), len(image3d[0]))
    return [[list(image3d[y // factor][x // factor]) for x in range(X * factor)] for y in range(Y * factor)]


def _psnr(reference: list[list[list[int]]], image3d: list[list[list[int]]], maxval: int) -> float:
    """PSNR in dB between two same-sized images, capped at :data:`PSNR_CAP_DB`."""

    squared_error = 0
    count = 0
    for row_reference, row_image in zip(reference, image3d):
        for pixel_reference, pixel_image in zip(row_reference, row_image):
            for value_reference, value_image in zip(pixel_reference, pixel_image):
                delta = value_reference - value_image
                squared_error += delta * delta
                count += 1
    if count == 0:
        return PSNR_CAP_DB
    mse = squared_error / count
    if mse == 0:
        return PSNR_CAP_DB
    return min(PSNR_CAP_DB, 10.0 * math.log10((maxval * maxval) / mse))


def _edge_sharpness(image3d: list[list[list[int]]]) -> float:
    """Mean absolute neighbour difference per channel; higher is sharper."""

    Y, X = (len(image3d), len(image3d[0]))
    total = 0
    count = 0
    for y in range(Y):
        row = image3d[y]
        row_below = image3d[y + 1] if y + 1 < Y else None
        for x in range(X):
            pixel = row[x]
            if x + 1 < X:
                neighbour = row[x + 1]
                for channel, value in enumerate(pixel):
                    total += abs(value - neighbour[channel])
                count += len(pixel)
            if row_below is not None:
                neighbour = row_below[x]
                for channel, value in enumerate(pixel):
                    total += abs(value - neighbour[channel])
                count += len(pixel)
    return total / count if count else 0.0


def _count_colors(image3d: list[list[list[int]]]) -> int:
    """Number of distinct pixel colours in an image."""

    return len({tuple(pixel) for row in image3d for pixel in row})


def _median(values: list[float]) -> float:
    """Median of a non-empty list of numbers."""

    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _resolve_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Validate and merge metric weights over :data:`DEFAULT_METRIC_WEIGHTS`.
    ----

    :param weights: mapping of metric name to a non-negative finite weight,
        overriding the defaults for the keys given; ``None`` keeps defaults;
    :raises TypeError: ``weights`` is not a dict;
    :raises ValueError: unknown metric, non-numeric/negative weight,
        or a weight set that is zero in total;
    :return: full weight mapping covering every metric in
        :data:`COMPARE_METRICS`;
    :rtype: dict[str, float]

    """

    resolved = dict(DEFAULT_METRIC_WEIGHTS)
    if weights is None:
        return resolved
    if not isinstance(weights, dict):
        raise TypeError(f'`weights` must be a dict, got `{type(weights).__name__}`.')
    for metric, weight in weights.items():
        if metric not in COMPARE_METRICS:
            raise ValueError(f'Unknown metric `{metric}` in weights; supported metrics are {COMPARE_METRICS}.')
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f'Weight for metric `{metric}` must be a non-negative finite number, got `{weight}`.')
        resolved[metric] = float(weight)
    if sum(resolved.values()) <= 0:
        raise ValueError('At least one metric weight must be positive.')
    return resolved


def _resolve_algorithms(algorithms: list | tuple | None) -> list[tuple[str, object]]:
    """Resolve the algorithm list into ``(display name, runner)`` pairs.
    ----

    Entries may be built-in names from :data:`COMPARE_ALGORITHMS`
    (case-insensitive) or callables accepting an image and returning an
    upscaled image. Duplicate callable names get a ``#N`` suffix.

    :param algorithms: list/tuple of names/callables, or ``None`` for all
        built-in algorithms;
    :raises TypeError: ``algorithms`` is a string or not a list/tuple of
        names/callables;
    :raises ValueError: empty list, unknown built-in name, or a duplicate
        built-in method;
    :return: ordered list of resolved display names and runners;
    :rtype: list[tuple[str, object]]

    """

    builtin = {name.lower(): name for name in COMPARE_ALGORITHMS}
    builtin_runners = {
        'Scale2x': scale2x,
        'Scale3x': scale3x,
        'Scale2xSFX': scale2xsfx,
        'Scale3xSFX': scale3xsfx,
    }

    if algorithms is None:
        selected: list = list(COMPARE_ALGORITHMS)
    else:
        if isinstance(algorithms, (str, bytes)) or not isinstance(algorithms, (list, tuple)):
            raise TypeError('`algorithms` must be a list or tuple of algorithm names/callables.')
        if len(algorithms) == 0:
            raise ValueError('Algorithm list is empty: nothing to compare.')
        selected = list(algorithms)

    resolved: list[tuple[str, object]] = []
    used_names: set[str] = set()
    for entry in selected:
        if isinstance(entry, str):
            canonical = builtin.get(entry.lower())
            if canonical is None:
                raise ValueError(f'Unknown algorithm `{entry}`; supported built-in names are {COMPARE_ALGORITHMS}.')
            if canonical in used_names:
                raise ValueError(f'Algorithm `{canonical}` is listed more than once.')
            resolved.append((canonical, builtin_runners[canonical]))
            used_names.add(canonical)
        elif callable(entry):
            base_name = getattr(entry, '__name__', 'algorithm')
            display = base_name
            suffix = 2
            while display in used_names:
                display = f'{base_name}#{suffix}'
                suffix += 1
            resolved.append((display, entry))
            used_names.add(display)
        else:
            raise TypeError(f'Algorithm entry must be a name or a callable, got `{type(entry).__name__}`.')
    return resolved


def _resolve_background(background, channels: int) -> list[int]:
    """Validate side-by-side background colour, defaulting to opaque black."""

    if background is None:
        return [0] * channels
    if not isinstance(background, (list, tuple)) or len(background) != channels:
        raise ValueError(f'`background` must be a sequence of exactly {channels} channel value(s).')
    return list(background)


def side_by_side(images: list[list[list[list[int]]]], background: list[int] | None = None, gap: int = 0) -> list[list[list[int]]]:
    """Place images in a horizontal row, top-aligned, padded to common height.
    ----

    Images keep their native size; shorter panels are padded at the bottom
    with ``background``. Panels are separated by ``gap`` columns of the same
    background colour, so the result width is the sum of panel widths plus
    ``gap`` times the number of gaps.

    :param images: non-empty list of images sharing one channel count;
    :param background: padding/gap colour, one value per channel;
        defaults to all zeros (black, fully transparent for alpha images);
    :param int gap: non-negative separator width in pixels;
    :raises ValueError: empty image list, mismatched channel counts,
        or malformed background/gap;
    :return: side-by-side image as a 3D nested list;
    :rtype: list[list[list[int]]]

    """

    if not isinstance(images, (list, tuple)) or len(images) == 0:
        raise ValueError('`images` must be a non-empty list of images.')
    if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
        raise ValueError(f'`gap` must be a non-negative integer, got `{gap}`.')
    checked = [_check_image3d(image, f'images[{index}]') for index, image in enumerate(images)]
    channels = checked[0][2]
    for index, (_, _, image_channels) in enumerate(checked):
        if image_channels != channels:
            raise ValueError(f'All images must have the same channel count; images[{index}] has {image_channels}, expected {channels}.')
    fill = _resolve_background(background, channels)

    heights = [geometry[0] for geometry in checked]
    widths = [geometry[1] for geometry in checked]
    total_width = sum(widths) + gap * (len(images) - 1)

    result: list[list[list[int]]] = []
    for y in range(max(heights)):
        row: list[list[int]] = []
        for index, image in enumerate(images):
            if y < heights[index]:
                row.extend([list(pixel) for pixel in image[y]])
            else:
                row.extend([list(fill) for _ in range(widths[index])])
            if gap and index < len(images) - 1:
                row.extend([list(fill) for _ in range(gap)])
        result.append(row)
    return result


def compare_scalers(
    image3d: list[list[list[int]]],
    algorithms: list | tuple | None = None,
    weights: dict[str, float] | None = None,
    repeats: int = 1,
    background: list[int] | None = None,
    gap: int = 0,
) -> dict:
    """Compare upscaling algorithms on one image and build a report.
    ----

    Every algorithm is applied to the *same* source image. Metric values:

    - ``psnr``: PSNR in dB against nearest neighbour upscaling of the source
      at the same factor, capped at 99 dB; higher is better;
    - ``sharpness``: mean absolute neighbour difference per channel;
      higher is sharper;
    - ``colors``: number of distinct pixel colours in the result;
      lower means less palette drift;
    - ``time``: median wall-clock seconds per single-image run over
      ``repeats`` runs; lower is faster.

    Each metric is min-max normalised across the compared algorithms and
    combined into a 0..100 score with ``weights``. Colors and time are
    inverted (fewer colours and less time are better). Ties are allowed:
    every algorithm sharing the top score is marked best.

    The returned side-by-side image holds algorithm results in the
    requested order, with no source panel.

    :param image3d: source image 3D nested list;
    :param algorithms: list/tuple of names from :data:`COMPARE_ALGORITHMS`
        (case-insensitive) or callables returning upscaled images; ``None``
        selects all four built-in ScaleNx methods. An explicitly empty list
        raises :exc:`ValueError`;
    :param weights: metric weight overrides, see :func:`_resolve_weights`;
    :param int repeats: timing runs per algorithm, at least 1;
    :param background: padding/gap colour for the side-by-side image;
    :param int gap: separator width between side-by-side panels;
    :raises ValueError: empty algorithm list, unknown algorithm, empty or
        ragged image, non-integer upscale factor, or invalid parameters;
    :raises TypeError: malformed ``weights``/``algorithms`` container;
    :return: report dict with keys:

        - ``metrics``: metric name tuple;
        - ``weights``: resolved weight mapping;
        - ``algorithms``: algorithm names in comparison order;
        - ``rows``: result dicts with ``algorithm``, ``factor``, ``width``,
          ``height``, ``psnr``, ``sharpness``, ``colors``, ``seconds``,
          ``score``, ``rank``, ``best_overall``, ``best_metrics``;
        - ``best``: dict with ``'overall'`` and per-metric name lists;
        - ``outputs``: mapping of algorithm name to its output image;
        - ``side_by_side``: combined image;
        - ``size``: side-by-side ``(width, height)``;

    :rtype: dict

    """

    Y, X, Z = _check_image3d(image3d)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f'`repeats` must be a positive integer, got `{repeats}`.')
    resolved_weights = _resolve_weights(weights)
    resolved_algorithms = _resolve_algorithms(algorithms)

    # ↓ Running every algorithm on the same source image
    rows: list[dict] = []
    outputs: dict[str, list[list[list[int]]]] = {}
    for display_name, runner in resolved_algorithms:
        timings: list[float] = []
        output = None
        for run in range(repeats):
            started = perf_counter()
            result = runner(image3d)
            timings.append(perf_counter() - started)
            if run == 0:
                output = result

        output_height, output_width, output_channels = _check_image3d(output, f'Output of `{display_name}`')
        if output_channels != Z:
            raise ValueError(f'Algorithm `{display_name}` changed channel count from {Z} to {output_channels}.')
        if output_height % Y or output_width % X:
            raise ValueError(f'Algorithm `{display_name}` output {output_width}x{output_height} is not an integer multiple of source {X}x{Y}.')
        factor_y, factor_x = output_height // Y, output_width // X
        if factor_y != factor_x or factor_y < 2:
            raise ValueError(f'Algorithm `{display_name}` must upscale by the same integer factor >= 2 in both directions, got {factor_x}x{factor_y}.')

        reference = _nearest_upscale(image3d, factor_y)
        if len(reference) != output_height or len(reference[0]) != output_width:
            raise ValueError(f'Algorithm `{display_name}` output size mismatch against nearest neighbour reference.')

        outputs[display_name] = output
        median_seconds = _median(timings)
        rows.append({
            'algorithm': display_name,
            'factor': factor_y,
            'width': output_width,
            'height': output_height,
            'psnr': _psnr(reference, output, _maxval_of(image3d)),
            'sharpness': _edge_sharpness(output),
            'colors': _count_colors(output),
            'time': median_seconds,
            'seconds': median_seconds,
        })

    # ↓ Normalising each metric and computing the weighted score
    normalized: dict[str, list[float]] = {}
    for metric in COMPARE_METRICS:
        values = [row[metric] for row in rows]
        lowest, highest = min(values), max(values)
        if highest == lowest:
            normalized[metric] = [1.0] * len(rows)
        else:
            scaled = [(value - lowest) / (highest - lowest) for value in values]
            if not METRIC_HIGHER_IS_BETTER[metric]:
                scaled = [1.0 - value for value in scaled]
            normalized[metric] = scaled

    weight_total = sum(resolved_weights.values())
    for index, row in enumerate(rows):
        score = sum(resolved_weights[metric] * normalized[metric][index] for metric in COMPARE_METRICS) / weight_total
        row['score'] = 100.0 * score

    best_score = max(row['score'] for row in rows)
    for row in rows:
        row['rank'] = 1 + sum(1 for other in rows if other['score'] > row['score'])
        row['best_overall'] = row['score'] == best_score

    best_per_metric: dict[str, list[str]] = {}
    for metric in COMPARE_METRICS:
        values = [row[metric] for row in rows]
        target = max(values) if METRIC_HIGHER_IS_BETTER[metric] else min(values)
        best_per_metric[metric] = [row['algorithm'] for row in rows if row[metric] == target]

    for row in rows:
        row['best_metrics'] = [metric for metric in COMPARE_METRICS if row['algorithm'] in best_per_metric[metric]]

    combined = side_by_side([outputs[row['algorithm']] for row in rows], background=background, gap=gap)

    return {
        'metrics': COMPARE_METRICS,
        'weights': resolved_weights,
        'algorithms': [row['algorithm'] for row in rows],
        'rows': rows,
        'best': {'overall': [row['algorithm'] for row in rows if row['best_overall']], **best_per_metric},
        'outputs': outputs,
        'side_by_side': combined,
        'size': (len(combined[0]), len(combined)),
    }


def format_compare_report(report: dict, precision: int = 4) -> str:
    """Render a :func:`compare_scalers` report as a plain-text table.
    ----

    The best overall algorithm(s) are marked ``*`` in the ``Best`` column.
    A footer lists the winner for every metric, the metric weights used,
    and the side-by-side image size.

    :param report: dict returned by :func:`compare_scalers`;
    :param int precision: decimal places for time and score;
    :raises TypeError: ``report`` is not a dict;
    :raises ValueError: ``precision`` is negative;
    :return: multi-line text report;
    :rtype: str

    """

    if not isinstance(report, dict):
        raise TypeError(f'`report` must be a dict, got `{type(report).__name__}`.')
    if isinstance(precision, bool) or not isinstance(precision, int) or precision < 0:
        raise ValueError(f'`precision` must be a non-negative integer, got `{precision}`.')

    headers = ('Algorithm', 'Factor', 'Size', 'PSNR dB', 'Sharpness', 'Colors', 'Time s', 'Score', 'Best')
    lines = ['Upscaling comparison report', '']
    table: list[tuple[str, ...]] = []
    for row in report['rows']:
        table.append((
            row['algorithm'],
            str(row['factor']),
            f"{row['width']}x{row['height']}",
            f"{row['psnr']:.3f}",
            f"{row['sharpness']:.3f}",
            str(row['colors']),
            f"{row['seconds']:.{precision}f}",
            f"{row['score']:.{precision}f}",
            '*' if row['best_overall'] else '',
        ))
    widths = [max(len(headers[column]), *(len(line[column]) for line in table)) for column in range(len(headers))]
    lines.append('  '.join(header.ljust(widths[column]) for column, header in enumerate(headers)))
    lines.append('  '.join('-' * widths[column] for column in range(len(headers))))
    for line in table:
        lines.append('  '.join(value.ljust(widths[column]) for column, value in enumerate(line)))
    lines.append('')
    lines.append('Best overall: ' + ', '.join(report['best']['overall']))
    lines.append('Best by metric: ' + '; '.join(f"{metric}: {', '.join(report['best'][metric]) or '-'}" for metric in report['metrics']))
    lines.append('Weights: ' + ', '.join(f'{metric}={report["weights"][metric]:g}' for metric in report['metrics']))
    lines.append(f'Side-by-side size: {report["size"][0]}x{report["size"][1]} px')
    return '\n'.join(lines)
