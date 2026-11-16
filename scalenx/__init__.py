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

Batch processing
----------------

For processing hundreds of images at once, use :class:`BatchScheduler`
(or the :func:`process_batch` convenience wrapper). The scheduler:

- groups jobs into **buckets by target image size** (all images sharing the
  same source dimensions produce the same target size for a given ``n``);
- **reuses the scaling parameters** (scaler function, method flag and target
  geometry) for every image inside a bucket, creating them only once per
  bucket instead of once per image;
- checkpoints progress to disk and can **resume an interrupted run**, skipping
  already completed jobs and reusing the cumulative statistics;
- reports per-run and cumulative reuse-hit counters and elapsed time.

Batch output is produced by the very same single-image scalers, therefore
each image is pixel-for-pixel identical to calling :func:`scaleNx` on it
one by one.

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
__all__ = [
    'scaleNx',
    'BatchScheduler',
    'ScalingParams',
    'BatchReport',
    'BatchInterrupted',
    'process_batch',
]

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

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


""" ╔══════════════════════════════════════════════════════════╗
    ║ Batch scheduling: target-size buckets, parameter reuse,  ║
    ║ checkpoints and resumable runs.                          ║
    ╚══════════════════════════════════════════════════════════╝ """


class BatchInterrupted(Exception):
    """Raised when a batch run stops before all jobs are processed.

    The checkpoint is written before the exception is raised, so repeating
    the same :meth:`BatchScheduler.run` call with ``resume=True`` continues
    from the first unfinished job.

    :param str checkpoint: path of the checkpoint holding the progress.
    :param int processed: number of jobs processed in the current run.
    """

    def __init__(self, checkpoint: str, processed: int):
        self.checkpoint = checkpoint
        self.processed = processed
        super().__init__(f'Batch interrupted after {processed} job(s); progress saved to {checkpoint!r}.')


@dataclass(frozen=True)
class ScalingParams:
    """Scaling parameters shared by all images of one target-size bucket.

    Instances are immutable and created exactly once per bucket; every
    further image in the bucket reuses the same instance, which is what the
    scheduler counts as a reuse hit.

    :param int n: scaling method, ``2`` or ``3``;
    :param bool sfx: ``True`` selects the ScaleNxSFX algorithms;
    :param tuple source_size: ``(height, width)`` of every source in the bucket;
    :param tuple target_size: ``(height, width)`` of every scaled image;
    :param callable scaler: bound single-image scaling function.
    """

    n: int
    sfx: bool
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    scaler: Callable[[list[list[list[int]]]], list[list[list[int]]]]

    def __call__(self, image3d: list[list[list[int]]]) -> list[list[list[int]]]:
        """Scale one image using the bucket's bound single-image scaler."""
        return self.scaler(image3d)


@dataclass
class BatchReport:
    """Result and statistics of one :meth:`BatchScheduler.run` call.

    ``processed``, ``reuse_hits`` and ``elapsed`` describe the current call
    only. The ``cumulative_*`` fields cover the whole batch, including jobs
    finished in earlier resumed runs.

    :param int total: total number of jobs in the batch;
    :param int processed: jobs processed during the current call;
    :param int skipped: jobs skipped because an earlier run finished them;
    :param int buckets: number of target-size buckets built for the batch;
    :param int reuse_hits: jobs that reused an already created bucket's
        :class:`ScalingParams` instead of creating new parameters;
    :param float elapsed: wall-clock seconds spent in the current call;
    :param bool completed: ``True`` when every job has been processed;
    :param bool resumed: ``True`` when the call continued an earlier run;
    :param int cumulative_processed: distinct jobs finished across all runs;
    :param int cumulative_reuse_hits: reuse hits accumulated across all runs;
    :param float cumulative_elapsed: seconds spent across all runs;
    :param results: scaled images in input order, ``None`` for jobs not yet
        processed (absent when a ``sink`` consumes the output).
    """

    total: int = 0
    processed: int = 0
    skipped: int = 0
    buckets: int = 0
    reuse_hits: int = 0
    elapsed: float = 0.0
    completed: bool = False
    resumed: bool = False
    cumulative_processed: int = 0
    cumulative_reuse_hits: int = 0
    cumulative_elapsed: float = 0.0
    results: list[list[list[list[int]]] | None] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        """Fraction of jobs that reused bucket parameters (0.0..1.0)."""
        if self.total == 0:
            return 0.0
        return self.cumulative_reuse_hits / self.total


def _validate_image(image3d) -> tuple[int, int]:
    """Validate one image and return its ``(height, width)``.

    Both lists and tuples are accepted at row and pixel level, matching the
    scalers themselves (which special-case tuple pixels for caching).
    """
    if not isinstance(image3d, (list, tuple)) or not image3d:
        raise TypeError('Each image must be a non-empty sequence of rows.')
    first = image3d[0]
    if not isinstance(first, (list, tuple)) or len(first) == 0:
        raise ValueError('Images must contain at least one pixel.')
    width = len(first)
    for row in image3d:
        if not isinstance(row, (list, tuple)) or len(row) != width:
            raise ValueError('All image rows must be sequences of the same width.')
        for pixel in row:
            if not isinstance(pixel, (list, tuple)) or not pixel:
                raise ValueError('Each pixel must be a non-empty sequence of channel values.')
    return len(image3d), width


def _validate_bool(value, name: str) -> bool:
    """Strict bool validation that rejects, e.g., ``1`` passed as a flag."""
    if not isinstance(value, bool):
        raise TypeError(f'Argument {name!r} must be bool, got {type(value).__name__}.')
    return value


class BatchScheduler:
    """Resumable batch scheduler with target-size bucketing.

    :param int n: scaling method, either ``2`` or ``3``;
    :param bool sfx: ``True`` selects the ScaleNxSFX algorithms;
    :param int bucket_size: maximum number of jobs per bucket. Images with
        the same target size share one bucket until it reaches this size,
        then another bucket is opened; ``0`` means a single unbounded bucket
        per target size (reuse is then maximal);
    :param str | os.PathLike checkpoint: path of the JSON checkpoint file
        used to resume interrupted runs;
    :param bool resume: when ``True``, an existing checkpoint is continued
        instead of restarting the batch;
    :param callable sink: optional ``sink(index, job_id, scaled_image)``
        callback; when provided, scaled images are passed to it instead of
        being retained in the :class:`BatchReport`.

    All constructor arguments except ``n``/``sfx`` are tunable per
    :meth:`run` call as well; invalid values raise :class:`TypeError` or
    :class:`ValueError` before any image is touched.
    """

    #: checkpoint format marker, used to reject foreign/corrupt files
    _CHECKPOINT_VERSION = 1

    def __init__(
        self,
        n: Literal[2, 3],
        sfx: bool,
        bucket_size: int = 0,
        checkpoint: str | os.PathLike = 'scalenx_batch.json',
        resume: bool = False,
        sink: Callable[[int, str, list[list[list[int]]]], None] | None = None,
    ):
        self.n = self._validate_n(n)
        self.sfx = _validate_bool(sfx, 'sfx')
        self.bucket_size = self._validate_bucket_size(bucket_size)
        self.checkpoint = os.fspath(checkpoint)
        self.resume = _validate_bool(resume, 'resume')
        self.sink = self._validate_sink(sink)

    # ------------------------------------------------------------------ #
    # Argument validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_n(n) -> int:
        if not isinstance(n, int) or isinstance(n, bool):
            raise TypeError(f'Argument {n!r} must be int 2 or 3, got {type(n).__name__}.')
        if n not in (2, 3):
            raise ValueError(f'Attempt to use nonexistent method n=`{n}`\nOnly methods n=2 and n=3 allowed.')
        return n

    @staticmethod
    def _validate_bucket_size(bucket_size) -> int:
        if not isinstance(bucket_size, int) or isinstance(bucket_size, bool):
            raise TypeError(f'Argument bucket_size must be int, got {type(bucket_size).__name__}.')
        if bucket_size < 0:
            raise ValueError(f'Argument bucket_size must be >= 0, got {bucket_size}.')
        return bucket_size

    @staticmethod
    def _validate_sink(sink):
        if sink is not None and not callable(sink):
            raise TypeError(f'Argument sink must be callable or None, got {type(sink).__name__}.')
        return sink

    def _resolve_overrides(self, bucket_size, resume, sink):
        """Merge per-call overrides with constructor settings, validating all."""
        if bucket_size is None:
            bucket_size = self.bucket_size
        else:
            bucket_size = self._validate_bucket_size(bucket_size)
        if resume is None:
            resume = self.resume
        else:
            resume = _validate_bool(resume, 'resume')
        if sink is None and self.sink is not None:
            sink = self.sink
        elif sink is not None:
            sink = self._validate_sink(sink)
        return bucket_size, resume, sink

    # ------------------------------------------------------------------ #
    # Checkpoint (de)serialization
    # ------------------------------------------------------------------ #
    def _write_checkpoint(self, payload: dict, durable: bool = False) -> None:
        """Atomically write the checkpoint.

        The temp-file + :func:`os.replace` dance guarantees an observer never
        sees a half-written checkpoint, so per-job writes stay cheap. The
        ``durable`` flush (``fsync`` of data and directory) is performed once,
        when a run stops or finishes - precisely the moment that matters for
        crash recovery.
        """
        tmp_path = f'{self.checkpoint}.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, ensure_ascii=False)
            if durable:
                fh.flush()
                os.fsync(fh.fileno())
        os.replace(tmp_path, self.checkpoint)
        if durable:
            directory = os.path.dirname(os.path.abspath(self.checkpoint))
            try:
                dir_fd = os.open(directory, os.O_RDONLY)
            except OSError:
                return
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def _read_checkpoint(self) -> dict:
        try:
            with open(self.checkpoint, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Cannot resume: checkpoint {self.checkpoint!r} is missing or unreadable: {exc}') from exc
        if not isinstance(payload, dict) or payload.get('version') != self._CHECKPOINT_VERSION:
            raise ValueError(f'Cannot resume: checkpoint {self.checkpoint!r} has an unknown format.')
        return payload

    def _check_spec(self, payload: dict, jobs: list[tuple[str, object]], bucket_size: int) -> None:
        """Ensure the checkpoint describes exactly the run being resumed."""
        spec = payload.get('spec')
        if not isinstance(spec, dict):
            raise ValueError('Cannot resume: checkpoint is missing its run specification.')
        if spec.get('n') != self.n or spec.get('sfx') != self.sfx or spec.get('bucket_size') != bucket_size:
            raise ValueError('Cannot resume: checkpoint parameters (n, sfx or bucket_size) do not match the current run.')
        saved_jobs = spec.get('jobs')
        if not isinstance(saved_jobs, list) or len(saved_jobs) != len(jobs):
            raise ValueError('Cannot resume: checkpoint job list does not match the current batch.')
        for saved, (job_id, _) in zip(saved_jobs, jobs):
            if saved != job_id:
                raise ValueError('Cannot resume: checkpoint job order/content does not match the current batch.')

    # ------------------------------------------------------------------ #
    # Bucketing
    # ------------------------------------------------------------------ #
    def _build_buckets(self, sizes: list[tuple[int, int]], bucket_size: int) -> tuple[list[list[int]], int, int]:
        """Group job indices into target-size buckets.

        :param sizes: ``(height, width)`` of every job, in input order;
        :param bucket_size: maximum jobs per bucket (0 = unbounded).
        :return: buckets (lists of job indices), bucket count and the number
            of reuse hits (every job after the first inside each bucket).
        """
        groups: dict[tuple[int, int], list[int]] = {}
        order: list[tuple[int, int]] = []
        for index, (height, width) in enumerate(sizes):
            target = (height * self.n, width * self.n)
            if target not in groups:
                groups[target] = []
                order.append(target)
            groups[target].append(index)

        buckets: list[list[int]] = []
        for target in order:
            members = groups[target]
            if bucket_size == 0:
                buckets.append(members)
            else:
                for start in range(0, len(members), bucket_size):
                    buckets.append(members[start : start + bucket_size])

        reuse_hits = sum(len(bucket) - 1 for bucket in buckets)
        return buckets, len(buckets), reuse_hits

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        images: list[list[list[int]]],
        bucket_size: int | None = None,
        resume: bool | None = None,
        sink: Callable[[int, str, list[list[list[int]]]], None] | None = None,
        fail_after: int | None = None,
    ) -> BatchReport:
        """Process a batch of images.

        :param images: source images as 3D nested lists, in processing order;
        :param bucket_size: per-call override of the constructor setting;
        :param resume: per-call override of the constructor setting;
        :param sink: per-call override of the constructor sink;
        :param fail_after: optional testing hook - stop with
            :class:`BatchInterrupted` after this many newly processed jobs;
            ``None`` or a non-positive value disables it.
        :raises TypeError | ValueError: invalid arguments or image data;
        :raises BatchInterrupted: the run stopped early; the checkpoint on
            disk allows a later ``resume=True`` call to finish the batch.
        :return: report with results (unless a sink is used) and statistics.
        :rtype: BatchReport
        """
        if not isinstance(images, list) or not images:
            raise TypeError('Argument images must be a non-empty list of 3D nested-list images.')
        if fail_after is not None:
            if not isinstance(fail_after, int) or isinstance(fail_after, bool):
                raise TypeError(f'Argument fail_after must be int or None, got {type(fail_after).__name__}.')
        bucket_size, resume, sink = self._resolve_overrides(bucket_size, resume, sink)

        # ↓ Validate everything and build stable job ids before any scaling.
        sizes: list[tuple[int, int]] = []
        jobs: list[tuple[str, object]] = []
        for index, image in enumerate(images):
            height, width = _validate_image(image)
            sizes.append((height, width))
            jobs.append((f'job-{index:08d}', index))

        buckets, bucket_count, total_reuse_hits = self._build_buckets(sizes, bucket_size)

        # ↓ Resume: load and verify the previous checkpoint.
        completed: set[int] = set()
        cumulative_hits = 0
        cumulative_elapsed = 0.0
        resumed = False
        if resume:
            if not os.path.exists(self.checkpoint):
                raise ValueError(f'Cannot resume: checkpoint {self.checkpoint!r} does not exist.')
            payload = self._read_checkpoint()
            self._check_spec(payload, jobs, bucket_size)
            completed = set(payload.get('completed', ()))
            completed = {idx for idx in completed if isinstance(idx, int) and 0 <= idx < len(jobs)}
            cumulative_hits = int(payload.get('cumulative_reuse_hits', 0))
            cumulative_elapsed = float(payload.get('cumulative_elapsed', 0.0))
            resumed = True

        results: list[list[list[list[int]]] | None] = [None] * len(images) if sink is None else []

        # ↓ On resume, already finished jobs are re-scaled to fill the report
        #   (without invoking the sink and without counting as new work).
        skipped = 0
        for index in sorted(completed):
            skipped += 1
            if sink is None:
                results[index] = scaleNx(images[index], self.n, self.sfx)

        report = BatchReport(total=len(images), buckets=bucket_count, resumed=resumed, results=results)

        # ↓ Nothing left to do: return the finished-batch statistics.
        if len(completed) == len(images):
            report.processed = 0
            report.skipped = skipped
            report.reuse_hits = 0
            report.elapsed = 0.0
            report.completed = True
            report.cumulative_processed = len(images)
            report.cumulative_reuse_hits = total_reuse_hits
            report.cumulative_elapsed = cumulative_elapsed
            return report

        started = time.monotonic()
        processed_now = 0
        hits_now = 0
        interrupted = False

        try:
            for members in buckets:
                # ↓ Parameters are created once per bucket and reused by the
                #   remaining bucket members. On resume a bucket may already
                #   have completed members, in which case its first new job
                #   reuses the parameters created by the earlier run.
                params: ScalingParams | None = None
                for job_index in members:
                    if job_index in completed:
                        continue  # finished by an earlier run

                    height, width = sizes[job_index]
                    bucket_seen = any(other in completed for other in members if other != job_index)
                    if params is not None or bucket_seen:
                        hits_now += 1  # reused the bucket's ScalingParams
                        cumulative_hits += 1
                    if params is None:
                        params = ScalingParams(
                            n=self.n,
                            sfx=self.sfx,
                            source_size=(height, width),
                            target_size=(height * self.n, width * self.n),
                            scaler=self._bind_scaler(),
                        )

                    scaled = params(images[job_index])

                    if sink is None:
                        results[job_index] = scaled
                    else:
                        sink(job_index, jobs[job_index][0], scaled)

                    completed.add(job_index)
                    processed_now += 1

                    # ↓ Checkpoint after every single job for safe resumption.
                    self._write_checkpoint(
                        {
                            'version': self._CHECKPOINT_VERSION,
                            'spec': {
                                'n': self.n,
                                'sfx': self.sfx,
                                'bucket_size': bucket_size,
                                'jobs': [job_id for job_id, _ in jobs],
                            },
                            'completed': sorted(completed),
                            'cumulative_reuse_hits': cumulative_hits,
                            'cumulative_elapsed': cumulative_elapsed + (time.monotonic() - started),
                        }
                    )

                    if fail_after is not None and fail_after > 0 and processed_now >= fail_after:
                        interrupted = True
                        break
                if interrupted:
                    break
        except KeyboardInterrupt:
            interrupted = True
        finally:
            elapsed = time.monotonic() - started
            cumulative_elapsed += elapsed

        report.processed = processed_now
        report.skipped = skipped
        report.reuse_hits = hits_now
        report.elapsed = elapsed
        report.cumulative_elapsed = cumulative_elapsed
        report.cumulative_processed = len(completed)
        report.cumulative_reuse_hits = cumulative_hits

        if interrupted or len(completed) < len(images):
            # Durable flush of the final state so an OS crash keeps progress.
            self._write_checkpoint(
                {
                    'version': self._CHECKPOINT_VERSION,
                    'spec': {
                        'n': self.n,
                        'sfx': self.sfx,
                        'bucket_size': bucket_size,
                        'jobs': [job_id for job_id, _ in jobs],
                    },
                    'completed': sorted(completed),
                    'cumulative_reuse_hits': cumulative_hits,
                    'cumulative_elapsed': cumulative_elapsed,
                },
                durable=True,
            )
            raise BatchInterrupted(self.checkpoint, processed_now)

        # ↓ Full batch finished: the checkpoint is no longer needed.
        report.completed = True
        try:
            os.remove(self.checkpoint)
        except FileNotFoundError:
            pass
        return report

    def _bind_scaler(self) -> Callable[[list[list[list[int]]]], list[list[list[int]]]]:
        """Return the single-image scaler selected by ``n`` and ``sfx``."""
        if self.sfx:
            return scale3xsfx if self.n == 3 else scale2xsfx
        return scale3x if self.n == 3 else scale2x


def process_batch(
    images: list[list[list[int]]],
    n: Literal[2, 3],
    sfx: bool,
    bucket_size: int = 0,
    checkpoint: str | os.PathLike = 'scalenx_batch.json',
    resume: bool = False,
    sink: Callable[[int, str, list[list[list[int]]]], None] | None = None,
) -> BatchReport:
    """One-shot convenience wrapper around :class:`BatchScheduler`.

    Equivalent to constructing a scheduler with the same arguments and
    calling :meth:`~BatchScheduler.run`. See that class for the semantics of
    bucketing, reuse hits and resumption.
    """
    scheduler = BatchScheduler(n=n, sfx=sfx, bucket_size=bucket_size, checkpoint=checkpoint, resume=resume, sink=sink)
    return scheduler.run(images)
