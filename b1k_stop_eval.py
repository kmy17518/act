"""Stop study metrics shared by training validation (b1k_training), the offline replay (scripts/b1k/eval_stop_offline.py)
and the completion verifier (scripts/b1k/stop_verifier.py). NumPy only."""

import numpy as np

FPS = 30
FALSE_STOP_BUDGET = 0.02  # the largest false-stop rate a tuned setting may have


def auroc(scores, labels):
    """Area under the ROC curve of `scores` for binary `labels` (Mann-Whitney U with average ranks for ties); NaN
    unless both classes occur."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if len(unique) != len(scores):  # ties share their average rank
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def stop_signal_metrics(predictions, targets, is_pad):
    """Validation metrics of the stop / progress columns of zero-latent predictions (N, T, 2) against their targets,
    both in network units [-1, 1]: val/stop_auroc over every chunk step (the padded ones included: they are the
    positives of an episode cut at T_end), val/stop_auroc_valid over the unpadded steps (positives there exist only
    with settle frames), val/stop_acc at the 0 threshold, val/stop_positive_rate, and val/progress_mae in progress
    units [0, 1]."""
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    is_pad = np.asarray(is_pad, dtype=bool)
    stop_label = targets[..., 0] > 0
    valid = ~is_pad
    return {'val/stop_auroc': auroc(predictions[..., 0], stop_label),
            'val/stop_auroc_valid': auroc(predictions[..., 0][valid], stop_label[valid]),
            'val/stop_acc': float(((predictions[..., 0] > 0) == stop_label).mean()),
            'val/stop_positive_rate': float(stop_label.mean()),
            'val/progress_mae': float(np.abs(predictions[..., 1] - targets[..., 1]).mean() / 2)}


def first_run_end(flags, steps):
    """Index of the last element of the first run of `steps` consecutive True values in `flags`, or None."""
    run = 0
    for index, flag in enumerate(flags):
        run = run + 1 if flag else 0
        if run >= steps:
            return index
    return None


def method_metrics(done_steps, program_lengths, lengths, after_s=5.0, tolerance_s=0.5, fps=FPS):
    """Episode-level stop metrics from per-episode DONE steps (frame indices; None: never declared).

    median_abs_dt_s / median_dt_s / p90_abs_dt_s: |t_done - T_end| (signed for median_dt_s) in seconds over the
    episodes with a DONE; false_stop_rate: DONE before T_end (false_stop_rate_tol: more than `tolerance_s` before);
    missed_stop_rate: no DONE within the recorded frames after T_end; missed_within_after_s_rate: no DONE within
    `after_s` after T_end, over the episodes whose recording lasts that long (episodes_allowing_after_s);
    stop_latency_s_median: t_done - T_end over the DONEs at or after T_end."""
    done = np.array([np.nan if t is None else t for t in done_steps], dtype=np.float64)
    program = np.asarray(program_lengths, dtype=np.float64)
    length = np.asarray(lengths, dtype=np.float64)
    fired = ~np.isnan(done)
    delta = (done - program) / fps
    window = np.minimum(program + after_s * fps, length - 1)  # the latest step a timely DONE can be observed at
    allows = length - 1 >= program + after_s * fps
    timely = fired & (done <= window)
    return {'episodes': int(len(done)), 'done_rate': float(fired.mean()),
            'median_abs_dt_s': float(np.nanmedian(np.abs(delta))) if fired.any() else None,
            'median_dt_s': float(np.nanmedian(delta)) if fired.any() else None,
            'p90_abs_dt_s': float(np.nanpercentile(np.abs(delta), 90)) if fired.any() else None,
            'false_stop_rate': float((fired & (done < program)).mean()),
            'false_stop_rate_tol': float((fired & (done < program - tolerance_s * fps)).mean()),
            'missed_stop_rate': float((~fired | (done > length - 1)).mean()),
            'missed_within_after_s_rate': float((~timely)[allows].mean()) if allows.any() else None,
            'episodes_allowing_after_s': int(allows.sum()),
            'stop_latency_s_median': (float(np.nanmedian(delta[fired & (done >= program)]))
                                      if (fired & (done >= program)).any() else None)}


def choose_setting(grid, budget=FALSE_STOP_BUDGET):
    """The fair setting of a tuning grid (dicts with method_metrics keys): at most `budget` false stops and the
    smallest median |dt| (ties: fewer missed stops); without any such setting, the fewest false stops."""
    within = [entry for entry in grid if entry['false_stop_rate'] <= budget and entry['median_abs_dt_s'] is not None]
    if within:
        return min(within, key=lambda entry: (entry['median_abs_dt_s'], entry['missed_stop_rate']))
    return min(grid, key=lambda entry: (entry['false_stop_rate'], entry['missed_stop_rate']))


def gate_proposals(stop, progress, stop_threshold, progress_threshold, hold_steps):
    """Boolean per-step array: the steps at which a StopGate with these settings would be at or past its hold count
    (its DONE proposals, unlatched), from per-step stop / progress signals."""
    active = (np.asarray(stop) >= stop_threshold) & (np.asarray(progress) >= progress_threshold)
    count = 0
    proposals = np.zeros(len(active), dtype=bool)
    for index, flag in enumerate(active):
        count = count + 1 if flag else 0
        proposals[index] = count >= hold_steps
    return proposals


def summary_line(name, metrics):
    """One readable line of a method's metrics."""
    def seconds(value):
        return 'n/a' if value is None else f'{value:.2f} s'
    return (f"{name}: median |dt| {seconds(metrics.get('median_abs_dt_s'))}, false stops {metrics['false_stop_rate'] * 100:.1f}% "
            f"({metrics['false_stop_rate_tol'] * 100:.1f}% beyond 0.5 s), missed {metrics['missed_stop_rate'] * 100:.1f}%, "
            f"DONE in {metrics['done_rate'] * 100:.0f}% of {metrics['episodes']} episodes"
            + (f", latency {seconds(metrics['stop_latency_s_median'])}" if metrics.get('stop_latency_s_median') is not None else ''))
